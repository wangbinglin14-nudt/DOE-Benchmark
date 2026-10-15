"""Lock the canonical hyperparameter table to the actual method pool."""

from __future__ import annotations

import pytest

from benchmark.hyperparameters import (
    HYPERPARAMETERS,
    assert_hyperparameters_consistent,
)
from benchmark.methods import bo_method_pool, full_method_pool


def test_hyperparameter_table_consistent_with_full_pool():
    """Every method in full_method_pool must be in HYPERPARAMETERS with
    matching kwargs (or method_kwargs for sequential methods)."""
    assert_hyperparameters_consistent()


def test_hyperparameter_table_has_no_extra_entries_beyond_full_pool_or_bo():
    """The canonical table is the union of ``full_method_pool`` and the
    BoTorch BO methods.  No spurious entries should accumulate."""
    pool_names = {m.name for m in full_method_pool()}
    bo_names = {"botorch_qei", "botorch_saasbo"}
    documented = set(HYPERPARAMETERS)
    extras = documented - pool_names - bo_names
    assert not extras, (
        f"hyperparameters.py documents methods not in full_method_pool "
        f"or the BoTorch BO list: {extras}"
    )


def test_each_static_method_entry_records_sampler_and_surrogate():
    """Static-design methods must report both sampler and surrogate."""
    sequential_keys = {"ego_ei", "max_variance", "akmcs",
                       "botorch_qei", "botorch_saasbo"}
    for name, entry in HYPERPARAMETERS.items():
        if name in sequential_keys:
            continue
        assert "sampler" in entry, f"missing sampler for {name!r}"
        assert "surrogate" in entry, f"missing surrogate for {name!r}"
        assert "surrogate_kwargs" in entry, (
            f"missing surrogate_kwargs for {name!r} — even an empty dict "
            "must be present so analysis code can introspect it."
        )


def test_each_sequential_method_entry_records_method_kwargs():
    sequential_keys = {"ego_ei", "max_variance", "akmcs",
                       "botorch_qei", "botorch_saasbo"}
    for name in sequential_keys:
        assert name in HYPERPARAMETERS, f"missing {name!r}"
        entry = HYPERPARAMETERS[name]
        assert "method_kwargs" in entry, (
            f"sequential method {name!r} must record method_kwargs"
        )
        assert isinstance(entry["method_kwargs"], dict)


# ─── Provenance lock-in (Reviewer-Critique C3) ──────────────────────────
_VALID_PROVENANCE_TAGS = {
    "library_default",
    "library_default+modified",
    "matched_for_ablation",
    "tuned_for_benchmark",
    "paper_recommended",
    "benchmark_internal",
}


def test_every_entry_has_provenance_string():
    """Every method in HYPERPARAMETERS must record a ``_provenance`` field.

    The field exists so a reviewer can spot-check ``did you cherry-pick
    these hyperparameters?'' without reading the markdown protocol.  The
    string must start with one of the six convention tags listed in
    the module docstring; this ensures provenance categories don't
    silently drift over time.
    """
    for name, entry in HYPERPARAMETERS.items():
        assert "_provenance" in entry, (
            f"method {name!r} is missing the _provenance field; every "
            "entry in HYPERPARAMETERS must explain where its values "
            "came from (see the module docstring for the six "
            "convention tags)."
        )
        prov = entry["_provenance"]
        assert isinstance(prov, str) and prov.strip(), (
            f"method {name!r} has a non-string or empty _provenance "
            f"value: {prov!r}."
        )
        tag = prov.split(":", 1)[0].strip()
        assert tag in _VALID_PROVENANCE_TAGS, (
            f"method {name!r} provenance starts with {tag!r}, which is "
            f"not in the canonical tag set {sorted(_VALID_PROVENANCE_TAGS)}.  "
            "Either pick an existing tag or add a new one to the "
            "module docstring AND this test."
        )


# --- BoTorch pool lock-in (guards the SAASBO NUTS defaults) -------------
def test_botorch_pool_is_covered_and_matches_live_attributes():
    """The optional BoTorch pool must be covered by ``HYPERPARAMETERS``.

    Every value documented in a ``method_kwargs`` entry must equal the
    value carried by the live method object built by ``bo_method_pool``.

    Regression guard: ``botorch_saasbo`` NUTS settings were once
    documented as ``warmup_steps=128 / num_samples=64 / thinning=16``,
    which is *not* what the archived sweep instantiated -- the
    constructor defaults (4/4/1) were used because ``bo_method_pool``
    passes only ``n_init_frac`` and ``n_candidates``.  The earlier
    consistency check walked only ``full_method_pool()``, so that drift
    was silent.  This test walks the optional pool explicitly.
    """
    pool = list(bo_method_pool())
    if not pool:
        pytest.skip("BoTorch is not installed in this environment")

    names = {m.name for m in pool}
    missing = names - set(HYPERPARAMETERS)
    assert not missing, (
        f"methods in bo_method_pool() are missing from HYPERPARAMETERS: "
        f"{sorted(missing)}"
    )

    for method in pool:
        canonical = HYPERPARAMETERS[method.name]
        assert "method_kwargs" in canonical, (
            f"BoTorch method {method.name!r} must document method_kwargs"
        )
        for key, documented in canonical["method_kwargs"].items():
            actual = getattr(method, key, "<missing>")
            assert actual == documented, (
                f"{method.name}.{key} drift: the live method carries "
                f"{actual!r} but hyperparameters.py documents "
                f"{documented!r}.  Either fix the table or state the "
                f"deviation in _provenance."
            )


def test_botorch_method_source_hashes_are_pinned_by_the_manifest():
    """``bo_methods.py`` is byte-pinned by RUN_MANIFEST.json.

    Editing that module after the sweep invalidates the archived
    ``method_source_sha256`` values, so this test fails loudly if the
    on-disk bytes diverge from the manifest.  Skipped when the manifest
    or the archived results are absent (e.g. a source-only checkout).
    """
    import hashlib
    import json
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = (
        repo_root / "benchmark" / "results" / "full_benchmark"
        / "RUN_MANIFEST.json"
    )
    if not manifest_path.exists():
        pytest.skip("RUN_MANIFEST.json not present in this checkout")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = manifest.get("method_sources", {}).get("methods", [])
    pinned = {}
    for entry in sources:
        for component in entry.get("components", []):
            path = component.get("path")
            sha = component.get("sha256")
            if path and sha:
                pinned.setdefault(path, sha)
    if not pinned:
        pytest.skip("manifest pins no method sources")

    for rel_path, expected in sorted(pinned.items()):
        target = repo_root / rel_path
        if not target.exists():
            pytest.skip(f"{rel_path} not present in this checkout")
        actual = hashlib.sha256(target.read_bytes()).hexdigest()
        assert actual == expected, (
            f"{rel_path} does not match the archived run manifest "
            f"(expected {expected}, found {actual}).  The archived "
            f"results were produced by the pinned bytes; do not edit "
            f"this file without regenerating the manifest."
        )


def test_documented_hyperparameters_match_the_frozen_archive_configuration():
    """``HYPERPARAMETERS`` must describe the run that produced the archive.

    ``RUN_MANIFEST.json`` records the *resolved* constructor configuration of
    every sampler, surrogate and method component the sweep instantiated, which
    makes it the ground truth for what actually ran.  It is stronger evidence
    than reading the current source: ``benchmark/methods.py`` is not byte-pinned,
    so a constructor argument could in principle change without invalidating any
    archived hash, whereas the manifest records the arguments themselves.

    Regression guard: ``botorch_saasbo`` once documented ``warmup_steps=128 /
    num_samples=64 / thinning=16`` while the frozen manifest records 4/4/1, so
    the article's hyperparameter table described a configuration that never ran.
    Comparing the registry against the frozen record key by key keeps that class
    of drift from reappearing silently.
    """
    import json
    from pathlib import Path

    repo_root = Path(__file__).resolve().parents[2]
    manifest_path = (
        repo_root / "benchmark" / "results" / "full_benchmark"
        / "RUN_MANIFEST.json"
    )
    if not manifest_path.exists():
        pytest.skip("RUN_MANIFEST.json not present in this checkout")

    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    sources = manifest.get("method_sources", {}).get("methods", [])
    if not sources:
        pytest.skip("manifest records no method sources")

    def slot_for(qualified_class: str) -> str:
        module = qualified_class.rsplit(".", 1)[0]
        if module == "benchmark.samplers":
            return "sampler_kwargs"
        if module == "benchmark.surrogates":
            return "surrogate_kwargs"
        return "method_kwargs"

    def equivalent(documented, recorded) -> bool:
        # JSON has no tuple type, so list/tuple pairs compare element-wise.
        if isinstance(documented, (list, tuple)) and isinstance(recorded, (list, tuple)):
            return list(documented) == list(recorded)
        if isinstance(documented, bool) or isinstance(recorded, bool):
            return documented == recorded
        if isinstance(documented, (int, float)) and isinstance(recorded, (int, float)):
            return abs(float(documented) - float(recorded)) <= 1e-12
        return documented == recorded

    checked = 0
    problems: list[str] = []
    for entry in sources:
        name = entry.get("method")
        documented_entry = HYPERPARAMETERS.get(name)
        if documented_entry is None:
            problems.append(f"{name}: absent from HYPERPARAMETERS")
            continue
        for component in entry.get("components", []):
            slot = slot_for(component.get("class", ""))
            documented = dict(documented_entry.get(slot) or {})
            recorded = {
                key: value
                for key, value in (component.get("config") or {}).items()
                if key != "name"
            }
            for key, value in sorted(recorded.items(), key=lambda item: item[0]):
                if key not in documented:
                    problems.append(
                        f"{name}.{slot}.{key}: the manifest records {value!r} "
                        f"but the registry documents no such key"
                    )
                elif not equivalent(documented[key], value):
                    problems.append(
                        f"{name}.{slot}.{key}: the manifest records {value!r} "
                        f"but the registry documents {documented[key]!r}"
                    )
                else:
                    checked += 1

    assert not problems, (
        "hyperparameters.py no longer matches the frozen run manifest:\n  "
        + "\n  ".join(problems)
    )
    assert checked >= 40, (
        f"the archive comparison only matched {checked} keys; the manifest or "
        f"the registry layout probably changed, which would quietly weaken "
        f"this guard."
    )
