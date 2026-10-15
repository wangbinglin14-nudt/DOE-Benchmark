"""
doe_testbed.registry.registry
================================
Global case registry (singleton) and public query API.
"""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Type

from doe_testbed.base.case import BaseCase
from doe_testbed.base.meta import CaseMeta

# ─────────────────────────────────────────────────────────────────────────────
# Global registry (singleton dict)
# ─────────────────────────────────────────────────────────────────────────────

_REGISTRY: Dict[str, Type[BaseCase]] = {}


def register_case(cls: Type[BaseCase]) -> Type[BaseCase]:
    """Case registration decorator.

    Apply to a BaseCase subclass; the case is auto-registered when the
    module is imported.

    Usage
    -----
    @register_case
    class MyCase(BaseCase):
        meta = CaseMeta(case_id="my_case", ...)
        ...
    """
    if not hasattr(cls, "meta") or not isinstance(cls.meta, CaseMeta):
        raise TypeError(
            f"Case class {cls.__name__} must define a 'meta' class attribute "
            f"(CaseMeta instance)."
        )
    cid = cls.meta.case_id
    if cid in _REGISTRY:
        # Allow re-registration of the same case_id (e.g. hot-reload)
        pass
    _REGISTRY[cid] = cls
    return cls


# ─────────────────────────────────────────────────────────────────────────────
# Query API
# ─────────────────────────────────────────────────────────────────────────────


def get_case(case_id: str) -> BaseCase:
    """Retrieve a case instance by its case_id.

    Parameters
    ----------
    case_id : str
        Unique case identifier, e.g. 'airfoil_stall_2d'.

    Returns
    -------
    BaseCase
        An instance of the registered case subclass.

    Raises
    ------
    KeyError
        If case_id is not registered; the error message lists all
        available case IDs.

    Examples
    --------
    >>> case = get_case("airfoil_stall_2d")
    """
    if case_id not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise KeyError(
            f"Case '{case_id}' not found.\n"
            f"Registered cases ({len(available)} total):\n"
            + "\n".join(f"  - {k}" for k in available)
        )
    return _REGISTRY[case_id]()


def get_case_meta(case_id: str) -> CaseMeta:
    """Retrieve a case's static metadata without instantiating it.

    Equivalent to ``get_case(case_id).meta`` but skips the BaseCase
    constructor.  Two benefits over instantiating:

    1. **Modestly faster** (~30 % per call, microseconds either way) —
       worth using on hot paths that iterate many cases (e.g. the
       ``/api/cases`` listing endpoint that reads metadata for all 34
       cases on every request).
    2. **No constructor side-effects** — guaranteed read-only access to
       the canonical class-level metadata.

    Parameters
    ----------
    case_id : str
        Unique case identifier.

    Returns
    -------
    CaseMeta
        The class-level metadata object (do not mutate).

    Raises
    ------
    KeyError
        If case_id is not registered.

    Examples
    --------
    >>> meta = get_case_meta("airfoil_stall_2d")
    >>> meta.dim_input
    4
    """
    if case_id not in _REGISTRY:
        available = sorted(_REGISTRY.keys())
        raise KeyError(
            f"Case '{case_id}' not found.\n"
            f"Registered cases ({len(available)} total):\n"
            + "\n".join(f"  - {k}" for k in available)
        )
    return _REGISTRY[case_id].meta


def list_cases(
    domain: Optional[str] = None,
    tags: Optional[List[str]] = None,
    applicable_scenarios: Optional[List[str]] = None,
    dim_input_range: Optional[Tuple[int, int]] = None,
    has_failure_boundary: Optional[bool] = None,
    response_topology: Optional[List[str]] = None,
    max_eval_time_ms: Optional[float] = None,
    supports_multifidelity: Optional[bool] = None,
    cost_level: Optional[str] = None,
    return_ids: bool = False,
) -> List:
    """Filter and return registered cases matching the given criteria.

    Parameters
    ----------
    domain : str, optional
        Discipline filter, e.g. 'aerodynamics'.
    tags : list of str, optional
        Required tags (AND logic).
    applicable_scenarios : list of str, optional
        Required VS scenarios (AND logic), e.g. ['VS-G'].
    dim_input_range : (int, int), optional
        Input dimension range (min_d, max_d).
    has_failure_boundary : bool, optional
        Whether a failure boundary is defined.
    response_topology : list of str, optional
        Required response surface topology traits (OR logic).
    max_eval_time_ms : float, optional
        Maximum single-evaluation time in milliseconds.
    supports_multifidelity : bool, optional
        Whether multi-fidelity support is required.
    cost_level : str, optional
        Cost level filter: 'L0' | 'L1' | 'L2' | 'L3'.
    return_ids : bool
        If True, return case_id strings instead of CaseMeta objects.

    Returns
    -------
    list of CaseMeta or list of str
        Matching metadata objects (or case_id strings when return_ids=True).
    """
    results = []
    for cid, cls in _REGISTRY.items():
        m: CaseMeta = cls.meta

        if domain is not None and m.domain != domain:
            continue
        if tags is not None and not all(t in m.tags for t in tags):
            continue
        if applicable_scenarios is not None:
            if not all(s in m.applicable_scenarios for s in applicable_scenarios):
                continue
        if dim_input_range is not None:
            lo, hi = dim_input_range
            if not (lo <= m.dim_input <= hi):
                continue
        if has_failure_boundary is not None:
            if m.has_failure_boundary != has_failure_boundary:
                continue
        if response_topology is not None:
            if not any(t in m.response_topology for t in response_topology):
                continue
        if max_eval_time_ms is not None:
            t_ms = m.typical_eval_time_ms.get("default", float("inf"))
            if t_ms > max_eval_time_ms:
                continue
        if supports_multifidelity is not None:
            if m.supports_multifidelity != supports_multifidelity:
                continue
        if cost_level is not None and m.cost_level != cost_level:
            continue

        results.append(cid if return_ids else m)

    if return_ids:
        return sorted(results)
    return sorted(results, key=lambda m: m.case_id)


def describe_case(case_id: str) -> str:
    """Print and return the full description of a case."""
    return get_case(case_id).describe()


def get_minimum_covering_set() -> List[BaseCase]:
    """Return a minimum covering set of cases spanning all 8 VS scenarios.

    Returns
    -------
    list of BaseCase
        Instances covering all VS-A through VS-H (greedy algorithm).
    """
    all_scenarios = {"VS-A", "VS-B", "VS-C", "VS-D", "VS-E", "VS-F", "VS-G", "VS-H"}
    covered: set[str] = set()
    selected = []
    # Greedy: pick the case covering the most uncovered scenarios
    candidates = sorted(
        _REGISTRY.items(),
        key=lambda kv: -len(set(kv[1].meta.applicable_scenarios)),
    )
    for _, cls in candidates:
        new = set(cls.meta.applicable_scenarios) - covered
        if new:
            selected.append(cls())
            covered |= new
        if covered >= all_scenarios:
            break
    return selected


def print_cases_table(
    metas: Optional[List[CaseMeta]] = None,
    show_scenarios: bool = True,
) -> None:
    """Print a formatted table of cases.

    Parameters
    ----------
    metas : list of CaseMeta, optional
        Metadata to print; defaults to all registered cases.
    """
    if metas is None:
        metas = list_cases()
    if not metas:
        print("(no matching cases)")
        return

    col_w = [14, 30, 14, 6, 6, 8, 20]
    header = (
        f"{'case_id':<{col_w[0]}}  "
        f"{'name':<{col_w[1]}}  "
        f"{'domain':<{col_w[2]}}  "
        f"{'d_in':<{col_w[3]}}  "
        f"{'d_out':<{col_w[4]}}  "
        f"{'time':>{col_w[5]}}  "
    )
    if show_scenarios:
        header += f"{'VS scenarios':<{col_w[6]}}"
    sep = "─" * (sum(col_w) + 12)
    print(sep)
    print(header)
    print(sep)
    for m in metas:
        t_ms = m.typical_eval_time_ms.get("default", 0)
        t_str = f"{t_ms:.1f} ms" if t_ms < 1000 else f"{t_ms/1000:.1f} s"
        row = (
            f"{m.case_id:<{col_w[0]}}  "
            f"{m.name[:col_w[1]]:<{col_w[1]}}  "
            f"{m.domain[:col_w[2]]:<{col_w[2]}}  "
            f"{m.dim_input:<{col_w[3]}}  "
            f"{m.dim_output:<{col_w[4]}}  "
            f"{t_str:>{col_w[5]}}  "
        )
        if show_scenarios:
            scns = " ".join(m.applicable_scenarios[:4])
            row += f"{scns:<{col_w[6]}}"
        print(row)
    print(sep)
    print(f"Total: {len(metas)} cases")
