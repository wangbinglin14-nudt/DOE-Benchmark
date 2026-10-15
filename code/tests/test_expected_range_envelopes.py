"""Regression tests for output ``expected_range`` metadata envelopes."""

from __future__ import annotations

import itertools

import numpy as np
import pytest

import doe_testbed as db


ALL_CASE_IDS = [meta.case_id for meta in db.list_cases()]


def _envelope_probe_points(case) -> np.ndarray:
    """Return deterministic physical points that probe each declared input box.

    For cases up to 12 dimensions, all lower/upper corners are cheap enough and
    catch the metadata drift patterns seen in the audit.  Higher-dimensional
    scalable cases use a midpoint plus one-factor-at-a-time lower/upper probes
    to avoid exponential blow-up while still checking every declared bound.
    """
    bounds = [(var.lower, var.upper) for var in case.meta.input_variables]
    if len(bounds) <= 12:
        return np.array(list(itertools.product(*bounds)), dtype=float)

    midpoint = np.array([(lo + hi) / 2.0 for lo, hi in bounds], dtype=float)
    points = [midpoint.copy()]
    for index, (lower, upper) in enumerate(bounds):
        for value in (lower, upper):
            point = midpoint.copy()
            point[index] = value
            points.append(point)
    return np.array(points, dtype=float)


@pytest.mark.parametrize("case_id", ALL_CASE_IDS)
def test_expected_ranges_cover_input_envelope_probes(case_id):
    case = db.get_case(case_id)
    X_phys = _envelope_probe_points(case)
    Y = case.evaluate(case.normalize(X_phys))

    assert np.all(np.isfinite(Y)), f"{case_id} produced non-finite envelope output"

    failures = []
    for index, output in enumerate(case.meta.output_variables):
        if output.expected_range is None:
            continue
        lower, upper = output.expected_range
        values = Y[:, index]
        observed_lower = float(np.nanmin(values))
        observed_upper = float(np.nanmax(values))
        if observed_lower < lower or observed_upper > upper:
            failures.append(
                f"{output.name}: observed [{observed_lower:.6g}, {observed_upper:.6g}] "
                f"outside expected_range [{lower:.6g}, {upper:.6g}]"
            )

    assert not failures, f"{case_id} expected_range envelope mismatch: {failures}"
