from __future__ import annotations

import pandas as pd

from scripts.review_response_analyses import method_ranks


def test_complete_case_drops_cases_with_any_missing_method():
    frame = pd.DataFrame([
        {"case_id": "complete", "method": "a", "status": "ok", "nrmse": 1.0},
        {"case_id": "complete", "method": "b", "status": "ok", "nrmse": 2.0},
        {"case_id": "partial", "method": "a", "status": "ok", "nrmse": 1.0},
        {"case_id": "partial", "method": "b", "status": "timeout", "nrmse": float("nan")},
    ])

    ranks = method_ranks(frame, "complete_case")
    assert ranks.to_dict() == {"a": 1.0, "b": 2.0}
