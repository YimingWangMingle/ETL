from __future__ import annotations

from etl_sar.pilot import build_pilot_summary


def local_comparison(
    env_id: str,
    return_delta: float,
    success_delta: float = 0.0,
) -> dict:
    return {
        "environment_steps": 20_000,
        "episodes": 10,
        "environment_id": env_id,
        "evaluation_seed": 10_007,
        "baseline_mean_return": 1.0,
        "extension_mean_return": 1.0 + return_delta,
        "mean_return_delta": return_delta,
        "success_rate_delta": success_delta,
    }


def test_pilot_is_positive_only_when_both_domains_pass() -> None:
    result = build_pilot_summary(
        local_comparison("myoHandReorient100-v0", 1.0),
        local_comparison("myoLegRoughTerrainWalk-v0", 2.0),
    )
    assert result["pilot_positive"] is True

    failed = build_pilot_summary(
        local_comparison("myoHandReorient100-v0", 1.0),
        local_comparison("myoLegRoughTerrainWalk-v0", -0.1),
    )
    assert failed["pilot_positive"] is False
    assert failed["domains"]["leg"]["passes"] is False


def test_pilot_rejects_wrong_local_protocol() -> None:
    result = build_pilot_summary(
        local_comparison("myoHandReorient100-v0", 1.0),
        local_comparison(
            "myoLegRoughTerrainWalk-v0",
            1.0,
        )
        | {"environment_steps": 19_000},
    )

    assert result["pilot_positive"] is False
    assert result["domains"]["leg"]["checks"]["matched_budget"] is False


def test_legacy_reference_is_not_comparable_when_protocol_differs() -> None:
    legacy = {
        "method": "ETL-Ray",
        "domains": {
            "hand": {
                "environment_id": "different-task",
                "protocol": "legacy",
                "metric": "mean_return",
                "value": 1.0,
            },
            "leg": {
                "environment_id": "different-task",
                "protocol": "legacy",
                "metric": "mean_return",
                "value": 1.0,
            },
        },
    }

    result = build_pilot_summary(
        local_comparison("myoHandReorient100-v0", 1.0),
        local_comparison("myoLegRoughTerrainWalk-v0", 1.0),
        legacy,
    )

    hand = result["legacy_reference"]["domains"]["hand"]
    assert hand["comparable"] is False
    assert "mean_return_delta" not in hand


def test_matching_legacy_reference_gets_extension_delta() -> None:
    legacy = {
        "method": "ETL-Ray",
        "domains": {
            "hand": {
                "environment_id": "myoHandReorient100-v0",
                "protocol": "myosuite-2.12.2-default",
                "metric": "mean_return",
                "value": 0.5,
            },
            "leg": {
                "environment_id": "myoLegRoughTerrainWalk-v0",
                "protocol": "myosuite-2.12.2-default",
                "metric": "mean_return",
                "value": 0.5,
            },
        },
    }

    result = build_pilot_summary(
        local_comparison("myoHandReorient100-v0", 1.0),
        local_comparison("myoLegRoughTerrainWalk-v0", 1.0),
        legacy,
    )

    hand = result["legacy_reference"]["domains"]["hand"]
    assert hand["comparable"] is True
    assert hand["mean_return_delta"] == 1.5


def test_missing_legacy_reference_is_explicitly_unavailable() -> None:
    result = build_pilot_summary(
        local_comparison("myoHandReorient100-v0", 1.0),
        local_comparison("myoLegRoughTerrainWalk-v0", 1.0),
    )

    assert result["legacy_reference"] == {"available": False}
