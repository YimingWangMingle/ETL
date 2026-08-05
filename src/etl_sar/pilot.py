from __future__ import annotations

from collections.abc import Mapping
from numbers import Real
from typing import Any


LOCAL_PROTOCOL = "myosuite-2.12.2-default"
EXPECTED_TARGETS = {
    "hand": "myoHandReorient100-v0",
    "leg": "myoLegRoughTerrainWalk-v0",
}
TARGET_STEPS = 20_000
EVALUATION_EPISODES = 10


def _is_real(value: Any) -> bool:
    return isinstance(value, Real) and not isinstance(value, bool)


def _local_domain_result(
    domain: str,
    comparison: Mapping[str, Any],
) -> dict[str, Any]:
    return_delta = comparison.get("mean_return_delta")
    success_delta = comparison.get("success_rate_delta")
    checks = {
        "expected_target": comparison.get("environment_id")
        == EXPECTED_TARGETS[domain],
        "matched_budget": comparison.get("environment_steps") == TARGET_STEPS,
        "matched_episodes": comparison.get("episodes") == EVALUATION_EPISODES,
        "return_improved": _is_real(return_delta) and return_delta > 0,
        "success_not_worse": _is_real(success_delta) and success_delta >= 0,
    }
    return {
        **dict(comparison),
        "checks": checks,
        "passes": all(checks.values()),
    }


def _legacy_summary(
    legacy_reference: Mapping[str, Any] | None,
    local_domains: Mapping[str, Mapping[str, Any]],
) -> dict[str, Any]:
    if legacy_reference is None:
        return {"available": False}

    source_domains = legacy_reference.get("domains", {})
    if not isinstance(source_domains, Mapping):
        source_domains = {}
    domains: dict[str, dict[str, Any]] = {}
    for domain, expected_target in EXPECTED_TARGETS.items():
        source = source_domains.get(domain, {})
        entry = dict(source) if isinstance(source, Mapping) else {}
        checks = {
            "environment_id": entry.get("environment_id") == expected_target,
            "protocol": entry.get("protocol") == LOCAL_PROTOCOL,
            "metric": entry.get("metric") == "mean_return",
        }
        entry["comparable"] = all(checks.values())
        entry["comparability_checks"] = checks
        local_return = local_domains[domain].get("extension_mean_return")
        legacy_value = entry.get("value")
        if (
            entry["comparable"]
            and _is_real(local_return)
            and _is_real(legacy_value)
        ):
            entry["mean_return_delta"] = float(local_return - legacy_value)
        domains[domain] = entry

    return {
        "available": True,
        "method": legacy_reference.get("method", "ETL-Ray"),
        "domains": domains,
    }


def build_pilot_summary(
    hand_comparison: Mapping[str, Any],
    leg_comparison: Mapping[str, Any],
    legacy_reference: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    domains = {
        "hand": _local_domain_result("hand", hand_comparison),
        "leg": _local_domain_result("leg", leg_comparison),
    }
    return {
        "protocol": LOCAL_PROTOCOL,
        "target_steps": TARGET_STEPS,
        "evaluation_episodes": EVALUATION_EPISODES,
        "pilot_positive": all(result["passes"] for result in domains.values()),
        "domains": domains,
        "legacy_reference": _legacy_summary(legacy_reference, domains),
    }
