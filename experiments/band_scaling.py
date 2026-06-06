"""No-trade band width against the Whalley-Wilmott power law.

Singular perturbation analysis predicts the no-trade band around the
model delta widens as the cube root of the proportional cost rate. The
exponent comes from balancing trading cost against tracking risk, so it
survives the change from exponential utility to the CVaR objective the
policies train on; only the prefactor moves. This script reads the
at-the-money inventory probes stored by the frontier runs, measures the
band width as the inventory interval the policy leaves in place, and
fits the log-log slope against cost. The width depends on the hold
threshold because a smooth network never sits exactly still, so the
slope is reported at several thresholds and the conclusion stands only
if it is threshold-robust. With five cost points spanning just over a
decade, a slope within [0.27, 0.40] is consistent with the one-third
law; tighter claims are not defensible at this lever arm.

    uv run python experiments/band_scaling.py
"""

import math
from collections import defaultdict

from deephedging.experiment import load_records

RESULTS = "experiments/hedging_frontier.jsonl"
HOLD_THRESHOLDS = (0.01, 0.02, 0.05)
SLOPE_TARGET = 1.0 / 3.0
SLOPE_BOUNDS = (0.27, 0.40)


def band_width(inventory: list[float], response: list[float], threshold: float) -> float:
    """Measures the inventory interval the policy leaves in place.

    Args:
        inventory: Held positions probed, in increasing order.
        response: Policy output at each probed position.
        threshold: Largest trade still counted as holding.

    Returns:
        Width of the no-trade interval, zero when the policy always
        trades by more than the threshold.
    """
    held = [p for p, a in zip(inventory, response, strict=True) if abs(a - p) < threshold]
    if not held:
        return 0.0
    return max(held) - min(held)


def fit_slope(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Fits log width against log cost by ordinary least squares.

    Args:
        points: Pairs of cost rate and measured band width.

    Returns:
        Slope, standard error of the slope, and coefficient of
        determination.

    Raises:
        ValueError: If fewer than three usable points remain after
            dropping zero widths, which cannot identify a slope.
    """
    usable = [(math.log(c), math.log(w)) for c, w in points if w > 0.0]
    if len(usable) < 3:
        raise ValueError(f"need at least 3 positive widths, got {len(usable)}")
    n = len(usable)
    mean_x = sum(x for x, _ in usable) / n
    mean_y = sum(y for _, y in usable) / n
    s_xx = sum((x - mean_x) ** 2 for x, _ in usable)
    s_xy = sum((x - mean_x) * (y - mean_y) for x, y in usable)
    slope = s_xy / s_xx
    intercept = mean_y - slope * mean_x
    residual = sum((y - intercept - slope * x) ** 2 for x, y in usable)
    total = sum((y - mean_y) ** 2 for _, y in usable)
    variance = residual / (n - 2) / s_xx if n > 2 else float("inf")
    r_squared = 1.0 - residual / total if total > 0.0 else float("nan")
    return slope, math.sqrt(variance), r_squared


def main() -> None:
    """Fits and prints the band-width scaling per risk level."""
    probes: dict[tuple[float, float], list[tuple[float, float]]] = defaultdict(list)
    for record in load_records(RESULTS):
        setup = record.setup
        probe = setup.get("band_probe")
        cost_rate = setup.get("cost_rate")
        alpha = setup.get("alpha")
        if probe is None or not isinstance(cost_rate, float) or cost_rate == 0.0:
            continue
        assert isinstance(probe, dict) and isinstance(alpha, float)
        for threshold in HOLD_THRESHOLDS:
            width = band_width(probe["inventory"], probe["response"], threshold)
            probes[(alpha, threshold)].append((cost_rate, width))

    if not probes:
        print(f"no band probes found in {RESULTS}; run the frontier experiment first")
        return

    lower, upper = SLOPE_BOUNDS
    print(f"target slope {SLOPE_TARGET:.3f}, consistent if within [{lower:.2f}, {upper:.2f}]")
    for (alpha, threshold), points in sorted(probes.items()):
        try:
            slope, stderr, r_squared = fit_slope(points)
        except ValueError as error:
            print(f"alpha {alpha:4.2f} hold {threshold:4.2f}  unusable ({error})")
            continue
        verdict = "consistent" if lower <= slope <= upper else "inconsistent"
        print(
            f"alpha {alpha:4.2f} hold {threshold:4.2f}  "
            f"slope {slope:6.3f} +/- {stderr:5.3f}  r2 {r_squared:5.3f}  "
            f"n {len(points):3d}  {verdict}"
        )


if __name__ == "__main__":
    main()
