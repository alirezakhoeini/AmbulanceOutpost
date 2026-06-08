from pathlib import Path
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

assignments = pd.read_csv(OUT / "baseline_assignments_all.csv")
summary = pd.read_csv(OUT / "baseline_frontier.csv")

def weighted_percentile(values, weights, q):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    order = np.argsort(values)
    values = values[order]
    weights = weights[order]

    cumulative = np.cumsum(weights)
    cutoff = q * weights.sum()

    return values[np.searchsorted(cumulative, cutoff)]

def weighted_gini(values, weights):
    values = np.asarray(values, dtype=float)
    weights = np.asarray(weights, dtype=float)

    mean = np.average(values, weights=weights)
    if mean == 0:
        return 0.0

    numerator = 0.0
    for i in range(len(values)):
        for j in range(len(values)):
            numerator += weights[i] * weights[j] * abs(values[i] - values[j])

    denominator = 2 * mean * (weights.sum() ** 2)
    return numerator / denominator

equity_rows = []

for maxfac, group in assignments.groupby("MaxFac"):
    distances = group["distance_km"].to_numpy()
    weights = group["population"].to_numpy()

    avg = np.average(distances, weights=weights)
    std = np.sqrt(np.average((distances - avg) ** 2, weights=weights))

    equity_rows.append({
        "MaxFac": maxfac,
        "P90Distance_km": round(weighted_percentile(distances, weights, 0.90), 4),
        "WeightedCV": round(std / avg, 4),
        "WeightedGini": round(weighted_gini(distances, weights), 4)
    })

equity = pd.DataFrame(equity_rows)
combined = summary.merge(equity, on="MaxFac")

combined.to_csv(OUT / "equity_frontier.csv", index=False)

print("\nEquity frontier:")
print(combined)
print("\nSaved:")
print(OUT / "equity_frontier.csv")