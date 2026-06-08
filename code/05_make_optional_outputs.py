from pathlib import Path
import itertools
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

# -------------------------------------------------
# Load data
# -------------------------------------------------

demand = pd.read_csv(DATA / "demand_points_sariyer_10.csv")
facilities = pd.read_csv(DATA / "candidate_facilities_sariyer_7.csv")
dist = pd.read_csv(DATA / "distance_matrix_seed_km.csv").set_index("demand_name")

frontier = pd.read_csv(OUT / "equity_frontier.csv")
assignments_all = pd.read_csv(OUT / "baseline_assignments_all.csv")

demand_names = demand["name_en"].tolist()
facility_ids = facilities["facility_id"].tolist()
pop = dict(zip(demand["name_en"], demand["population"]))


# -------------------------------------------------
# Helper functions
# -------------------------------------------------

def solve_for_subset(subset):
    """
    Assign each demand point to the nearest facility in subset.
    """
    assignments = {}

    for d in demand_names:
        nearest = min(subset, key=lambda f: dist.loc[d, f])
        assignments[d] = nearest

    distances = np.array([dist.loc[d, assignments[d]] for d in demand_names], dtype=float)
    weights = np.array([pop[d] for d in demand_names], dtype=float)

    avg = np.average(distances, weights=weights)
    max_d = distances.max()
    covered_3km = np.average(distances <= 3.0, weights=weights)
    covered_4km = np.average(distances <= 4.0, weights=weights)
    covered_5km = np.average(distances <= 5.0, weights=weights)

    return assignments, avg, max_d, covered_3km, covered_4km, covered_5km


def best_solution_for_maxfac(maxfac):
    """
    Exact enumeration because there are only 7 facilities.
    """
    best = None

    for subset in itertools.combinations(facility_ids, maxfac):
        assignments, avg, max_d, cov3, cov4, cov5 = solve_for_subset(subset)

        candidate = {
            "subset": subset,
            "assignments": assignments,
            "avg": avg,
            "max_d": max_d,
            "coverage_3km": cov3,
            "coverage_4km": cov4,
            "coverage_5km": cov5,
        }

        if best is None:
            best = candidate
        elif avg < best["avg"]:
            best = candidate
        elif np.isclose(avg, best["avg"]) and max_d < best["max_d"]:
            best = candidate

    return best


# -------------------------------------------------
# Output 1: sensitivity_by_maxfac.csv
# -------------------------------------------------

sensitivity_rows = []

for maxfac in range(1, len(facility_ids) + 1):
    sol = best_solution_for_maxfac(maxfac)

    sensitivity_rows.append({
        "MaxFac": maxfac,
        "OpenedFacilities": ";".join(sol["subset"]),
        "AvgWeightedDistance_km": round(sol["avg"], 4),
        "MaxDistance_km": round(sol["max_d"], 4),
        "CoverageWithin3km": round(sol["coverage_3km"], 4),
        "CoverageWithin4km": round(sol["coverage_4km"], 4),
        "CoverageWithin5km": round(sol["coverage_5km"], 4),
    })

sensitivity_by_maxfac = pd.DataFrame(sensitivity_rows)
sensitivity_by_maxfac.to_csv(OUT / "sensitivity_by_maxfac.csv", index=False)


# -------------------------------------------------
# Output 2: sensitivity_service_thresholds.csv
# -------------------------------------------------

thresholds = [2, 3, 4, 5, 6, 8, 10]

threshold_rows = []

for maxfac in range(1, len(facility_ids) + 1):
    sol = best_solution_for_maxfac(maxfac)
    assigned_distances = np.array(
        [dist.loc[d, sol["assignments"][d]] for d in demand_names],
        dtype=float
    )
    weights = np.array([pop[d] for d in demand_names], dtype=float)

    for threshold in thresholds:
        coverage = np.average(assigned_distances <= threshold, weights=weights)

        threshold_rows.append({
            "MaxFac": maxfac,
            "Threshold_km": threshold,
            "PopulationCoverage": round(coverage, 4),
            "PopulationCoverage_percent": round(100 * coverage, 2),
            "OpenedFacilities": ";".join(sol["subset"]),
        })

sensitivity_thresholds = pd.DataFrame(threshold_rows)
sensitivity_thresholds.to_csv(OUT / "sensitivity_service_thresholds.csv", index=False)


# -------------------------------------------------
# Output 3: equity_frontier_comparison.png
# -------------------------------------------------

plt.figure(figsize=(8, 5))

plt.plot(
    frontier["MaxFac"],
    frontier["AvgWeightedDistance_km"],
    marker="o",
    label="Average weighted distance"
)

plt.plot(
    frontier["MaxFac"],
    frontier["MaxDistance_km"],
    marker="s",
    label="Maximum distance"
)

if "P90Distance_km" in frontier.columns:
    plt.plot(
        frontier["MaxFac"],
        frontier["P90Distance_km"],
        marker="^",
        label="90th percentile distance"
    )

plt.xlabel("Maximum number of opened facilities")
plt.ylabel("Distance (km)")
plt.title("Efficiency and Equity Frontier Comparison")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(OUT / "equity_frontier_comparison.png", dpi=300)
plt.close()


# -------------------------------------------------
# Output 4: map_p4_seed.png
# -------------------------------------------------

recommended_maxfac = 4
recommended_assignments = assignments_all[
    assignments_all["MaxFac"] == recommended_maxfac
].copy()

opened_facilities = sorted(recommended_assignments["assigned_facility"].unique())

facility_lookup = facilities.set_index("facility_id")
demand_lookup = demand.set_index("name_en")

plt.figure(figsize=(8, 7))

# Plot all demand points
plt.scatter(
    demand["lon"],
    demand["lat"],
    s=demand["population"] / 150,
    alpha=0.6,
    label="Demand neighborhoods"
)

# Plot all candidate facilities
plt.scatter(
    facilities["lon"],
    facilities["lat"],
    marker="x",
    s=80,
    label="Candidate facilities"
)

# Plot opened facilities
opened_df = facilities[facilities["facility_id"].isin(opened_facilities)]
plt.scatter(
    opened_df["lon"],
    opened_df["lat"],
    marker="*",
    s=250,
    label="Opened facilities"
)

# Draw assignment lines
for _, row in recommended_assignments.iterrows():
    d_name = row["demand_name"]
    f_id = row["assigned_facility"]

    d_lat = demand_lookup.loc[d_name, "lat"]
    d_lon = demand_lookup.loc[d_name, "lon"]

    f_lat = facility_lookup.loc[f_id, "lat"]
    f_lon = facility_lookup.loc[f_id, "lon"]

    plt.plot([d_lon, f_lon], [d_lat, f_lat], linewidth=0.8, alpha=0.5)

# Labels for demand points
for _, row in demand.iterrows():
    plt.text(row["lon"], row["lat"], row["name_en"], fontsize=8)

# Labels for opened facilities
for _, row in opened_df.iterrows():
    plt.text(row["lon"], row["lat"], row["facility_id"], fontsize=10, fontweight="bold")

plt.xlabel("Longitude")
plt.ylabel("Latitude")
plt.title("Recommended Ambulance Outpost Assignments (MaxFac = 4)")
plt.grid(True)
plt.legend()
plt.tight_layout()

plt.savefig(OUT / "map_p4_seed.png", dpi=300)
plt.close()


print("Created:")
print(OUT / "map_p4_seed.png")
print(OUT / "sensitivity_by_maxfac.csv")
print(OUT / "sensitivity_service_thresholds.csv")
print(OUT / "equity_frontier_comparison.png")