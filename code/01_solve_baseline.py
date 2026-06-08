from pathlib import Path
import itertools
import pandas as pd
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"
OUT = ROOT / "outputs"
OUT.mkdir(exist_ok=True)

demand = pd.read_csv(DATA / "demand_points_sariyer_10.csv")
facilities = pd.read_csv(DATA / "candidate_facilities_sariyer_7.csv")
dist = pd.read_csv(DATA / "distance_matrix_seed_km.csv").set_index("demand_name")

demand_names = demand["name_en"].tolist()
facility_ids = facilities["facility_id"].tolist()
pop = dict(zip(demand["name_en"], demand["population"]))

def weighted_average_distance(assignments):
    total_pop = sum(pop.values())
    return sum(pop[d] * dist.loc[d, assignments[d]] for d in demand_names) / total_pop

def solve_for_maxfac(maxfac):
    best_solution = None

    for subset in itertools.combinations(facility_ids, maxfac):
        assignments = {}

        for d in demand_names:
            nearest_facility = min(subset, key=lambda f: dist.loc[d, f])
            assignments[d] = nearest_facility

        avg_distance = weighted_average_distance(assignments)
        max_distance = max(dist.loc[d, assignments[d]] for d in demand_names)

        candidate = {
            "MaxFac": maxfac,
            "OpenedFacilities": ";".join(subset),
            "AvgWeightedDistance_km": avg_distance,
            "MaxDistance_km": max_distance,
            "Assignments": assignments
        }

        if best_solution is None:
            best_solution = candidate
        elif avg_distance < best_solution["AvgWeightedDistance_km"]:
            best_solution = candidate
        elif np.isclose(avg_distance, best_solution["AvgWeightedDistance_km"]) and max_distance < best_solution["MaxDistance_km"]:
            best_solution = candidate

    return best_solution

summary_rows = []
assignment_rows = []

for maxfac in range(1, len(facility_ids) + 1):
    sol = solve_for_maxfac(maxfac)

    summary_rows.append({
        "MaxFac": sol["MaxFac"],
        "OpenedFacilities": sol["OpenedFacilities"],
        "AvgWeightedDistance_km": round(sol["AvgWeightedDistance_km"], 4),
        "MaxDistance_km": round(sol["MaxDistance_km"], 4)
    })

    for d, f in sol["Assignments"].items():
        assignment_rows.append({
            "MaxFac": maxfac,
            "demand_name": d,
            "assigned_facility": f,
            "distance_km": dist.loc[d, f],
            "population": pop[d]
        })

summary = pd.DataFrame(summary_rows)
assignments = pd.DataFrame(assignment_rows)

summary.to_csv(OUT / "baseline_frontier.csv", index=False)
assignments.to_csv(OUT / "baseline_assignments_all.csv", index=False)

print("\nBaseline efficient frontier:")
print(summary)
print("\nSaved:")
print(OUT / "baseline_frontier.csv")
print(OUT / "baseline_assignments_all.csv")