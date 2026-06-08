from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data"

demand_path = DATA / "demand_points_sariyer_10.csv"
facility_path = DATA / "candidate_facilities_sariyer_7.csv"
distance_path = DATA / "distance_matrix_seed_km.csv"

demand = pd.read_csv(demand_path)
facilities = pd.read_csv(facility_path)
dist = pd.read_csv(distance_path)

print("\n--- Demand Points ---")
print(demand.head())
print(demand.columns)

print("\n--- Candidate Facilities ---")
print(facilities.head())
print(facilities.columns)

print("\n--- Distance Matrix ---")
print(dist.head())
print(dist.columns)

required_demand_cols = {"demand_id", "name_en", "population", "lat", "lon"}
required_facility_cols = {"facility_id", "name_en", "type", "lat", "lon"}

missing_demand = required_demand_cols - set(demand.columns)
missing_facilities = required_facility_cols - set(facilities.columns)

if missing_demand:
    raise ValueError(f"Missing demand columns: {missing_demand}")

if missing_facilities:
    raise ValueError(f"Missing facility columns: {missing_facilities}")

# Distance matrix format check
if "demand_name" not in dist.columns:
    raise ValueError("distance_matrix_seed_km.csv must contain a column named 'demand_name'.")

facility_ids = facilities["facility_id"].tolist()
missing_distance_cols = set(facility_ids) - set(dist.columns)

if missing_distance_cols:
    raise ValueError(f"Distance matrix is missing facility columns: {missing_distance_cols}")

demand_names = demand["name_en"].tolist()
distance_names = dist["demand_name"].tolist()

missing_distance_rows = set(demand_names) - set(distance_names)

if missing_distance_rows:
    raise ValueError(f"Distance matrix is missing demand rows: {missing_distance_rows}")

# Numeric checks
if demand["population"].isna().any():
    raise ValueError("Population column contains missing values.")

if (demand["population"] <= 0).any():
    raise ValueError("Population values must be positive.")

for col in facility_ids:
    if dist[col].isna().any():
        raise ValueError(f"Distance column {col} contains missing values.")
    if (dist[col] < 0).any():
        raise ValueError(f"Distance column {col} contains negative values.")

print("\nData validation passed.")
print(f"Demand points: {len(demand)}")
print(f"Candidate facilities: {len(facilities)}")
print(f"Distance matrix shape: {dist.shape}")