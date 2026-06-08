from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

RECOMMENDED_MAXFAC = 4

assignments = pd.read_csv(OUT / "baseline_assignments_all.csv")
recommended = assignments[assignments["MaxFac"] == RECOMMENDED_MAXFAC].copy()

recommended.to_csv(OUT / "assignments_p4_seed.csv", index=False)

print(recommended)
print("\nSaved:", OUT / "assignments_p4_seed.csv")