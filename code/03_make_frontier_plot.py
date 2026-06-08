from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "outputs"

frontier = pd.read_csv(OUT / "equity_frontier.csv")

plt.figure(figsize=(7, 5))
plt.plot(frontier["MaxFac"], frontier["AvgWeightedDistance_km"], marker="o")
plt.xlabel("Maximum number of opened facilities")
plt.ylabel("Average population-weighted distance (km)")
plt.title("Efficient Frontier: Ambulance Outpost Siting in Sarıyer")
plt.grid(True)
plt.tight_layout()

plt.savefig(OUT / "efficient_frontier_seed.png", dpi=300)
plt.show()

print("Saved:", OUT / "efficient_frontier_seed.png")