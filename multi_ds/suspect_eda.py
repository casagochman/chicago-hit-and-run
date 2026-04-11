import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_parquet("train_crashes_final.parquet")
TARGET_COL = "is_hit_and_run"

suspect_cols = ["driver_age_avg", "driver_vision_obscured", "unknown_use_veh_involved"]
df_susp = df[[TARGET_COL] + suspect_cols].copy()
df_susp["label"] = df_susp[TARGET_COL].map({0: "Non-H&R", 1: "H&R"})

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

# 1. driver_age_avg
ax = axes[0]
miss = df_susp.groupby("label")["driver_age_avg"].apply(lambda s: s.isna().mean() * 100)
print("=== driver_age_avg missingness (%) ===")
print(miss.to_string())

for label, color in [("Non-H&R", "#4C72B0"), ("H&R", "#DD8452")]:
    vals = df_susp.loc[df_susp["label"] == label, "driver_age_avg"].dropna()
    ax.hist(vals, bins=40, alpha=0.6, label=label, color=color, density=True)
ax.set_title("driver_age_avg distribution\n(non-null values only)")
ax.set_xlabel("Average driver age")
ax.set_ylabel("Density")
ax.legend()
for i, (label, pct) in enumerate(miss.items()):
    ax.text(0.05, 0.95 - i*0.08, f"{label} missing: {pct:.1f}%",
            transform=ax.transAxes, fontsize=9, color="#333")

# 2. driver_vision_obscured
ax = axes[1]
rate = df_susp.groupby("label")["driver_vision_obscured"].mean() * 100
print("\n=== driver_vision_obscured rate (%) ===")
print(rate.to_string())
bars = ax.bar(rate.index, rate.values, color=["#4C72B0", "#DD8452"])
ax.set_title("driver_vision_obscured = 1 rate (%)")
ax.set_ylabel("% of crashes")
ax.set_ylim(0, rate.max() * 1.4)
for bar, val in zip(bars, rate.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")

# 3. unknown_use_veh_involved
ax = axes[2]
rate2 = df_susp.groupby("label")["unknown_use_veh_involved"].mean() * 100
print("\n=== unknown_use_veh_involved rate (%) ===")
print(rate2.to_string())
bars2 = ax.bar(rate2.index, rate2.values, color=["#4C72B0", "#DD8452"])
ax.set_title("unknown_use_veh_involved = 1 rate (%)")
ax.set_ylabel("% of crashes")
ax.set_ylim(0, rate2.max() * 1.4)
for bar, val in zip(bars2, rate2.values):
    ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
            f"{val:.1f}%", ha="center", fontsize=11, fontweight="bold")

plt.suptitle("Suspect Feature EDA — H&R vs Non-H&R", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("imgs/suspect_feature_eda.png", dpi=150, bbox_inches="tight")
plt.close()
print("\nSaved imgs/suspect_feature_eda.png")

print("\n=== driver_age_avg summary by class ===")
print(df_susp.groupby("label")["driver_age_avg"].agg(
    count="count", mean="mean", median="median",
    n_missing=lambda s: s.isna().sum()
))
