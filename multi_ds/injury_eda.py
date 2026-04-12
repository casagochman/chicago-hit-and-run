import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

df = pd.read_parquet("train_crashes_final.parquet")
TARGET = "is_hit_and_run"

df["label"] = df[TARGET].map({0: "Non-H&R", 1: "H&R"})

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# ── 1. Injury severity breakdown ─────────────────────────────────────────────
ax = axes[0]
df["has_injury"] = (df["injuries_total"] > 0).astype(int)
df["has_fatal"]  = (df["injuries_fatal"] > 0).astype(int)

injury_summary = df.groupby("label").agg(
    total_crashes    = (TARGET, "count"),
    pct_zero_injury  = ("has_injury", lambda x: (1 - x.mean()) * 100),
    pct_any_injury   = ("has_injury", lambda x: x.mean() * 100),
    pct_fatal        = ("has_fatal",  lambda x: x.mean() * 100),
    mean_injuries    = ("injuries_total", "mean"),
).round(2)

print("=== Injury Summary by Class ===")
print(injury_summary.to_string())
print()

categories = ["No injury\n(property only)", "Any injury", "Fatal"]
hnr_vals    = [
    injury_summary.loc["H&R",    "pct_zero_injury"],
    injury_summary.loc["H&R",    "pct_any_injury"],
    injury_summary.loc["H&R",    "pct_fatal"],
]
nonhnr_vals = [
    injury_summary.loc["Non-H&R","pct_zero_injury"],
    injury_summary.loc["Non-H&R","pct_any_injury"],
    injury_summary.loc["Non-H&R","pct_fatal"],
]
x = np.arange(len(categories))
w = 0.35
ax.bar(x - w/2, nonhnr_vals, w, label="Non-H&R", color="#4C72B0")
ax.bar(x + w/2, hnr_vals,    w, label="H&R",     color="#DD8452")
ax.set_xticks(x); ax.set_xticklabels(categories)
ax.set_ylabel("% of crashes")
ax.set_title("Injury Profile — H&R vs Non-H&R")
ax.legend()
for xi, (nv, hv) in enumerate(zip(nonhnr_vals, hnr_vals)):
    ax.text(xi - w/2, nv + 0.5, f"{nv:.1f}%", ha="center", fontsize=8)
    ax.text(xi + w/2, hv + 0.5, f"{hv:.1f}%", ha="center", fontsize=8)

# ── 2. Crash type breakdown for H&R only ─────────────────────────────────────
ax = axes[1]
hnr = df[df[TARGET] == 1]
top_types = hnr["first_crash_type"].value_counts().head(8)
pct = (top_types / len(hnr) * 100).round(1)
colors = ["#DD8452" if t == "PARKED MOTOR VEHICLE" else "#aac4e0" for t in top_types.index]
bars = ax.barh(top_types.index[::-1], pct.values[::-1], color=colors[::-1])
ax.set_xlabel("% of H&R crashes")
ax.set_title("H&R Crash Types")
for bar, val in zip(bars, pct.values[::-1]):
    ax.text(bar.get_width() + 0.3, bar.get_y() + bar.get_height()/2,
            f"{val}%", va="center", fontsize=9)

print("=== H&R crash type breakdown ===")
print(pct.to_string())
print()

# ── 3. Injury distribution for H&R by crash type ─────────────────────────────
ax = axes[2]
hnr_injury = hnr.groupby("first_crash_type").agg(
    n_crashes   = (TARGET, "count"),
    pct_injured = ("has_injury", lambda x: x.mean() * 100)
).query("n_crashes >= 200").sort_values("pct_injured", ascending=True)

colors2 = ["#DD8452" if t == "PARKED MOTOR VEHICLE" else "#aac4e0"
           for t in hnr_injury.index]
ax.barh(hnr_injury.index, hnr_injury["pct_injured"], color=colors2)
ax.set_xlabel("% with injuries")
ax.set_title("H&R: % with Injuries by Crash Type")
for i, (idx, row) in enumerate(hnr_injury.iterrows()):
    ax.text(row["pct_injured"] + 0.5, i, f"{row['pct_injured']:.1f}%", va="center", fontsize=8)

plt.suptitle("H&R Injury Profile", fontsize=14, fontweight="bold")
plt.tight_layout()
plt.savefig("imgs/injury_eda.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved imgs/injury_eda.png")
