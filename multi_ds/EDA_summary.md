# Chicago Hit-and-Run EDA — Full Analysis

## P0: Data Overview

**Class Balance:** 121,560 normal vs 53,493 hit-and-run — a **~70/30 split**. Moderate imbalance, manageable without aggressive resampling, but worth using class weights or stratified splits.

**Missing Values:** Remarkably clean — only `most_severe_injury` has any missingness, at ~0.2% (well under the 5% threshold). The dataset is essentially complete, which is a big advantage.

---

## P1: Temporal Patterns — Strong Signal

**Hour of day** is one of the strongest features:
- H&R rate peaks at **~46% from midnight to 4am**, then drops sharply to ~25% during morning commute hours (8–10am)
- Rises again through evening into late night
- This ~20pp swing is very actionable for modeling

**Day of week:** Clear weekend effect — **Sunday (~37%) and Saturday (~33%)** significantly above baseline (30.6%), while weekdays cluster at 28–30%. Consistent with alcohol/impairment-driven flight behavior.

**Month:** Essentially flat — all months within ±1.5pp of baseline. Month is not a useful feature.

**168-hour heatmap:** Confirms the pattern compounds: **Sunday early morning (1–4am) is the hottest cell** at ~52–53%, while weekday mornings are coldest at ~23–25%. This interaction (`hour × day`) is a meaningful engineered feature.

---

## P2: Nightlife — Non-linear, Context-dependent

**Density bins:** The relationship is **non-monotonic** — H&R rate rises from 0→31–75 establishments (30% → 37%), then drops at 75+ back to ~29%. The 75+ group is probably downtown Chicago proper where:
- More witnesses/cameras deter flight
- More police presence
- n=128k (majority of crashes), possibly diluting the effect

**Spatial map:** Nightlife is hyper-concentrated — there's one extreme hotspot in central Chicago (likely River North/Wicker Park corridor) with values 8000+, and the rest of the city is sparse.

**KDE comparison:** Hit-and-run crashes have a **more diffuse, multi-modal spatial pattern** spread across the city, while normal crashes are more centrally concentrated. H&R perpetrators may be fleeing to or from outer neighborhoods.

**Nightlife × Time interaction:** The key finding — weekend nights amplify nightlife density effects by **~10–15pp** across all density bins. A `nightlife_density × is_weekend_night` interaction term could be valuable.

---

## P3: Categorical Features — Several Strong Predictors

**Lighting:** Clear gradient:
- UNKNOWN: ~50% H&R (likely a proxy for nighttime/poor documentation)
- DARKNESS: ~40%
- DARKNESS, LIGHTED ROAD: ~37%
- DAYLIGHT: ~27% (below baseline)

Darkness doubles hit-and-run probability vs daylight. Strong feature.

**First crash type:**
- PARKED MOTOR VEHICLE: ~50%+ (driver clips parked car and flees — the canonical H&R scenario)
- PEDESTRIAN: ~40%
- SIDESWIPE OPPOSITE DIRECTION: above baseline
- REAR TO REAR, TURNING, ANGLE: below baseline

PARKED MOTOR VEHICLE being the top category makes intuitive sense — least witnesses, no injured party present.

**Primary contributory cause — EXCLUDED (leakage risk):**
Despite being a strong predictor (reckless driving ~55%+, drinking above baseline), this field is subject to circular leakage: officers write "UNABLE TO DETERMINE" *because* the driver fled — the very outcome we are predicting. The field is determined simultaneously with the H&R designation and is therefore excluded from the feature matrix. See RESEARCH_LOG for full reasoning.

**Trafficway type:**
- ONE-WAY streets: highest H&R rate (~40%) — likely downtown streets where escape is easier
- PARKING LOT, ALLEY: also elevated
- FOUR WAY (intersections): below baseline — more witnesses

**Weather:** Minimal effect — most conditions cluster near baseline. UNKNOWN weather is elevated (~40%) but this may be the same nighttime documentation artifact. Weather itself doesn't seem to drive H&R.

**Traffic control device:** RR crossing signs and "other regulatory signs" are highest — likely low-traffic areas. NO CONTROLS also above baseline.

**Road alignment:** Overwhelmingly "STRAIGHT AND LEVEL" in the dataset (n=172k), so other categories have tiny n. Not a useful feature due to near-zero variance.

---

## P4: Numerical Correlations — Weak Individual Signal

**Point-biserial correlations:** All correlations with the target are small (≤0.05 in magnitude):
- `total_vehicles_in_crash`: slight positive correlation
- `active_nightlife_index`: slight positive
- `male_count_in_crash`: **negative** — more males involved → less likely H&R (perhaps more confrontational situations where drivers stay)
- `driver_age_avg`: most negative — older drivers less likely to flee

**Key takeaway:** No single numerical feature is strongly linearly correlated with H&R. The signal is in combinations and non-linearities — good argument for tree-based models.

**Correlation matrix:** `veh_year_avg/min/max` are highly inter-correlated (expected). `total_vehicles` and `total_people` are correlated. Multicollinearity to manage.

---

## P5: Vulnerable Road Users & Flags

**Risk ratios chart** reveals the most extreme predictors:
- `no_safety_equipment`: **RR ~20x** — massive signal (though likely co-linear with crash type)
- `driver_vision_obscured`: very high RR
- `unknown_use_veh_involved`: high RR
- `is_weekend_night`: elevated
- `pedestrian_involved`: elevated
- `cyclist_involved`: **below 1** (protective — cyclists are more visible/vocal?)

**Pedestrian vs Cyclist asymmetry:** Pedestrian involvement increases H&R rate to ~40%; cyclist involvement actually *decreases* it to ~27% (below baseline). Striking — cyclists may be more likely to pursue/confront the driver, or crashes are slower-speed.

---

## P6: Interaction Effects

**Lighting × Weather heatmap:** The worst combination is **UNKNOWN lighting + UNKNOWN weather** at 64% H&R rate, and DARKNESS + SNOW at 60%. Clear confirmation that poor visibility/documentation compounds risk.

**Nightlife × Weekend Night:** Weekend night adds a consistent ~10pp premium at every nightlife density level. The sweet spot is **31–75 establishments + weekend night** at ~50% H&R rate.

**Speed × First crash type:** PARKED MOTOR VEHICLE crashes at 50+ mph zones hit 51–53% H&R. The interaction of high speed + parked vehicle is notable.

---

## Summary: Feature Importance Preview

| Category | Key Features | Signal Strength |
|---|---|---|
| Temporal | `crash_hour`, `is_weekend_night`, `hour×dow` | ★★★★★ |
| Categorical | `lighting_condition`, `first_crash_type`, `trafficway_type` | ★★★★☆ |
| Flags | `no_safety_equipment`, `driver_vision_obscured`, `pedestrian_involved` | ★★★★☆ |
| Spatial | `nightlife_density_1000m` (non-linear), KDE zone | ★★★☆☆ |
| Numerical | `driver_age_avg`, `male_count_in_crash` | ★★☆☆☆ |
| Month/Weather | Month, weather alone | ★☆☆☆☆ |

The data tells a coherent story: **H&R is primarily a nighttime, weekend, impairment-driven, low-accountability phenomenon** — drivers flee when it's dark, late, they're drunk, and there are few witnesses.
