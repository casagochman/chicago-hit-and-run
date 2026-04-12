# Writeup Outline — Chicago Hit-and-Run Prediction
## Word Budget: ~1,500 words (narrative only, code/comments excluded)

---

## Section Breakdown

| Section | Target Words |
|---|---|
| 1. Introduction | 200 |
| 2. Research Questions | 50 |
| 3. Data | 200 |
| 4. Methodology | 250 |
| 5. Results & Discussion | 650 |
| 6. Conclusion | 100 |
| 7. References | — |
| **Total** | **~1,450** |

---

## 1. Introduction (~200 words)
**Include:**
- H&R crashes as urban public safety problem — scale, enforcement challenge
- Why spatial/nightlife context is theorised to matter
- 3 credible references: Han et al. (2025) PLOS ONE confirmed; 2 more on H&R correlates or urban crash prediction — **find on Google Scholar, do NOT use LLM-generated citations**
- Gap: no multi-dataset Chicago study combining behavioural + spatial features

**Do NOT include:**
- Technical method details
- Results or findings

**Images:** None

---

## 2. Research Questions (~50 words)
**Include:**
- Primary RQ (exact wording, ending with ?)
- Secondary RQ (exact wording, ending with ?)

**Images:** None

---

## 3. Data (~200 words)
**Include:**
- 4 datasets: Chicago Traffic Crashes, Vehicles, People, Business Licences (Nightlife)
- Source, size (~221k records), time period (2024–2025)
- Key preprocessing:
  - Leakage prevention: excluded `bac_result`, `towed_i`, `arrest_related_i`, `prim_contributory_cause`
  - UNKNOWN values retained as informative signals (driver absence = signal)
  - Binary encoding bug fixed (UNKNOWN→0, not 1; `driver_vision_obscured` dropped from 97.9% → 0.6% in H&R)
  - Crash-level aggregation from Vehicles + People (mean/min/max per crash)
  - Spatial join: 1000m buffer for nightlife density
  - 80/20 stratified train/test split; 30.6% H&R class balance

**Do NOT include:**
- Full feature list
- Integration code details

**Images:** `p0_class_balance.png`

---

## 4. Methodology (~250 words)
**Include:**
- Frame as explanatory ML, not deployment (cite Han et al. 2025)
- 4 methods justified:
  1. **LightGBM** — best of 4-model comparison; tuned via RandomizedSearchCV (2 rounds, 25 iter, 5-fold CV); F2 threshold selection
  2. **SHAP** — global feature importance to answer primary RQ
  3. **Beat-level recall/precision choropleth** — error rate (not count) per beat; Queen contiguity
  4. **Moran's I** — spatial autocorrelation test; 999 permutations; answers secondary RQ
- Preprocessing: target encoding for `beat_of_occurrence`, OHE for low-cardinality cats, median imputation

**Do NOT include:**
- LR/RF/XGB details beyond brief mention
- Spatial-blind baseline details (mention in results)

**Images:** None

---

## 5. Results & Discussion (~650 words)

### 5a. Model Performance (~100 words)
- 4-model CV: LR 0.900, RF 0.914, XGB 0.933, LGBM 0.937 → LightGBM best
- Test ROC-AUC: **0.9388**, Recall: **0.934**, Precision: **0.653** at threshold 0.298
- High AUC = richness of crash report data, not operational power (explanatory framing)

**Images:** `final_evaluation.png`

### 5b. H&R Crash Profile (~100 words)
- 87.4% no injury; 38.8% parked vehicle (opportunistic property damage dominates)
- Pedestrian H&R (3.5%, >80% injury) and cyclist H&R (2.0%, >60% injury) are serious minority
- Model signal tuned to parked-car majority

**Images:** `injury_eda.png`

### 5c. Feature Importance — Primary RQ (~250 words)
- 3 SHAP feature groups:
  1. **Witness/victim absence** — `driver_age_avg` (#1, encodes no persons at scene in 40.9% H&R), `unknown_use_veh_involved` (#2), `total_people_in_crash` (#4), gender counts (#6,#7)
  2. **Temporal anonymity** — `hour_cos` (#10, midnight peak)
  3. **Crash geometry** — `intersection_related_i` (#12), `front_impact_involved` (#13)
- Nightlife (#15): bivariate association present but +0.0002 AUC when removed — shared-variance finding, not causal
- Explicit answer to primary RQ: nightlife proximity not independently predictive

**Images:** `shap_bar.png`, `shap_beeswarm.png`

### 5d. Spatial Error Analysis — Secondary RQ (~200 words)
- Beat recall mostly 0.90–1.00 city-wide; visual NW cluster
- Moran's I recall: I=0.039, p=0.106 → NOT significant → no geographic blind spots
- Moran's I precision: I=0.130, p=0.001 → SIGNIFICANT → false alarms cluster in dense north/central corridors
- Explicit answer to secondary RQ: missed H&Rs spatially random; false alarms geographically concentrated

**Images:** `beat_choropleth.png`, `morans_i_precision.png`

---

## 6. Conclusion (~100 words)
**Include:**
- Primary RQ answer (nightlife not independent; behavioural/temporal features dominate)
- Secondary RQ answer (no blind spots in recall; false alarm zones cluster)
- Most H&R = opportunistic property damage
- Limitations: observational (no causality), explanatory framing, parked-car majority dominates
- Future work: targeted model for pedestrian/cyclist H&R

**Do NOT include:**
- New findings
- Long methodological caveats

**Images:** None

---

## 7. References
- Han et al. (2025) PLOS ONE ✓ — verified
- [CITATION 2] — find on Google Scholar yourself
- [CITATION 3] — find on Google Scholar yourself

---

## Key Figures (max ~7)

| Figure | Section | Purpose |
|---|---|---|
| `p0_class_balance.png` | Data | Show 30.6% H&R class balance |
| `final_evaluation.png` | Results 5a | ROC + confusion matrix + PR curve |
| `injury_eda.png` | Results 5b | H&R crash type + injury profile |
| `shap_bar.png` | Results 5c | Feature importance ranking |
| `shap_beeswarm.png` | Results 5c | Feature direction + magnitude |
| `beat_choropleth.png` | Results 5d | Recall + precision by beat |
| `morans_i_precision.png` | Results 5d | Precision autocorrelation (significant) |
