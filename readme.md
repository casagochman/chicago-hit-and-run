# Chicago Hit-and-Run Crash Prediction

  Can spatial proximity to nightlife areas predict whether a driver will flee a crash scene?

  This project integrates four City of Chicago datasets to classify hit-and-run incidents using machine learning, treating the model as a **feature importance engine** rather than a deployable predictor.

  ## Method

  - **Multi-model comparison** (Logistic Regression, Random Forest, XGBoost, LightGBM) with 5-fold stratified cross-validation and randomized hyperparameter search
  - **SHAP** for explainability — quantifying each feature's contribution to individual predictions
  - **Novel spatial feature**: nightlife density derived from active liquor licences, evaluated against behavioural, temporal, and environmental predictors
  - **Local Moran's I (LISA)** applied to per-beat recall scores to identify spatial clusters of model underperformance across Chicago's 271 police beats

  ## Key Finding

  Nightlife density is a meaningful but not dominant predictor. Spatial error analysis revealed significant Low-Low clusters — beats with systematically below-average detection surrounded by similarly underperforming neighbours — pointing
   to geographic blind spots likely tied to underreporting rather than model failure.

  ## Data Sources

  - City of Chicago Open Data Portal: Traffic Crashes, Vehicles, People, Business Licences (2024–2025)
