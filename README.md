# Credit Offer Acceptance Prediction

Binary classification project for predicting whether a corporate client will accept a credit offer. The evaluation metric is ROC-AUC.

## Approach

The train and test sets are separated in time, so the validation scheme preserves temporal order.

The final pipeline includes:

- expanding-window temporal validation;
- leakage-safe feature engineering;
- missingness indicators and activity dynamics;
- empirical rank and residual features;
- XGBoost and CatBoost models;
- rank-based blending;
- a separate XGBoost model without absolute time features to reduce extrapolation risk.

## Key findings

- Random validation was not suitable because train and test belong to different time periods.
- Class weighting did not improve XGBoost.
- XGBoost and CatBoost produced sufficiently different rankings for ensembling.
- Rank blending improved temporal stability.
- Removing absolute time features improved the latest validation periods.
- Recency weighting and shorter training windows did not improve validation performance.

```bash
pip install -r requirements.txt
```

Place the competition files in `data/` and run the notebook from top to bottom. The final prediction file is written to `outputs/submission.csv`.
