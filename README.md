# Credit Offer Acceptance Prediction

Binary classification project for predicting whether a corporate client will accept a credit offer. The evaluation metric is ROC-AUC.

## Result

- Public leaderboard ROC-AUC: **0.759966**
- Public leaderboard rank: **64** at the time of submission

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

## Repository

The full pipeline is contained in [`credit_offer_modeling.ipynb`](credit_offer_modeling.ipynb): data loading, temporal validation, feature engineering, model training, blending, holdout evaluation and submission generation.

Competition data is not included in the repository.

Expected local structure:

```text
credit-offer-acceptance/
├── credit_offer_modeling.ipynb
├── data/
│   ├── train_apps.csv
│   └── test_apps.csv
└── outputs/
```

## Run

```bash
pip install -r requirements.txt
```

Place the competition files in `data/` and run the notebook from top to bottom. The final prediction file is written to `outputs/submission.csv`.
