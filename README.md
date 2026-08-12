# Brief overview of the project:

This project aimed to detect fraudulent credit card transactions using anonymised, PCA-transformed transaction data.

The main challenges were extreme class imbalance, with only **0.17% fraud cases**, and limited feature interpretability because the original variables were transformed into anonymised PCA components.

## Methodology

1. Loaded and explored the credit card transaction dataset containing **284,807 transactions and 31 columns**.
2. Checked data types, missing values, class distribution and duplicate observations. There were no missing values, and **1,081 exact duplicate rows (0.38%)** were removed before splitting the data to avoid leakage.
3. Performed exploratory data analysis and examined the relationship between the anonymised PCA features and the fraud target.
4. Created a stratified training/test split and applied `RobustScaler` to the non-PCA numerical features so that preprocessing was fitted only on the training data.
5. Addressed the severe class imbalance using **class-weighted models** rather than changing the original test-set distribution through resampling.
6. Evaluated several supervised models, including **Logistic Regression, Decision Tree, AdaBoost, Random Forest, and HistGradientBoosting**, using precision, recall, F1-score, ROC-AUC and especially **PR-AUC (Average Precision)**.
7. Compared the models and selected Random Forest and HistGradientBoosting as the strongest candidates for further hyperparameter tuning.
8. Performed hyperparameter tuning using `RandomizedSearchCV` with stratified cross-validation, using PR-AUC as the main selection metric.
9. Selected **Random Forest** as the final model based on its strongest cross-validated PR-AUC performance.
10. Optimised the classification threshold using **out-of-fold predictions** rather than directly using the test set, preventing threshold-selection leakage.
11. Evaluated the final model once on the untouched test set using precision, recall, F1-score, ROC-AUC, PR-AUC and the confusion matrix.
12. Checked the train, cross-validation and test PR-AUC values to assess possible overfitting.
13. Used permutation importance to analyse the contribution of features to the final model.
14. Saved the complete preprocessing pipeline, trained model and selected decision threshold together using `joblib` for reproducible inference.

## Results:

Random Forest achieved the strongest cross-validated PR-AUC among the tuned candidates.

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC | PR-AUC |
|---|---:|---:|---:|---:|---:|---:|
| **Random Forest (tuned)** | **0.9994** | **0.8861** | 0.7368 | **0.8046** | 0.9457 | **0.7970** |
| Hist Gradient Boosting (tuned) | 0.9980 | 0.4514 | **0.8316** | 0.5852 | **0.9563** | 0.7343 |
| Logistic Regression | 0.9748 | 0.0554 | 0.8737 | 0.1041 | 0.9685 | 0.6703 |
| Decision Tree | 0.9953 | 0.2327 | 0.7789 | 0.3584 | 0.8886 | 0.4990 |
| AdaBoost | 0.9707 | 0.0453 | 0.8211 | 0.0859 | 0.9448 | 0.0983 |

Because fraud detection is highly imbalanced, **PR-AUC was given more importance than accuracy**.

The tuned Random Forest achieved a cross-validated PR-AUC of approximately **0.8182**, compared with **0.7365** for tuned HistGradientBoosting.

For the final Random Forest, the decision threshold was selected from out-of-fold predictions. The selected threshold was approximately **0.559**.

At this threshold on the untouched test set, the model achieved approximately:

- **Precision:** 0.921
- **Recall:** 0.737
- **F1-score:** 0.819

The final model was therefore selected as **Random Forest**, providing a strong balance between detecting fraudulent transactions and limiting false fraud alerts.

## Model Saving

The complete trained pipeline and decision threshold were saved using `joblib`:

```python
joblib.dump(
    {
        "pipeline": best_model,
        "threshold": best_threshold
    },
    "fraud_detection_pipeline.pkl"
)
```

This allows the same preprocessing, model and decision threshold to be reused during inference.
