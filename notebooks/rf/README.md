# Random Forest Baseline (BRISC2025)

This notebook implements a Random Forest baseline for 4-class brain tumor classification on BRISC2025.

## Notebook
- `rf.ipynb`

## Method Summary
- Preprocessing: resize to 128x128, normalize to [0, 1]
- Feature extraction: HOG
- Dimensionality reduction: PCA
- Classifier: RandomForestClassifier with hyperparameter search

## Reported Results
| Metric | Value |
|---|---:|
| Accuracy | 0.92 |
| F1-score | 0.92 |
| Precision | 0.92 |
| Recall | 0.92 |

## Notes
- This is a traditional ML baseline used for comparison against SVM/CNN approaches.
- Update dataset paths in notebook cells before execution.
