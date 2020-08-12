# Linear models for Classification

## One class
### Logistic Regression
- Penalized LR: C is inverse to alpha in Linear Regression.

### SVM
- Bigger C, more overfitted the model.
```python
param_grid = {'svc__C': np.logspace(-3, 2, 6),
              'svc__gamma': np.logspace(-3, 2, 6) / X_train.shape[0]}
```

## Multiclass Classification
### One vs. Rest
- For 4 classes, for models: 1v{234}, 2v{134}, 3v{124}, 4v{123}
- Select the biggest score.

### One vs. One
- 1v2, 2v3, 1v4, 2v3, 2v4, 3v4 (uses a subset of the data for each model)
- Return most commonly predicted class.
