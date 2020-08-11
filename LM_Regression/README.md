# Linear models for Regression 

### Ordinary Least Squares:
- Always has a unique solution.
- If you have more rows than columns it will work fine, else this optimization doesn't work.

### Ridge
- Always has a unique solution.
- Alpha: L2 penalization (tune with Grid search) 
```np.logspace(-3,3,13)```
- Very important to scale data.

### Lasso
- Set some coefficients to zero.
- We get automatic feature selection.
- Alpha: L1 penalization (tune with Grid search) 
```np.logspace(-5, 0, 10)```

### Elastic Net
- Combines benefits of Ridge and Lasso
- Two parameters to tune.
```python 
param_grid = {'alpha': np.logspace(-4, -1, 10),
              'l1_ratio': [0.01, .1, .5, .8, .9, .95, .98, 1]}
```

*Linear models work better with a normal distribuited target (log transformation).*

- With skewed target:
```python
cross_val_score(make_pipeline(preprocess, LinearRegression()),
                X_train, y_train, cv=5)

array([0.928, 0.927, 0.932, 0.898, 0.884])
```

- With a normal target:

```python
from sklearn.compose import TransformedTargetRegressor

log_regressor = TransformedTargetRegressor(LinearRegression(), 
                                           func=np.log, 
                                           inverse_func=np.exp)

cross_val_score(make_pipeline(preprocess, log_regressor),
                X_train, y_train, cv=5)

array([0.95 , 0.943, 0.941, 0.913, 0.922])
```
*TransformedTargetRegressor* applies log, train and applies exp.

