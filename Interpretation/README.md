# Model Interpretation and Feature Selection


## 1. Feature importance
- Types of explanations:
	- **Explain model globally:** How does the output depend on the input?
	- **Explain model locally:** Why did it classify this point this way?

- Explaining the model != explaining the data:
	- Model inspection only tells you about the model
	- The model might not acurately reflect the data
	
- Features important to the model:
	- **Naive:** 
		- ```coef_``` - linear models
		- ```feature_importances_``` - tree based models
	- **Linear Models coefficients:** 
		- only meaningful after scaling.
		- correlation among features might make coefficients uninterpretable.
		- L1 picks one at random from a correlated group.
		- Any penalty will invalidate usual interpretation of linear coefficients.
		
### 1.1 Drop Feature Importance

![Drop feature importance](images/drop.png)

- Doesn't really explain model, because we fit a new model in each iteration.
- Can't deal with correlated features well
- Very slow
- Can be used for feature selection

### 1.2 Permutation Importance
- Instead of dropping the feature, we shuffle the feature column and calculates the difference score N times (because we want the expected value of the difference).
- We don't fit a new model, we use a validation set.
- Very slow, but not as dropping features.
```python 
from sklearn.inspection import permutation_importance
```

### 1.3 LIME
https://github.com/marcotcr/lime

### 1.4 SHAP
https://github.com/slundberg/shap

### 1.5 Partial dependence plots
Not only says how important a feature is, but also says how interact with the prediction.
Suppose you would like to understand importance of variable pi in the model, PDP builds the model averaging other predictor variable except one choosen predictor variable pi and measures change in response yhat and y, change in response can help identify how a varaible is affecting the model.


## 2. Feature selection
More interpretable model, faster prediction and training and less storage for model and dataset.

### 2.1 Unsupervised Feature Selection
- Covariance-based: remove correlated features
- Variance-based: 0 variance or mostly constant

### 2.2 Univariate Statistics
- Pick statistic, check p-values
- f_regression, f_classif, chi2
```python
from sklearn.feature_selection import SelectKBest, SelectPercentile, SelectFpr
```

### 2.3 Model-Based Feature selection



