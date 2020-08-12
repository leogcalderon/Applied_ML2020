# Trees, Forests & Ensembles
The cool thing about Tree based models is that they can learn non linear relationships between features and targets and also requieres less preprocessing. They are interpretable.

## 1. Trees
### 1.1 Decision trees for Classification
- Idea: series of binary questions.
- "Questions" are thresholds on single features.
- Criteria (for classification): Gini-Index or Cross-Entropy. How pure are the leaves.

### 1.2 Regression trees
- Prediction: average of the points that lie in that leaf.
- Purity: MSE or MAE.

*Visualizing trees with Sklearn*
```python
from sklearn.tree import plot_tree

tree = DecisionTreeClassifier()
tree.fit(X_train, y_train)
tree_dot = plot_tree(tree, feature_names=feature_names)
```

### 1.3 Parameter Tuning
- **Pre-pruning:** limit tree size while you are building it *(pick one, maybe  to tune):*
    - max_depth
    - max_leaf_nodes (the best)
    - min_samples_split

*We can use* ```get_depth()``` *and* ```get_n_leaves()``` *to a full grown tree, so we have a baseline of those values.*

- **Post-pruning:** build the tree, and then prune it.
    - Cost complexity pruning
    - ```ccp_alpha: np.linspace(0,0.03,20)```

### 1.4 Other considerations
- **Extrapolation:** it isn't something bad, because extrapolation is difficult for every model, but you have to keep it in mind. (also to kNN)
- **Inestability:** when you get a different tree structure with differents splits of the data. (losses interpretability)
- **Feature importance:** summary of the structure of the tree.
- **Categorical data:** can split on categorical data directly, not in sklearn.

## 2. Ensemble models

### 2.1 Poor man's Ensembles
- Build different models. (the more the better - if they are not correlated -different random seeds of the same model)
- Average the results.
- Also works with NN.
- Sklearn: ```VotingClassifier```

### 2.2 Bagging
- Creates differents models by injecting randomness in the training set of each models.
- Creates bootstrap samples for the train sets.
- Sklearn: ```BaggingClassifier``` ```BaggingRegressor```

### 2.3 Random Forests
- Average of Trees. More trees are always better.
- For each tree: pick bootstrap sample of data
- For each split: pick random sample of features.
- Main parameter: ```max_features```
      - sqrt(n_features) for classificationn
      - (n_features) for Regression

### 2.4 Extremely randomized Trees
- Randomly draw threshold for each feature.
- Doesn't use bootstrap.
- Can have smoother boundaries.

### 2.5 considerations
- If you want to figure out how many trees to use, there is no point to do GridSearch with the number of trees because more trees will be always better. With ```warm_start``` reuse the solution of the previous call to fit and add more estimators to the ensemble.

```python
train_scores = []
test_scores = []
rf = RandomForestClassifier(warm_start=True)
estimator_range = range(1, 100, 5)
for n_estimators in estimator_range:
    rf.n_estimators = n_estimators
    rf.fit(X_train, y_train)
    train_scores.append(rf.score(X_train, y_train))
    test_scores.append(rf.score(X_test, y_test))
```
- OOB estimates: give us an approximation of the test set for free.
- Scale and distribution of features don't matter to a tree.
