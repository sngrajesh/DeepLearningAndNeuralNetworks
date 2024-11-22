# Scikit-learn (sklearn) Reference

## Preprocessing (sklearn.preprocessing)

### Scalers
* `StandardScaler(copy=True, with_mean=True, with_std=True)`
* `MinMaxScaler(feature_range=(0, 1), copy=True)`
* `RobustScaler(quantile_range=(25.0, 75.0), copy=True, with_centering=True)`
* `Normalizer(copy=True, norm='l2')`
* `QuantileTransformer(n_quantiles=1000, output_distribution='uniform')`
* `PowerTransformer(method='yeo-johnson', standardize=True)`

### Encoders
* `LabelEncoder()`
* `OneHotEncoder(drop=None, sparse=True, handle_unknown='error')`
* `OrdinalEncoder(categories='auto', dtype=np.float64)`
* `LabelBinarizer(neg_label=0, pos_label=1, sparse_output=False)`

### Other Preprocessors
* `KBinsDiscretizer(n_bins=5, encode='onehot', strategy='quantile')`
* `Binarizer(threshold=0.0, copy=True)`
* `PolynomialFeatures(degree=2, interaction_only=False, include_bias=True)`

## Feature Selection (sklearn.feature_selection)

### Univariate Feature Selection
* `SelectKBest(score_func=<function f_classif>, k=10)`
* `SelectPercentile(score_func=<function f_classif>, percentile=10)`
* `SelectFpr(score_func=<function f_classif>, alpha=0.05)`
* `SelectFdr(score_func=<function f_classif>, alpha=0.05)`
* `SelectFromModel(estimator, threshold=None, prefit=False)`

### Recursive Feature Selection
* `RFE(estimator, n_features_to_select=None, step=1)`
* `RFECV(estimator, step=1, cv=None, scoring=None, min_features_to_select=1)`

## Classification (sklearn.linear_model, sklearn.tree, etc.)

### Linear Models
* `LogisticRegression(penalty='l2', C=1.0, solver='lbfgs', max_iter=100)`
* `SGDClassifier(loss='hinge', penalty='l2', alpha=0.0001, max_iter=1000)`
* `LinearDiscriminantAnalysis(solver='svd', shrinkage=None)`
* `RidgeClassifier(alpha=1.0, fit_intercept=True, normalize=False)`

### Support Vector Machines
* `SVC(C=1.0, kernel='rbf', degree=3, gamma='scale')`
* `LinearSVC(penalty='l2', loss='squared_hinge', C=1.0, max_iter=1000)`
* `NuSVC(nu=0.5, kernel='rbf', degree=3, gamma='scale')`

### Trees
* `DecisionTreeClassifier(criterion='gini', splitter='best', max_depth=None)`
* `ExtraTreeClassifier(criterion='gini', splitter='random', max_depth=None)`

### Ensemble Methods
* `RandomForestClassifier(n_estimators=100, criterion='gini', max_depth=None)`
* `GradientBoostingClassifier(n_estimators=100, learning_rate=0.1, max_depth=3)`
* `AdaBoostClassifier(base_estimator=None, n_estimators=50, learning_rate=1.0)`
* `ExtraTreesClassifier(n_estimators=100, criterion='gini', max_depth=None)`
* `BaggingClassifier(base_estimator=None, n_estimators=10)`
* `VotingClassifier(estimators, voting='hard', weights=None)`
* `StackingClassifier(estimators, final_estimator=None)`

### Naive Bayes
* `GaussianNB(priors=None, var_smoothing=1e-09)`
* `MultinomialNB(alpha=1.0, fit_prior=True, class_prior=None)`
* `BernoulliNB(alpha=1.0, binarize=0.0, fit_prior=True)`
* `ComplementNB(alpha=1.0, fit_prior=True, class_prior=None)`

### Nearest Neighbors
* `KNeighborsClassifier(n_neighbors=5, weights='uniform', algorithm='auto')`
* `RadiusNeighborsClassifier(radius=1.0, weights='uniform')`

## Regression (sklearn.linear_model, sklearn.tree, etc.)

### Linear Models
* `LinearRegression(fit_intercept=True, normalize=False)`
* `Ridge(alpha=1.0, fit_intercept=True, normalize=False)`
* `Lasso(alpha=1.0, fit_intercept=True, normalize=False)`
* `ElasticNet(alpha=1.0, l1_ratio=0.5, fit_intercept=True)`
* `SGDRegressor(loss='squared_loss', penalty='l2', alpha=0.0001)`
* `HuberRegressor(epsilon=1.35, max_iter=100, alpha=0.0001)`
* `TheilSenRegressor(fit_intercept=True, copy_X=True, max_iter=300)`
* `RANSACRegressor(base_estimator=None, min_samples=None, residual_threshold=None)`

### Support Vector Machines
* `SVR(kernel='rbf', degree=3, C=1.0, epsilon=0.1)`
* `LinearSVR(epsilon=0.0, tol=0.0001, C=1.0, loss='epsilon_insensitive')`
* `NuSVR(nu=0.5, C=1.0, kernel='rbf', degree=3, gamma='scale')`

### Trees
* `DecisionTreeRegressor(criterion='mse', splitter='best', max_depth=None)`
* `ExtraTreeRegressor(criterion='mse', splitter='random', max_depth=None)`

### Ensemble Methods
* `RandomForestRegressor(n_estimators=100, criterion='mse', max_depth=None)`
* `GradientBoostingRegressor(n_estimators=100, learning_rate=0.1, max_depth=3)`
* `AdaBoostRegressor(base_estimator=None, n_estimators=50, learning_rate=1.0)`
* `ExtraTreesRegressor(n_estimators=100, criterion='mse', max_depth=None)`
* `BaggingRegressor(base_estimator=None, n_estimators=10)`
* `VotingRegressor(estimators, weights=None)`
* `StackingRegressor(estimators, final_estimator=None)`

### Nearest Neighbors
* `KNeighborsRegressor(n_neighbors=5, weights='uniform', algorithm='auto')`
* `RadiusNeighborsRegressor(radius=1.0, weights='uniform')`

## Clustering (sklearn.cluster)

### Algorithms
* `KMeans(n_clusters=8, init='k-means++', n_init=10, max_iter=300)`
* `DBSCAN(eps=0.5, min_samples=5, metric='euclidean')`
* `AgglomerativeClustering(n_clusters=2, linkage='ward')`
* `SpectralClustering(n_clusters=8, eigen_solver=None, affinity='rbf')`
* `MeanShift(bandwidth=None, seeds=None, bin_seeding=False)`
* `AffinityPropagation(damping=0.5, max_iter=200, convergence_iter=15)`
* `Birch(threshold=0.5, branching_factor=50, n_clusters=3)`

## Dimensionality Reduction

### Linear Methods
* `PCA(n_components=None, copy=True, whiten=False)`
* `TruncatedSVD(n_components=2, algorithm='randomized')`
* `FastICA(n_components=None, algorithm='parallel', whiten=True)`
* `NMF(n_components=None, init=None, solver='cd', beta_loss='frobenius')`
* `LDA(n_components=None, solver='svd', shrinkage=None)`

### Non-linear Methods
* `TSNE(n_components=2, perplexity=30.0, early_exaggeration=12.0)`
* `Isomap(n_neighbors=5, n_components=2, eigen_solver='auto')`
* `LocallyLinearEmbedding(n_neighbors=5, n_components=2, method='standard')`
* `SpectralEmbedding(n_components=2, affinity='nearest_neighbors')`

## Model Selection (sklearn.model_selection)

### Splitting
* `train_test_split(*arrays, test_size=None, train_size=None, random_state=None)`
* `KFold(n_splits=5, shuffle=False, random_state=None)`
* `StratifiedKFold(n_splits=5, shuffle=False, random_state=None)`
* `GroupKFold(n_splits=5)`
* `TimeSeriesSplit(n_splits=5, max_train_size=None)`

### Parameter Search
* `GridSearchCV(estimator, param_grid, cv=None, scoring=None)`
* `RandomizedSearchCV(estimator, param_distributions, n_iter=10, cv=None)`
* `HalvingGridSearchCV(estimator, param_grid, factor=3, cv=5)`
* `HalvingRandomSearchCV(estimator, param_distributions, factor=3, cv=5)`

## Metrics (sklearn.metrics)

### Classification Metrics
* `accuracy_score(y_true, y_pred)`
* `precision_score(y_true, y_pred, average='binary')`
* `recall_score(y_true, y_pred, average='binary')`
* `f1_score(y_true, y_pred, average='binary')`
* `roc_auc_score(y_true, y_score)`
* `confusion_matrix(y_true, y_pred, labels=None)`
* `classification_report(y_true, y_pred, labels=None)`

### Regression Metrics
* `mean_squared_error(y_true, y_pred, sample_weight=None)`
* `mean_absolute_error(y_true, y_pred, sample_weight=None)`
* `r2_score(y_true, y_pred, sample_weight=None)`
* `explained_variance_score(y_true, y_pred, sample_weight=None)`

### Clustering Metrics
* `silhouette_score(X, labels, metric='euclidean')`
* `calinski_harabasz_score(X, labels)`
* `davies_bouldin_score(X, labels)`
* `adjusted_rand_score(labels_true, labels_pred)`
* `adjusted_mutual_info_score(labels_true, labels_pred)`

## Pipeline (sklearn.pipeline)

### Construction
* `Pipeline(steps)`
* `FeatureUnion(transformer_list, n_jobs=None, transformer_weights=None)`
* `make_pipeline(*steps, memory=None)`
* `make_union(*transformers, n_jobs=None)`

## Utilities

### Dataset Loading
* `load_boston()`
* `load_iris()`
* `load_digits()`
* `load_diabetes()`
* `load_breast_cancer()`

### Data Generation
* `make_classification(n_samples=100, n_features=20, n_classes=2)`
* `make_regression(n_samples=100, n_features=100)`
* `make_blobs(n_samples=100, n_features=2, centers=3)`
* `make_circles(n_samples=100, shuffle=True, noise=None)`
* `make_moons(n_samples=100, shuffle=True, noise=None)`