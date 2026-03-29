# Generic Data Science Pipeline - Customer Churn

**Objective:**
Build a robust classification model to predict customer churn. You have complete algorithmic freedom, but you must strictly adhere to the following professional data science methodology:

---

### Phase 1: Data Understanding & Observaton
1. **Understand Target & Distribution:** Start by using your metadata query tools to load the dataset, inspect its schema, and confirm the target variable.
2. **Check Health:** Identify column types, count missing values, and mathematically verify if there is class imbalance.

### Phase 2: Exploratory Data Analysis (EDA) & Visualisations
1. **Unbiased Exploration:** Perform Bivariate and Multivariate analysis on the numerical features.
2. **Visualisations:** You MUST generate distribution plots or correlation heatmaps using `matplotlib` / `seaborn` and actively save them to disk as `.png` files for review.

### Phase 3: Data Cleaning & Feature Engineering
1. **Imputation & Outliers:** Handle any missing values or severe outliers detected in Phase 1 computationally.
2. **Engineering:** Synthesize derived features (e.g., groupby stats, ratios, or mathematical interactions).
3. **Transformations:** Scale continuous variables (e.g., `StandardScaler`) and rigorously encode categorical columns (`OneHotEncoder`, `Target Encoding`).

### Phase 4: Model Selection & Train Strategy
1. **Start Simple:** Always establish a baseline first using Linear/Logistic regression.
2. **Progressive Complexity:** After conquering the baseline, move up to Tree models (Random Forest, LightGBM, XGBoost).
3. **Train-Test Integrity:** Implement rigorous Train-Test splits or K-Fold cross-validation to prevent data leakage.

### Phase 5: Tuning & Multi-Metric Evaluation
1. **Track Multi-Metrics:** Your scripts must simultaneously calculate and print **Accuracy, Precision, Recall, F1-Score, and ROC-AUC** for the model's performance. Example: 
`print("METRICS:" + json.dumps({"f1_macro": 0.75, "roc_auc": 0.81, "accuracy": 0.82}))`
2. **Autonomy:** Determine which metric should be the ultimate ranker on the Leaderboard (`set_evaluation_policy`) based on your analysis of class imbalance.
3. **Hyperparameter Tuning:** Use Grid/Random search methodologies on models that show promise (tune depth, regularization, learning rate).

### Phase 6: Ensembling & The Final Loop
1. **Iteration:** Loop through "new features -> new models -> new transformations" until you hit the local maximum.
2. **Final Combinations:** Before finishing your session, attempt to cross the efficiency ceiling using Ensembling combinations (Voting or Stacking the best models you discovered).
