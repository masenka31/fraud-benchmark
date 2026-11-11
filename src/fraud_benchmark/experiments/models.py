"""Fixed XGBoost hyperparameters of the paper's Figure 2."""

XGB_PARAMS = {
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'tree_method': 'hist',
    'n_jobs': 4,
    'eval_metric': 'aucpr',
}
