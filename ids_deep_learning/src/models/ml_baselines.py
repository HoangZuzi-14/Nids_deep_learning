from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


def build_logistic_regression(seed: int = 42):
    return LogisticRegression(max_iter=1000, class_weight="balanced", random_state=seed)


def build_random_forest(seed: int = 42):
    return RandomForestClassifier(
        n_estimators=200,
        class_weight="balanced_subsample",
        n_jobs=-1,
        random_state=seed,
    )


def build_xgboost(seed: int = 42):
    import xgboost as xgb
    return xgb.XGBClassifier(
        n_estimators=200,
        random_state=seed,
        n_jobs=-1,
        eval_metric="mlogloss",
    )


def build_lightgbm(seed: int = 42):
    import lightgbm as lgb
    return lgb.LGBMClassifier(
        n_estimators=200,
        random_state=seed,
        n_jobs=-1,
        class_weight="balanced",
        verbose=-1,
    )


