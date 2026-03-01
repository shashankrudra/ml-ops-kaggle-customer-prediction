# Author: GitHub Copilot

from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
import lightgbm as lgb
import xgboost as xgb

def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)

def train_lightgbm(X_train, y_train, X_val, y_val):
    lgb_model = lgb.LGBMClassifier(
        n_estimators=600,
        learning_rate=0.1,
        max_depth=3,
        num_leaves=8,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbose=-1,
        scale_pos_weight=2.0
    )
    lgb_callbacks = [lgb.early_stopping(stopping_rounds=20, verbose=False)]
    lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=lgb_callbacks)
    y_val_pred = lgb_model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_val_pred)
    return lgb_model, auc

def train_xgboost(X_train, y_train, X_val, y_val):
    xgb_model = xgb.XGBClassifier(
        n_estimators=2000,
        learning_rate=0.03,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=0,
        scale_pos_weight=8.95,
        eval_metric='auc',
        early_stopping_rounds=50
    )
    xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    y_val_pred = xgb_model.predict_proba(X_val)[:, 1]
    auc = roc_auc_score(y_val, y_val_pred)
    return xgb_model, auc