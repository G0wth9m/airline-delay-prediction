
from sklearn.metrics import mean_absolute_error, mean_squared_error, roc_auc_score, average_precision_score
import numpy as np

def regression_report(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    return {"MAE": mae, "RMSE": rmse}

def classification_report(y_true, proba):
    auc = roc_auc_score(y_true, proba)
    ap = average_precision_score(y_true, proba)
    return {"ROC_AUC": auc, "AveragePrecision": ap}
