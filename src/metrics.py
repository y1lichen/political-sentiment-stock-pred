import numpy as np
from sklearn.metrics import f1_score
from scipy import stats

def compute_f1_scores(y_true, y_pred): 
    return f1_score(y_true, y_pred, average="macro", zero_division=0)

def test_incremental_power(market_only_preds, full_model_preds, y_true):
    market_correct, full_correct = (market_only_preds == y_true), (full_model_preds == y_true)
    b, c = ((market_correct) & (~full_correct)).sum(), ((~market_correct) & (full_correct)).sum()  
    chi2 = (abs(b - c) - 1) ** 2 / (b + c) if (b + c) > 0 else 0
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    print(f"   [McNemar's test] χ²={chi2:.4f}, p={p_value:.4f} | 文本模型多答對 {c} 題，少答對 {b} 題")
    return p_value