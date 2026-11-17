"""落札率予測モデルの学習・評価スクリプト

実行: python train.py
出力: model/model.joblib（学習済みモデル）, model/metrics.json（評価結果）
"""
import json
from pathlib import Path

import joblib
import sklearn
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from features import CAT_COLS, NUM_COLS, GroupMeanRegressor, load_dataset, make_features

ROOT = Path(__file__).parent
SEED = 42


def preprocessor(scale: bool) -> ColumnTransformer:
    num_steps = [("impute", SimpleImputer(strategy="median"))]
    if scale:
        num_steps.append(("scale", StandardScaler()))
    return ColumnTransformer([
        ("num", Pipeline(num_steps), NUM_COLS),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_COLS),
    ])


MODELS = {
    "基準モデル（種類別の平均）": GroupMeanRegressor("floor_type"),
    "リッジ回帰": Pipeline([("pre", preprocessor(True)), ("model", Ridge(alpha=3.0))]),
    "ランダムフォレスト": Pipeline([("pre", preprocessor(False)), ("model", RandomForestRegressor(
        n_estimators=300, min_samples_leaf=3, random_state=SEED))]),
    "勾配ブースティング": Pipeline([("pre", preprocessor(False)), ("model", GradientBoostingRegressor(
        n_estimators=150, max_depth=2, learning_rate=0.05, subsample=0.8, random_state=SEED))]),
}


def time_split_mae(model, X, y, test_ratio=0.2):
    """開札日の古い80%で学習し、新しい20%でテスト（実際の使い方に近い評価）"""
    n_test = max(1, int(round(len(X) * test_ratio)))
    m = clone(model).fit(X.iloc[:-n_test], y.iloc[:-n_test])
    return mean_absolute_error(y.iloc[-n_test:], m.predict(X.iloc[-n_test:])), n_test


def repeated_cv(model, X, y):
    """件数が少ないため、5分割×10回の交差検証で誤差を安定して測る"""
    rkf = RepeatedKFold(n_splits=5, n_repeats=10, random_state=SEED)
    maes, oof = [], []
    for tr, te in rkf.split(X):
        m = clone(model).fit(X.iloc[tr], y.iloc[tr])
        p = m.predict(X.iloc[te])
        maes.append(mean_absolute_error(y.iloc[te], p))
        oof.extend(np.abs(y.iloc[te].values - p))
    return float(np.mean(maes)), float(np.std(maes)), np.array(oof)


def main():
    df = load_dataset(ROOT / "data" / "bids.csv")
    X, y = make_features(df), df["rate"]
    print(f"学習データ: {len(df)}件（{df['open_date'].min():%Y-%m-%d}〜{df['open_date'].max():%Y-%m-%d}）")

    results, errors = {}, {}
    for name, model in MODELS.items():
        cv_mae, cv_std, oof = repeated_cv(model, X, y)
        ts_mae, n_test = time_split_mae(model, X, y)
        results[name] = {"cv_mae_pt": cv_mae * 100, "cv_std_pt": cv_std * 100, "time_split_mae_pt": ts_mae * 100}
        errors[name] = oof
        print(f"{name:<16} 交差検証MAE {cv_mae*100:5.2f}pt (±{cv_std*100:.2f})  時系列テストMAE {ts_mae*100:5.2f}pt")

    candidates = {k: v for k, v in results.items() if not k.startswith("基準")}
    best = min(candidates, key=lambda k: candidates[k]["cv_mae_pt"])
    baseline = next(k for k in results if k.startswith("基準"))
    print(f"\n採用モデル: {best}")

    final = clone(MODELS[best]).fit(X, y)
    (ROOT / "model").mkdir(exist_ok=True)
    joblib.dump(final, ROOT / "model" / "model.joblib")

    # 予測の幅：交差検証の絶対誤差の80%点を「おおよその誤差の範囲」としてアプリに表示する
    err80 = float(np.quantile(errors[best], 0.8))

    # 参考：種類・カテゴリ別の過去実績（アプリで比較表示）
    hist = (df.groupby(["floor_type", "category"])["rate"]
              .agg(["count", "mean", "min", "max"]).reset_index())

    metrics = {
        "n_samples": int(len(df)),
        "period": [str(df["open_date"].min().date()), str(df["open_date"].max().date())],
        "results": results,
        "best_model": best,
        "baseline_model": baseline,
        "error80_pt": err80 * 100,
        "history": hist.to_dict(orient="records"),
        "features": NUM_COLS + CAT_COLS,
        "sklearn_version": sklearn.__version__,
    }
    (ROOT / "model" / "metrics.json").write_text(json.dumps(metrics, ensure_ascii=False, indent=1), encoding="utf-8")

    # 特徴量の重要度（木のモデルのみ）
    est = final.named_steps.get("model") if hasattr(final, "named_steps") else None
    if est is not None and hasattr(est, "feature_importances_"):
        names = final.named_steps["pre"].get_feature_names_out()
        imp = pd.Series(est.feature_importances_, index=names).sort_values(ascending=False)
        print("\n重要度の高い特徴量:")
        print(imp.head(8).round(3).to_string())


if __name__ == "__main__":
    main()
