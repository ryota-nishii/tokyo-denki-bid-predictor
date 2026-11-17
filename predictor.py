"""学習済みモデルを読み込み、1件分の予測を行う"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import agency_of, make_features, price_category

ROOT = Path(__file__).parent


class Predictor:
    def __init__(self, model_dir: Path = ROOT / "model"):
        import sklearn
        try:
            self.metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
            if self.metrics.get("sklearn_version") != sklearn.__version__:
                raise ValueError("scikit-learn のバージョンが学習時と違う")
            self.model = joblib.load(model_dir / "model.joblib")
        except Exception:
            # モデルがない・バージョン違いで読み込めない場合は、その場で学習し直す
            import train
            train.main()
            self.metrics = json.loads((model_dir / "metrics.json").read_text(encoding="utf-8"))
            self.model = joblib.load(model_dir / "model.joblib")

    def predict(self, planned_price: float, publish: str, floor_type: str,
                agency: str, work_type: str, work_days: float | None) -> dict:
        row = pd.DataFrame([{
            "planned_price": planned_price,
            "publish": publish,
            "floor_type": floor_type,
            "agency": agency_of(agency),
            "work_type": work_type,
            "work_days": np.nan if work_days is None else work_days,
        }])
        rate = float(self.model.predict(make_features(row))[0])
        rate = min(max(rate, 0.80), 1.00)  # 予定価格を超える落札はないため上限100%
        err = self.metrics["error80_pt"] / 100
        lo, hi = max(rate - err, 0.80), min(rate + err, 1.00)
        cat = price_category(planned_price)
        hist = [h for h in self.metrics["history"]
                if h["floor_type"] == floor_type and h["category"] == cat]
        return {
            "rate": rate,
            "rate_low": lo,
            "rate_high": hi,
            "price": planned_price * rate,
            "price_low": planned_price * lo,
            "price_high": planned_price * hi,
            "category": cat,
            "history": hist[0] if hist else None,
        }
