"""特徴量の作成（学習とアプリの両方で同じ処理を使う）"""
import numpy as np
import pandas as pd

# 発注機関（契約部署の先頭の局名）
AGENCIES = ["財務局", "建設局", "水道局", "警視庁", "東京消防庁", "その他"]
WORK_TYPES = ["受変電", "幹線・動力", "照明・LED", "弱電・通信", "防災設備",
              "太陽光・蓄電", "改修一式", "新築一式", "その他"]
PUBLISH = ["事前公表", "事後公表"]
FLOOR_TYPES = ["最低制限価格", "調査基準価格", "基準価格"]

# 東京都 電気工事の等級別発注標準金額（税抜・円）。年度ごとに見直されるため要確認
CATEGORY_BOUNDS = [(55_000_000, "A"), (18_000_000, "B"), (6_000_000, "C"), (0, "D")]

NUM_COLS = ["log_planned_price", "work_days"]
CAT_COLS = ["agency", "work_type", "publish", "floor_type"]


def price_category(price: float) -> str:
    for lower, cat in CATEGORY_BOUNDS:
        if price >= lower:
            return cat
    return "D"


def agency_of(dept: str) -> str:
    for a in AGENCIES[:-1]:
        if str(dept).startswith(a):
            return a
    return "その他"


def load_dataset(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, parse_dates=["open_date", "end_date"])
    df = df[df["win_price"].notna()].copy()          # 不調（落札者なし）は除外
    df["rate"] = df["win_price"] / df["planned_price"]  # 目的変数：落札率
    df["agency"] = df["dept"].map(agency_of)
    df["work_days"] = (df["end_date"] - df["open_date"]).dt.days
    df["category"] = df["planned_price"].map(price_category)
    return df.sort_values("open_date").reset_index(drop=True)


class GroupMeanRegressor:
    """比較用の基準モデル：同じグループ（基準価格の種類）の平均落札率をそのまま返す"""

    def __init__(self, col="floor_type"):
        self.col = col

    def fit(self, X, y):
        y = pd.Series(np.asarray(y), index=X.index)
        self.means_ = y.groupby(X[self.col]).mean().to_dict()
        self.overall_ = float(y.mean())
        return self

    def predict(self, X):
        return np.array([self.means_.get(v, self.overall_) for v in X[self.col]])

    def get_params(self, deep=True):
        return {"col": self.col}

    def set_params(self, **p):
        self.col = p.get("col", self.col)
        return self


def make_features(df: pd.DataFrame) -> pd.DataFrame:
    """入札前に分かる項目だけで特徴量を作る（入札者数・辞退数などは使わない）"""
    X = pd.DataFrame(index=df.index)
    X["log_planned_price"] = np.log10(df["planned_price"].astype(float))
    X["work_days"] = df["work_days"].astype(float)
    for c in CAT_COLS:
        X[c] = df[c].astype(str)
    return X
