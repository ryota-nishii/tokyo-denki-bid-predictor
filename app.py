"""東京都 電気工事 落札額予測アプリ（ドラフト版）

起動: streamlit run app.py
"""
import streamlit as st

from features import AGENCIES, FLOOR_TYPES, PUBLISH, WORK_TYPES
from predictor import Predictor

st.set_page_config(page_title="電気工事 落札額予測", page_icon="⚡", layout="centered")


@st.cache_resource
def load_predictor():
    return Predictor()


pred = load_predictor()
m = pred.metrics

st.title("⚡ 電気工事 落札額予測（ドラフト版）")
st.caption(
    "東京都電子調達システムの電気工事案件について、過去の入札結果から落札率と落札額の目安を予測します。"
    "外部積算に依頼する前の「ふるい分け」用です。"
)

with st.form("input"):
    st.subheader("案件の条件")
    planned = st.number_input(
        "予定価格（税抜・円）", min_value=1_000_000, max_value=5_000_000_000,
        value=100_000_000, step=1_000_000, format="%d",
        help="事後公表の案件は、公告や過去の類似案件から見積もった概算額を入れてください。",
    )
    c1, c2 = st.columns(2)
    publish = c1.selectbox("予定価格の公表区分", PUBLISH)
    floor_type = c2.selectbox(
        "基準価格の種類", FLOOR_TYPES,
        help="最低制限価格／調査基準価格（低入札価格調査）／基準価格（総合評価方式）。入札公告で確認できます。",
    )
    c3, c4 = st.columns(2)
    agency = c3.selectbox("発注機関", AGENCIES)
    work_type = c4.selectbox("工事内容区分", WORK_TYPES, index=WORK_TYPES.index("改修一式"))
    days_known = st.checkbox("工期が分かる", value=True)
    days = st.slider("工期（日数）", 30, 1500, 300, step=10, disabled=not days_known)
    submitted = st.form_submit_button("予測する", type="primary")

if submitted:
    r = pred.predict(planned, publish, floor_type, agency, work_type, days if days_known else None)

    st.subheader("予測結果")
    k1, k2 = st.columns(2)
    k1.metric("予測落札率", f"{r['rate']*100:.1f}%")
    k2.metric("予測落札額（税抜）", f"{r['price']/10_000:,.0f} 万円")
    st.write(
        f"おおよその範囲（過去データでの誤差の8割が収まる幅）："
        f"**{r['rate_low']*100:.1f}%〜{r['rate_high']*100:.1f}%**　"
        f"（{r['price_low']/10_000:,.0f}〜{r['price_high']/10_000:,.0f} 万円）"
    )
    st.write(f"金額カテゴリ：**{r['category']}**（東京都 電気工事の発注標準金額による区分）")

    h = r["history"]
    if h:
        st.info(
            f"過去実績（{floor_type}・カテゴリ{r['category']}、{h['count']}件）："
            f"平均 {h['mean']*100:.1f}%、最低 {h['min']*100:.1f}%、最高 {h['max']*100:.1f}%"
        )
    else:
        st.warning("同じ種類・カテゴリの過去実績がまだありません。予測の信頼度は低めです。")

    if floor_type == "最低制限価格":
        st.caption(
            "最低制限価格の案件では、多くの会社が最低制限価格の直上に入札し、同額ならくじ引きになる傾向があります。"
            "有効な入札が1者だけの場合は、落札率が95〜100%になることがあります。"
        )

with st.expander("モデルについて"):
    st.write(
        f"- 学習データ：{m['n_samples']}件（開札日 {m['period'][0]}〜{m['period'][1]}、等級A・多摩地区の電気工事）\n"
        f"- 採用モデル：{m['best_model']}（scikit-learn）\n"
        "- 入力に使う項目：予定価格、公表区分、基準価格の種類、発注機関、工事内容区分、工期。"
        "入札者数など入札前に分からない項目は使っていません。"
    )
    rows = [
        {"モデル": k, "交差検証MAE（pt）": round(v["cv_mae_pt"], 2), "時系列テストMAE（pt）": round(v["time_split_mae_pt"], 2)}
        for k, v in m["results"].items()
    ]
    st.dataframe(rows, hide_index=True)
    st.caption("MAE：落札率の予測誤差の平均（ポイント）。基準モデルは「基準価格の種類ごとの平均落札率」をそのまま返すもの。")

st.divider()
st.caption(
    "出典：東京都電子調達システム 入札情報サービス「入札(見積)経過情報」をもとに作成。"
    "本アプリの予測は参考値であり、入札額の決定や積算の代わりにはなりません。"
)
