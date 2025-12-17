import streamlit as st
import pandas as pd
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier
import time
import matplotlib.pyplot as plt

# --- 1. 設定網頁版面 ---
st.set_page_config(
    page_title="Telco Churn Prediction",
    page_icon="🔮",
    layout="wide"
)

# 加入自訂 CSS
st.markdown("""
    <style>
    .main {background-color: #f8f9fa;}
    .stButton>button {
        width: 100%;
        background-color: #ff4b4b;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px;
    }
    .explanation-box {
        background-color: #e8f4f8;
        padding: 15px;
        border-radius: 5px;
        margin-top: 10px;
        border-left: 5px solid #3498db;
        font-size: 14px;
    }
    </style>
    """, unsafe_allow_html=True)

# --- 定義字典 (介面顯示用，圖表改為英文) ---
FIELD_LABELS = {
    "SeniorCitizen": "是否為高齡者 (Senior Citizen)",
    "tenure": "使用月數 (Tenure)",
    "MonthlyCharges": "月費 (Monthly Charges)",
    "TotalCharges": "總費用 (Total Charges)",
    "InternetService": "網路服務類型 (Internet Service)",
    "Contract": "合約類型 (Contract)",
    "PaymentMethod": "付款方式 (Payment Method)",
    "OnlineSecurity": "網路安全 (Online Security)",
    "OnlineBackup": "雲端備份 (Online Backup)",
    "DeviceProtection": "設備保護 (Device Protection)",
    "TechSupport": "技術支援 (Tech Support)",
    "StreamingTV": "串流電視 (Streaming TV)",
    "StreamingMovies": "串流電影 (Streaming Movies)",
    "MultipleLines": "多線電話 (Multiple Lines)",
    "PhoneService": "電話服務 (Phone Service)",
    "Dependents": "親屬/被撫養人 (Dependents)",
    "Partner": "伴侶 (Partner)",
    "PaperlessBilling": "無紙化帳單 (Paperless Billing)",
    "gender": "性別 (Gender)"
}

OPTION_MAP = {
    "No": "No (無/否)",
    "Yes": "Yes (有/是)",
    "DSL": "DSL (數位迴路)",
    "Fiber optic": "Fiber optic (光纖)",
    "No internet service": "No internet service (無網路服務)",
    "No phone service": "No phone service (無電話服務)",
    "Month-to-month": "Month-to-month (按月)",
    "One year": "One year (一年約)",
    "Two year": "Two year (兩年約)",
    "Electronic check": "Electronic check (電子支票)",
    "Mailed check": "Mailed check (郵寄支票)",
    "Bank transfer (automatic)": "Bank transfer (自動轉帳)",
    "Credit card (automatic)": "Credit card (信用卡自動扣款)",
    "Female": "Female (女性)",
    "Male": "Male (男性)"
}

SERVICE_LABELS = {
    "OnlineSecurity": "網路安全",
    "OnlineBackup": "雲端備份",
    "DeviceProtection": "設備保護",
    "TechSupport": "技術支援",
    "StreamingTV": "串流電視",
    "StreamingMovies": "串流電影",
}

# --- 2. 載入資料與訓練模型 ---
@st.cache_resource
def load_and_train_model():
    try:
        df = pd.read_csv("telco_cleaned_data.csv")
    except FileNotFoundError:
        st.error("找不到 'telco_cleaned_data.csv'，請確認檔案已上傳至同目錄。")
        return None, None, None, None

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce").fillna(0)
    df["ChurnFlag"] = df["Churn"].map({"Yes": 1, "No": 0})
    
    drop_cols = ["customerID", "Churn", "ChurnFlag"]
    X = df.drop(columns=[c for c in drop_cols if c in df.columns])
    y = df["ChurnFlag"]
    
    num_features = ["SeniorCitizen", "tenure", "MonthlyCharges", "TotalCharges"]
    cat_features = [c for c in X.columns if c not in num_features]
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", "passthrough", num_features),
            ("cat", OneHotEncoder(handle_unknown="ignore"), cat_features),
        ]
    )
    
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", DecisionTreeClassifier(max_depth=5, random_state=42))
    ])
    
    model.fit(X, y)
    
    stats = {
        "tenure_mean": int(df["tenure"].mean()),
        "monthly_mean": float(df["MonthlyCharges"].mean()),
        "total_mean": float(df["TotalCharges"].mean()),
        "churn_rate": df["ChurnFlag"].mean(),
        "avg_tenure_churn": df[df["ChurnFlag"]==1]["tenure"].mean(),
        "avg_tenure_no_churn": df[df["ChurnFlag"]==0]["tenure"].mean(),
        "avg_monthly_churn": df[df["ChurnFlag"]==1]["MonthlyCharges"].mean(),
        "avg_monthly_no_churn": df[df["ChurnFlag"]==0]["MonthlyCharges"].mean(),
        "choices": {col: sorted(df[col].unique().tolist()) for col in cat_features}
    }
    
    return model, X.columns.tolist(), stats, cat_features

model, feature_cols, stats, cat_features = load_and_train_model()

if model is None:
    st.stop()

# --- 繪圖函式 (全英文版 - 保證不亂碼) ---
def plot_comparison(user_tenure, user_monthly, stats):
    """繪製使用者與平均值的比較圖 (English Labels)"""
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
    
    # 全英文標籤
    labels = ['Current', 'Retained Avg', 'Churned Avg']
    colors = ['#3498db', '#2ecc71', '#e74c3c'] 
    
    # 1. Tenure
    values = [user_tenure, stats["avg_tenure_no_churn"], stats["avg_tenure_churn"]]
    ax1.bar(labels, values, color=colors, alpha=0.8)
    ax1.set_title("Tenure Comparison")   # 英文標題
    ax1.set_ylabel("Months")             # 英文Y軸
    ax1.axhline(y=user_tenure, color='#3498db', linestyle='--', alpha=0.5)

    # 2. Monthly Charges
    values_money = [user_monthly, stats["avg_monthly_no_churn"], stats["avg_monthly_churn"]]
    ax2.bar(labels, values_money, color=colors, alpha=0.8)
    ax2.set_title("Monthly Fee Comparison") # 英文標題
    ax2.set_ylabel("USD Amount")            # 英文Y軸
    ax2.axhline(y=user_monthly, color='#3498db', linestyle='--', alpha=0.5)

    plt.tight_layout()
    return fig

# --- 3. 側邊欄 ---
st.sidebar.header("📝 客戶資料輸入")
st.sidebar.markdown("請調整下方參數進行預測")

input_data = {}

def format_option(option_value):
    return OPTION_MAP.get(option_value, option_value)

with st.sidebar.form(key='input_form'):
    st.subheader("基本數值 (Basic Info)")
    tenure = st.slider(FIELD_LABELS["tenure"], 0, 72, int(stats["tenure_mean"]))
    monthly = st.number_input(FIELD_LABELS["MonthlyCharges"], 0.0, 120.0, float(stats["monthly_mean"]))
    
    st.write("---")
    use_auto_total = st.checkbox("使用自動計算總費用?", value=True, help="勾選後，將自動使用「月數 x 月費」作為總費用")
    
    if use_auto_total:
        calculated_total = float(tenure * monthly)
        total = st.number_input(FIELD_LABELS["TotalCharges"] + " (Auto)", value=calculated_total, disabled=True)
    else:
        total = st.number_input(FIELD_LABELS["TotalCharges"] + " (Manual)", min_value=0.0, max_value=10000.0, value=float(stats["total_mean"]))
    
    senior = st.selectbox(FIELD_LABELS["SeniorCitizen"], [0, 1], format_func=lambda x: "是 (Yes)" if x==1 else "否 (No)")
    
    input_data.update({
        "tenure": tenure,
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "SeniorCitizen": senior
    })

    st.subheader("服務與合約 (Service & Contract)")
    important_cats = ["InternetService", "Contract", "PaymentMethod", "OnlineSecurity", "TechSupport"]
    other_cats = [c for c in cat_features if c not in important_cats]
    
    for col in important_cats:
        label = FIELD_LABELS.get(col, col)
        val = st.selectbox(label, stats["choices"][col], format_func=format_option)
        input_data[col] = val
        
    with st.expander("更多選項 (其他加值服務與個資)"):
        for col in other_cats:
            label = FIELD_LABELS.get(col, col)
            val = st.selectbox(label, stats["choices"][col], format_func=format_option)
            input_data[col] = val
            
    submit_button = st.form_submit_button(label='🚀 開始預測 (Predict)')

# --- 4. 主畫面 ---

st.title("📊 電信客戶流失預測系統")
st.markdown("### Telco Customer Churn Prediction Dashboard")
st.write("本系統使用機器學習模型分析客戶特徵，並評估其流失風險。")
st.divider()

if submit_button:
    df_input = pd.DataFrame([input_data])
    
    with st.spinner('正在分析客戶畫像...'):
        time.sleep(0.5)
        prediction = model.predict(df_input)[0]
        prob = model.predict_proba(df_input)[0][1]

    # --- 結果頁面 ---
    st.subheader("👤 客戶輪廓摘要")
    m1, m2, m3, m4 = st.columns([2, 1, 1, 1])
    with m1:
        st.metric(label="合約類型", value=format_option(input_data['Contract']).split('(')[0])
    with m2:
        st.metric(label="年資 (Tenure)", value=f"{input_data['tenure']} 個月")
    with m3:
        st.metric(label="月費", value=f"${input_data['MonthlyCharges']:.1f}")
    with m4:
        st.metric(label="總費用", value=f"${input_data['TotalCharges']:.0f}")
    
    st.divider()

    col1, col2 = st.columns([1.5, 1])

    with col1:
        st.subheader("📈 數據比較與解讀")
        st.markdown("**Benchmark Analysis (基準比較)**")
        
        # 繪圖 (使用英文，無需字體檔)
        fig = plot_comparison(input_data['tenure'], input_data['MonthlyCharges'], stats)
        st.pyplot(fig)
        
        # 圖表解讀 (中文說明保留)
        insight_html = "<div class='explanation-box'><b>📊 圖表解讀助手：</b><br>"
        if input_data['tenure'] < stats['avg_tenure_churn']:
            insight_html += "- <span style='color:#e74c3c;'>⚠️ <b>年資過短：</b></span> 此客戶年資低於流失者平均，屬於不穩定期。<br>"
        else:
            insight_html += "- <span style='color:#2ecc71;'>✅ <b>年資穩定：</b></span> 此客戶年資已累積一定長度，忠誠度較高。<br>"
            
        if input_data['MonthlyCharges'] > stats['avg_monthly_churn']:
            insight_html += "- <span style='color:#e74c3c;'>⚠️ <b>資費壓力：</b></span> 月費 <b>高於</b> 流失群體平均，價格可能是流失主因。<br>"
        elif input_data['MonthlyCharges'] < stats['avg_monthly_no_churn']:
            insight_html += "- <span style='color:#2ecc71;'>✅ <b>資費安全：</b></span> 月費低於留存群體平均，價格競爭力強。<br>"
        else:
            insight_html += "- <span style='color:#f39c12;'>ℹ️ <b>資費適中：</b></span> 月費介於平均值之間。<br>"
        insight_html += "<br><i>(Blue=Current, Green=Retained Avg, Red=Churned Avg)</i></div>"
        st.markdown(insight_html, unsafe_allow_html=True)

        st.write("")
        st.markdown("**📦 已訂閱加值服務:**")
        subscribed_services = [ch_label for eng_col, ch_label in SERVICE_LABELS.items() if input_data.get(eng_col) == 'Yes']
        if subscribed_services:
            st.success("  |  ".join(subscribed_services))
        else:
            st.caption("無訂閱任何加值服務")

    with col2:
        st.subheader("🎯 預測判讀")
        st.write(f"流失機率: **{prob:.1%}**")
        st.progress(prob)
        
        if prob < 0.3:
            st.success("✅ **低風險 (Low Risk)**")
            st.info("💡 **建議**: 維持現有服務品質。")
        elif prob < 0.6:
            st.warning("⚠️ **中風險 (Medium Risk)**")
            st.info("💡 **建議**: 優先檢查合約或提供續約優惠。")
        else:
            st.error("🚨 **高風險 (High Risk)**")
            st.info("💡 **建議**: 立即介入並提供挽留方案。")

else:
    # --- 預設畫面 ---
    st.info("👈 請從左側側邊欄輸入客戶資料，並點擊「開始預測」按鈕。")
    st.subheader("📊 資料集概況 (Dataset Overview)")
    
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.markdown("**整體流失比例 (Overall Churn Rate)**")
        sizes = [stats['churn_rate'], 1-stats['churn_rate']]
        fig1, ax1 = plt.subplots(figsize=(5, 4))
        # 圓餅圖使用英文標籤
        wedges, texts, autotexts = ax1.pie(
            sizes, labels=['Churn', 'Retain'], 
            autopct='%1.1f%%', colors=['#e74c3c', '#2ecc71'], 
            startangle=90, 
            textprops=dict(color="black")
        )
        ax1.axis('equal') 
        st.pyplot(fig1)
    
    st.write("")
    st.markdown("#### 📚 歷史資料預覽 (Historical Data Preview)")
    st.dataframe(pd.read_csv("telco_cleaned_data.csv").head(10), use_container_width=True)

st.markdown("---")
st.caption("Designed for Machine Learning Final Project | 2025")
# --- QR Code ---
st.sidebar.markdown("---")
st.sidebar.subheader("📱 手機體驗")
share_url = "https://telco-churn-app-njwb97mjvapp5eoawhyqcsd.streamlit.app" 
st.sidebar.image(
    f"https://api.qrserver.com/v1/create-qr-code/?size=150x150&data={share_url}",
    caption="掃描 QR Code 分享"
)
