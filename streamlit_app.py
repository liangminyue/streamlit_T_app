import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import time

# 主题配置
st.set_page_config(
    page_title="HGB预测系统",
    page_icon="🩸",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 加载模型和缩放器
model = joblib.load('xgboost_model.pkl')
scaler = joblib.load('scaler.pkl')

# 页面标题
st.title('血红蛋白(HGB)预测系统')
st.markdown("""
    **XGBoost算法驱动**  
    *请输入患者基本信息与临床参数进行预测*
""")
st.divider()

# 使用说明
with st.expander("ℹ️ 使用说明"):
    st.markdown("""
    - 所有数值输入请参考实际测量值
    - 输血量单位：1U=200ml全血制备的浓缩红细胞
    - 正常HGB参考范围：男性 130-175g/L，女性 115-150g/L
    """)

# 表单输入
with st.form("prediction_form"):
    st.header("输入特征")

    # 使用columns分栏布局
    col1, col2 = st.columns(2)
    with col1:
        age = st.number_input("年龄", min_value=0, max_value=100, value=30)
        height = st.number_input("身高 (cm)", min_value=20.0, max_value=250.0, value=170.0, step=0.1)
        blood_transfusion = st.number_input("本次输血量 (U)", min_value=0, max_value=12, value=0)
        
    with col2:
        gender = st.selectbox("性别", ["男", "女"], index=0)
        weight = st.number_input("体重 (kg)", min_value=10.0, max_value=200.0, value=70.0, step=0.1)
        hgb_before = st.number_input("HGB前值 (g/L)", min_value=20, max_value=200, value=120, help="输血前血红蛋白浓度")

    submitted = st.form_submit_button("开始预测")

if submitted:  # 注意这个判断在with语句块外
    with st.spinner('正在计算中...'):
        # 转换性别为数值
        gender_value = 1 if gender == "男" else 0

        # 构建输入数据
        input_data = pd.DataFrame([[age, gender_value, height, weight, blood_transfusion, hgb_before]],
                                columns=['年龄', '性别', '身高', '体重', '本次输血量', 'HGB前'])

        # 数据缩放
        scaled_data = scaler.transform(input_data)

        # 预测
        prediction = model.predict(scaled_data)

        # 优化结果显示
        st.metric(label="预测HGB值", value=f"{prediction[0]:.2f} g/L", delta=f"较输血前变化 {prediction[0]-hgb_before:.2f}g/L")
        st.caption("注：预测结果仅供参考，实际临床决策需结合其他检查指标")
