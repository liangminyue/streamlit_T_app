import streamlit as st
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import time
import shap  # 新增import语句
import matplotlib.pyplot as plt  # 新增import语句
import os  # 新增import语句

# 主题配置
st.set_page_config(
    page_title="HGB预测系统",
    page_icon="🩸",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# 加载模型和缩放器
try:
    model = joblib.load('xgboost_model.pkl')
    if not os.path.exists('scaler.pkl'):
        st.error("scaler.pkl文件未找到！")
    else:
        scaler = joblib.load('scaler.pkl')
except Exception as e:
    st.error(f"加载scaler失败: {str(e)}")

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

        # 新增SHAP解释器
        explainer = shap.Explainer(model)
        shap_values = explainer.shap_values(scaled_data)
        
        # 创建SHAP可视化
        st.subheader("特征影响分析")
        
        # 方式1：使用瀑布图（推荐单个预测解释）
        fig, ax = plt.subplots()
        shap.plots.waterfall(shap.Explanation(values=shap_values[0], 
                                            base_values=explainer.expected_value,
                                            data=scaled_data[0],
                                            feature_names=input_data.columns),
                            max_display=10,
                            show=False)
        st.pyplot(fig)
        
        # 方式2：使用force_plot（需要安装ipython）
        # force_plot = shap.force_plot(explainer.expected_value,
        #                             shap_values[0],
        #                             scaled_data[0],
        #                             feature_names=input_data.columns,
        #                             matplotlib=False)
        # st.components.v1.html(force_plot.html(), height=400)
        
        # 在SHAP可视化后添加特征重要性说明
        st.markdown("""
        **图例说明：**
        -  红色特征：提升最终预测值
        -  蓝色特征：降低最终预测值
        - 条形长度：影响程度绝对值
        - 基础值：{:.2f} (模型在训练集上的平均预测值)
        """.format(explainer.expected_value))