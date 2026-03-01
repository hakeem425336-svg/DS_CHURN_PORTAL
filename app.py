import streamlit as st
import numpy as np
import pandas as pd
import pickle
import tensorflow as tf

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="Customer Churn Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---------------- CUSTOM CSS ---------------- #
st.markdown("""
<style>

[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #0f2027, #203a43, #2c5364);
    color: white;
}

.main-title {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    color: #ffffff;
    animation: fadeIn 2s ease-in-out;
}

.sub-text {
    text-align: center;
    font-size: 18px;
    color: #cccccc;
    margin-bottom: 30px;
}

.card {
    background: rgba(255, 255, 255, 0.08);
    padding: 25px;
    border-radius: 20px;
    backdrop-filter: blur(10px);
    box-shadow: 0px 0px 25px rgba(0,0,0,0.6);
}

.stButton>button {
    width: 100%;
    border-radius: 10px;
    height: 50px;
    background: linear-gradient(90deg, #00c6ff, #0072ff);
    color: white;
    font-size: 18px;
    font-weight: bold;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg, #0072ff, #00c6ff);
}

@keyframes fadeIn {
    0% {opacity: 0;}
    100% {opacity: 1;}
}

.result-box {
    text-align: center;
    font-size: 30px;
    font-weight: bold;
    padding: 20px;
    border-radius: 15px;
    margin-top: 20px;
    animation: fadeIn 1.5s ease-in-out;
}

.success-box {
    background: linear-gradient(90deg, #11998e, #38ef7d);
}

.danger-box {
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
}

</style>
""", unsafe_allow_html=True)

# ---------------- LOAD MODEL ---------------- #
model = tf.keras.models.load_model('model.h5')

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender = pickle.load(file)

with open('one_hot_geo.pkl','rb') as file:
    one_hot_geo = pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler = pickle.load(file)

# ---------------- HEADER ---------------- #
st.markdown('<div class="main-title">🏦 Customer Churn Prediction</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-text">AI Powered Banking Retention Dashboard</div>', unsafe_allow_html=True)

# ---------------- INPUT SECTION ---------------- #
with st.container():
    st.markdown('<div class="card">', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        geography = st.selectbox('🌍 Geography', one_hot_geo.categories_[0])
        gender = st.selectbox('👤 Gender', label_encoder_gender.classes_)
        age = st.slider('🎂 Age', 18, 90, 30)
        credit_score = st.number_input('💳 Credit Score', 0, 900)

    with col2:
        balance = st.number_input('💰 Balance', min_value=0.0)
        estimated_salary = st.number_input('💵 Estimated Salary', min_value=0.0)
        tenure = st.slider('📅 Tenure', 0, 10, 3)
        num_of_products = st.slider('📦 Number of Products', 0, 4, 1)

    has_cr_card = st.selectbox('💳 Has Credit Card', [0, 1])
    is_active_member = st.selectbox('🔥 Is Active Member', [0, 1])

    predict_button = st.button("🚀 Predict Customer Status")

    st.markdown('</div>', unsafe_allow_html=True)

# ---------------- PREDICTION ---------------- #
if predict_button:

    input_data = pd.DataFrame({
        'CreditScore': [credit_score],
        'Gender': [label_encoder_gender.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [num_of_products],
        'HasCrCard': [has_cr_card],
        'IsActiveMember': [is_active_member],
        'EstimatedSalary': [estimated_salary]
    })

    geo_encoded = one_hot_geo.transform([[geography]]).toarray()
    geo_encoded_df = pd.DataFrame(
        geo_encoded,
        columns=one_hot_geo.get_feature_names_out(['Geography'])
    )

    input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)
    input_scaled = scaler.transform(input_data)

    prediction = model.predict(input_scaled)
    prediction_proba = prediction[0][0]

    st.markdown("### 📊 Prediction Result")

    st.progress(float(prediction_proba))

    if prediction_proba > 0.5:
        st.markdown(
            f'<div class="result-box danger-box">⚠️ High Risk of Churn<br>{prediction_proba:.2%}</div>',
            unsafe_allow_html=True
        )
    else:
        st.markdown(
            f'<div class="result-box success-box">✅ Customer Likely to Stay<br>{prediction_proba:.2%}</div>',
            unsafe_allow_html=True
        )