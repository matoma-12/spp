import streamlit as st
import joblib
import pandas as pd
from datetime import datetime


# =========================
# 설정
# =========================

st.set_page_config(
    page_title="Solar Power Prediction",
    page_icon="☀️"
)


# =========================
# 모델 불러오기
# =========================

@st.cache_resource
def load_model():
    model = joblib.load("solar_model.pkl")
    return model


model = load_model()


# =========================
# 화면
# =========================

st.title("☀️ 태양광 발전량 예측 AI")

st.write(
    "기상 데이터를 입력하면 AI 모델이 예상 발전량을 예측합니다."
)


st.divider()


# =========================
# 입력값
# =========================

st.subheader("🌤 기상 데이터 입력")


col1, col2 = st.columns(2)


with col1:

    solar = st.number_input(
        "일사량 (MJ/m²)",
        value=15.0
    )

    temp = st.number_input(
        "기온 (℃)",
        value=25.0
    )

    humidity = st.number_input(
        "습도 (%)",
        value=60.0
    )


with col2:

    wind = st.number_input(
        "풍속 (m/s)",
        value=2.0
    )

    previous_power = st.number_input(
        "전날 발전량 (kWh)",
        value=1000.0
    )


# =========================
# 예측
# =========================

if st.button("☀️ 발전량 예측"):


    input_data = pd.DataFrame(
        [[
            solar,
            temp,
            humidity,
            wind,
            previous_power
        ]],

        columns=[
            "solar_radiation",
            "temperature",
            "humidity",
            "wind_speed",
            "previous_power"
        ]
    )


    try:

        result = model.predict(input_data)


        st.success(
            f"예상 발전량 : {result[0]:.2f} kWh"
        )


    except Exception as e:

        st.error("입력 변수 구조가 모델 학습 데이터와 다릅니다.")
        st.write(e)



st.divider()

st.caption(
    "실행 시간 : "
    + datetime.now().strftime("%Y-%m-%d %H:%M")
)
