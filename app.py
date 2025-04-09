import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# Set dark theme with red highlights
st.set_page_config(page_title="Heart Disease App", layout="centered", initial_sidebar_state="expanded")
st.markdown("""
    <style>
    body {
        background-color: #111111;
    }
    .main {
        background-color: #111111;
        color: #ffffff;
    }
    h1, h2, h3, h4, h5, h6 {
        color: #e50914;
    }
    .st-bf, .st-at, .st-cx, .st-cu {
        background-color: #111111 !important;
        color: #ffffff !important;
    }
    .stButton>button {
        background-color: #e50914;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 0.5em 1.5em;
    }
    .stButton>button:hover {
        background-color: #ff3b3f;
        color: white;
    }
    .stRadio > div {
        background-color: #222222;
        border-radius: 10px;
        padding: 0.5em;
    }
    .stSelectbox, .stSlider, .stTextInput {
        background-color: #222222;
        color: white;
    }
    .stAlert {
        border-left: 5px solid #e50914;
    }
    </style>
""", unsafe_allow_html=True)

# Load model and encoders
model = joblib.load("heart_model.pkl")
le_dict = joblib.load("label_encoders.pkl")

# Sidebar menu
menu = st.sidebar.radio("เลือกเมนู", ["🔍 ทำนายโรคหัวใจ", "📢 ข่าวบุหรี่ไฟฟ้า"])

# หน้า 1: ทำนายโรคหัวใจ
if menu == "🔍 ทำนายโรคหัวใจ":
    st.title("Heart Disease Prediction App 💓")
    st.write("กรอกข้อมูลด้านล่างเพื่อทำนายความเสี่ยงโรคหัวใจ")

    def user_input_form():
        gender = st.selectbox("Gender", le_dict['Sex'].classes_)
        smoking = st.selectbox("Smoking", le_dict['Smoking'].classes_)
        alcohol = st.selectbox("AlcoholDrinking", le_dict['AlcoholDrinking'].classes_)
        stroke = st.selectbox("Stroke", le_dict['Stroke'].classes_)
        physical = st.selectbox("PhysicalActivity", le_dict['PhysicalActivity'].classes_)
        diff_walk = st.selectbox("DiffWalking", le_dict['DiffWalking'].classes_)
        diabetic = st.selectbox("Diabetic", le_dict['Diabetic'].classes_)
        asthma = st.selectbox("Asthma", le_dict['Asthma'].classes_)
        kidney = st.selectbox("KidneyDisease", le_dict['KidneyDisease'].classes_)
        skin = st.selectbox("SkinCancer", le_dict['SkinCancer'].classes_)
        bmi = st.slider("BMI", 10.0, 50.0, 25.0)
        age = st.selectbox("AgeCategory", le_dict['AgeCategory'].classes_)
        race = st.selectbox("Race", le_dict['Race'].classes_)
        genhealth = st.selectbox("GenHealth", le_dict['GenHealth'].classes_)
        sleep = st.slider("SleepTime (hrs/day)", 0, 24, 7)
        mental = st.slider("MentalHealth (days/month)", 0, 30, 0)
        physical_health = st.slider("PhysicalHealth (days/month)", 0, 30, 0)

        data = {
            'BMI': bmi,
            'Smoking': le_dict['Smoking'].transform([smoking])[0],
            'AlcoholDrinking': le_dict['AlcoholDrinking'].transform([alcohol])[0],
            'Stroke': le_dict['Stroke'].transform([stroke])[0],
            'PhysicalHealth': physical_health,
            'MentalHealth': mental,
            'DiffWalking': le_dict['DiffWalking'].transform([diff_walk])[0],
            'Sex': le_dict['Sex'].transform([gender])[0],
            'AgeCategory': le_dict['AgeCategory'].transform([age])[0],
            'Race': le_dict['Race'].transform([race])[0],
            'Diabetic': le_dict['Diabetic'].transform([diabetic])[0],
            'PhysicalActivity': le_dict['PhysicalActivity'].transform([physical])[0],
            'GenHealth': le_dict['GenHealth'].transform([genhealth])[0],
            'SleepTime': sleep,
            'Asthma': le_dict['Asthma'].transform([asthma])[0],
            'KidneyDisease': le_dict['KidneyDisease'].transform([kidney])[0],
            'SkinCancer': le_dict['SkinCancer'].transform([skin])[0]
        }

        return pd.DataFrame([data])

    input_df = user_input_form()

    if st.button("Predict"):
        prediction = model.predict(input_df)
        prob = model.predict_proba(input_df)[0][1]

        st.subheader("ผลการทำนาย:")
        if prediction[0] == 1:
            st.error(f"⚠️ ความเสี่ยงโรคหัวใจสูง ({prob*100:.2f}%)")
        else:
            st.success(f"✅ ความเสี่ยงโรคหัวใจต่ำ ({prob*100:.2f}%)")

# หน้า 2: ข่าวบุหรี่ไฟฟ้า
elif menu == "📢 ข่าวบุหรี่ไฟฟ้า":
    st.title("📢 ข่าวสารและสถิติเกี่ยวกับบุหรี่ไฟฟ้า")

    st.header("🗞 ข่าวล่าสุดเกี่ยวกับบุหรี่ไฟฟ้า")
    st.markdown("""
    - **[ราชกิจจาฯ ประกาศเพิ่ม “บุหรี่ไฟฟ้า-บารากู่” ในกฎกระทรวงกำหนดความประพฤติ นร.-นศ.](https://www.thairath.co.th/news/politic/2851450)** – เมื่อวันที่ 4 เมษายน 2568 ราชกิจจานุเบกษาเผยแพร่กฎกระทรวงใหม่ที่กำหนดความประพฤติของนักเรียนและนักศึกษา โดยได้มีการ เพิ่ม “บุหรี่ไฟฟ้า” และ “บารากู่ไฟฟ้า” เข้าไปในรายการพฤติกรรมต้องห้าม เช่นเดียวกับสุรา บุหรี่ และยาเสพติดเดิม
    - **[พบ! เด็กสูบบุหรี่ไฟฟ้าตั้งแต่ประถมศึกษา ไม่รู้ว่าเป็นสิ่งเสพติดอันตราย](https://www.hfocus.org/content/2023/11/28938)** – VicHealth สสส.แคว้นวิกตอเรีย เผย บุหรี่ไฟฟ้าปัญหาความท้าทายใหม่ของโลก ตัวการทำลายสุขภาพ เสี่ยงเด็กเป็นนักเสพสูง เฉพาะออสเตรเลียเริ่มสูบอายุ 14 ปี เด็กสูบบุหรี่ไฟฟ้ามากกว่าบุหรี่ธรรมดา 3 เท่า เร่งทุกประเทศให้ความสำคัญการจัดการกับปัจจัยการค้ากำหนดสุขภาพ นักวิชาการไทย ชี้ เด็ก เยาวชนไม่รู้ว่าบุหรี่ไฟฟ้าอันตราย เหตุธุรกิจยาสูบบิดเบือนข้อเท็จจริง
    """)

    st.header("📊 จำนวนผู้ใช้บุหรี่ไฟฟ้าในไทย (จำลอง)")

    vape_data = pd.DataFrame({
        "ปี": [2019, 2020, 2021, 2022, 2023, 2024],
        "จำนวนผู้ใช้ (คน)": [150000, 200000, 250000, 300000, 380000, 450000]
    })

    fig = px.line(
        vape_data,
        x="ปี",
        y="จำนวนผู้ใช้ (คน)",
        markers=True,
        title="แนวโน้มจำนวนผู้ใช้บุหรี่ไฟฟ้าในไทย",
        labels={"จำนวนผู้ใช้ (คน)": "จำนวนผู้ใช้", "ปี": "ปี"},
    )

    fig.update_traces(line_color='red')
    fig.update_layout(
        title_font_size=20,
        title_x=0.5,
        hovermode="x unified",
        template="plotly_dark"
    )

    st.plotly_chart(fig, use_container_width=True)
    st.info("📌 หมายเหตุ: ข้อมูลจำลองเพื่อการนำเสนอ หากต้องการใช้ข้อมูลจริงสามารถเชื่อมกับแหล่งข้อมูลภายนอกได้")
