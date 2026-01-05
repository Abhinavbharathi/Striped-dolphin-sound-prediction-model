import streamlit as st
import requests
import wave
import contextlib

# ---------------- API URL ----------------
API_URL = "http://127.0.0.1:8000/predict-audio"

# ---------------- Page Config ----------------
st.set_page_config(
    page_title="Striped Dolphin Vocal Classification",
    layout="centered"
)

# ---------------- CSS Styling ----------------
st.markdown("""
<style>
.title {
    color: #007BFF;   /* vivid blue */
    font-size: 28px;
    font-weight: bold;
    text-align: center;
    margin-bottom: 20px;
}

.upload-box {
    border: 2px dashed #7a7a7a;  /* grey */
    padding: 40px;
    text-align: center;
    margin-bottom: 15px;
    font-size: 18px;
    color: #4f4f4f;
}

.stButton > button {
    background-color: #1f77b4;
    color: white;
    font-size: 16px;
    padding: 8px 25px;
    border-radius: 6px;
}

.result-box {
    border: 2px solid #7a7a7a;   /* grey */
    padding: 15px;
    font-size: 18px;
    font-weight: bold;
    margin-top: 20px;
    text-align: center;
    color: white;               /* RESULT TEXT → WHITE */
    background-color: #4f4f4f;  /* dark grey background */
}

</style>
""", unsafe_allow_html=True)

# ---------------- Helper: Audio Duration ----------------
def get_audio_duration(file):
    with contextlib.closing(wave.open(file, 'rb')) as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
        duration = frames / float(rate)
    return round(duration, 2)

# ---------------- Title ----------------
st.markdown(
    '<div class="title">Striped Dolphin Vocal Classification</div>',
    unsafe_allow_html=True
)

# ---------------- Upload Section ----------------
st.markdown(
    '<div class="upload-box">➕<br/>Upload WAV Audio File</div>',
    unsafe_allow_html=True
)

uploaded_file = st.file_uploader("", type=["wav"])

# ---------------- Audio Preview + Duration ----------------
if uploaded_file:
    st.audio(uploaded_file)

    duration = get_audio_duration(uploaded_file)
    st.markdown(f"🎵 **Audio Duration:** {duration} seconds")

    if duration > 5:
        st.warning("Only the first 5 seconds will be used for prediction")

# ---------------- Verify Button ----------------
if st.button("Verify"):
    if uploaded_file is None:
        st.warning("Please upload an audio file")
    else:
        with st.spinner("Analyzing dolphin sound..."):
            files = {
                "file": (uploaded_file.name, uploaded_file.getvalue(), "audio/wav")
            }
            response = requests.post(API_URL, files=files)

        if response.status_code == 200:
            prediction = response.json()["prediction"].upper()
            st.markdown(
                f'<div class="result-box">Result: {prediction}</div>',
                unsafe_allow_html=True
            )
        else:
            st.error("Prediction failed. Please try again.")
