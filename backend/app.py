from fastapi import FastAPI, UploadFile, File, HTTPException
from audio_utils import audio_to_spectrogram
from model import predict

app = FastAPI(title="Dolphin Sound Classification API")


@app.get("/")
def home():
    return {"status": "API is running"}


@app.post("/predict-audio")
async def predict_audio(file: UploadFile = File(...)):
    try:
        print("✅ Received request")

        # Read file ONCE
        audio_bytes = await file.read()
        print("✅ Audio read, size:", len(audio_bytes))

        # Convert to spectrogram
        spectrogram = audio_to_spectrogram(audio_bytes)
        print("✅ Spectrogram created")

        # Predict
        prediction = predict(spectrogram)
        print("✅ Prediction done:", prediction)

        return {
            "prediction": prediction
        }

    except Exception as e:
        print("❌ Backend error:", str(e))
        raise HTTPException(
            status_code=500,
            detail="Audio processing failed"
        )
