from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from audio_utils import audio_to_spectrogram
from model import predict

app = FastAPI(title="Dolphin Sound Classification API")

# Allow requests from the frontend (Vercel)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace "*" with your Vercel URL after deployment
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
def home():
    return {"status": "API is running"}


@app.post("/predict-audio")
async def predict_audio(file: UploadFile = File(...)):
    try:
        print("✅ Received request")

        # Read uploaded audio file
        audio_bytes = await file.read()
        print("✅ Audio read, size:", len(audio_bytes))

        # Convert audio to spectrogram
        spectrogram = audio_to_spectrogram(audio_bytes)
        print("✅ Spectrogram created")

        # Run prediction
        prediction = predict(spectrogram)
        print("✅ Prediction done:", prediction)

        return {
            "prediction": prediction
        }

    except Exception as e:
        print("❌ Backend error:", str(e))
        raise HTTPException(
            status_code=500,
            detail=f"Audio processing failed: {str(e)}"
        )
