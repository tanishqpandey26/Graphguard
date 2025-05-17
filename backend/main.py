from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from prediction import fraud_detection_pipeline
import pandas as pd
import os
import uvicorn

app = FastAPI()

# Allow CORS for your frontend domains
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000",
        "https://graphguard.vercel.app"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def get_prediction(request: Request):
    data = await request.json()
    df = pd.DataFrame([data])
    result = fraud_detection_pipeline(
        input_data=df,
        model_path="model/model.pt",
        artifacts_dir="model",
        hidden_channels=32
    )

    print("\nPrediction Result:")
    print(f"  {'Fraud Probability:':<20} {result.get('fraud_probability', 'N/A'):.6f}")
    print(f"  {'Classification:':<20} {result.get('prediction', 'N/A')}")
    print(result)

    return result

@app.get("/", response_class=JSONResponse)
@app.head("/")
def health_check():
    return {"status": "Backend is running!"}

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
