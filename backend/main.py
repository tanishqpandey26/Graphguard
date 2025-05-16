from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from prediction import fraud_detection_pipeline
import pandas as pd
app = FastAPI()

# Allow CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.post("/predict")
async def get_prediction(request: Request):
        data = await request.json()
        df = pd.DataFrame([data])  # convert dict to DataFrame
        result = fraud_detection_pipeline (
        input_data= df,
        model_path="model\model.pt",
        artifacts_dir="model",
        hidden_channels=32
        )
    
        print("\nPrediction Result:")
        print(f"  {'Fraud Probability:':<20} {result.get('fraud_probability', 'N/A'):.6f}")
        print(f"  {'Classification:':<20} {result.get('prediction', 'N/A')}")
   
        print(result)
        return result