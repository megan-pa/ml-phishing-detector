import joblib
from fastapi import FastAPI, File, UploadFile
from pydantic import BaseModel
from typing import List
from detector import final_decision, ai_result_explanation
from contextlib import asynccontextmanager

BEST_MODEL = "artifacts/best_phishing_model.pkl"

@asynccontextmanager
async def lifespan(app: FastAPI):
    try:
        app.state.model = joblib.load(BEST_MODEL)       
    except Exception as e:
        print(f"Error loading model: {e}")  
    yield

app = FastAPI(lifespan=lifespan)

@app.post("/classify_email")
def quick_classify_email(email_text: str):
    detector_result = final_decision(email_text=email_text, model=app.state.model)
    return {"classification": detector_result}

@app.post("/classify_email_file")
async def classify_email(email_file: UploadFile = File(...)):
    contents = await email_file.read()
    detector_result = final_decision(email_text=contents.decode('utf-8'), model=app.state.model)
    return {"classification": detector_result}

@app.post("/classify_email_batch")
async def classify_email_batch(email_files: List[UploadFile] = File(...)):
    results = []

    for email in email_files:
        contents = await email.read()
        detector_result = final_decision(email_text=contents.decode('utf-8'), model=app.state.model)
        results.append(detector_result)
    return {"batch_classification": results}

@app.post("/explain_email")
def explain_email(email_text: str):
    detector_result = final_decision(email_text=email_text, model=app.state.model)
    explanation = ai_result_explanation(email_text=email_text, result=detector_result)
    return {"detector_result": detector_result, "explanation": explanation}
