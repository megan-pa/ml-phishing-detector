import joblib
from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from detector import final_decision

class EmailBatch(BaseModel):
    email_texts: List[str]

app = FastAPI()

BEST_MODEL = "artifacts/best_phishing_model.pkl"
MODEL = joblib.load(BEST_MODEL)

@app.get("/")
def root():
    return {"message": "Detector API running"}

# TODO allow users to input emails as files as well as string
@app.post("/classify_email")
def classify_email(email_text: str):
    detector_result = final_decision(email_text=email_text, model=MODEL)
    return {"classification": detector_result}

@app.post("/classify_email_batch")
def classify_email_batch(email_texts: EmailBatch):
    results = []
    for email in email_texts.email_texts:
        detector_result = final_decision(email_text=email, model=MODEL)
        results.append(detector_result)
    return {"batch_classification": results}

# TODO implement after AI explanation feature is added
@app.post("/explain_email")
def explain_email():
    return {"explanation": "placeholder"}
