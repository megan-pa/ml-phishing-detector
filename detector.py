import os
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / "prod.env", override=True) 

import joblib
from rules import final_risk_score
from explanation import get_chat_completion

RISK_SCORE_THRESHOLD = 6

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "best_phishing_model.pkl"

API_KEY = os.environ["OPENAI_KEY"]

def final_decision(email_text, model):
    ml_score = float(model.decision_function([email_text])[0])
    rule_score = final_risk_score(email_text)

    if rule_score >= RISK_SCORE_THRESHOLD:
        return {
            "label": "phishing",
            "reason": "High risk score from rules",
            "ml_score": ml_score,
            "rule_score": rule_score
        }
    
    if ml_score >= 0.7:
        return {
            "label": "phishing",
            "reason": "ML prediction",
            "ml_score": ml_score,
            "rule_score": rule_score
        }

    return {
        "label": "legitimate",
        "reason": "Low risk score",
        "ml_score": ml_score,
        "rule_score": rule_score
    }

def ai_result_explanation(email_text, result):
    response = get_chat_completion(
        prompt = f"""
        You are a cybersecurity assistant. You have been tasked with determining whether an email you have received is either phishing or legitiamte. 
        The email has been run through a ML model and a rule system to generate a risk score of whether the email is legitimate.

        Here is the email text:
        \"\"\"{email_text}\"\"\"

        Classification result:
        {result}

        Explain clearly why this email was classifed as either phishing or legitimate. 
        """
    )

    return response

if __name__ == "__main__":
    best_model = joblib.load(MODEL_PATH)
    result = final_decision("This is a test email, urgent", best_model)
    explanation = ai_result_explanation("This is a test email, urgent", result)

    print(result)
    print(explanation)
