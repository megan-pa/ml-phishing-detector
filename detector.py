import joblib
from rules import final_risk_score
from pathlib import Path

RISK_SCORE_THRESHOLD = 6

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "best_phishing_model.pkl"

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

if __name__ == "__main__":
    best_model = joblib.load(MODEL_PATH)
    result = final_decision("This is a test email, urgent", best_model)
    print(result)
