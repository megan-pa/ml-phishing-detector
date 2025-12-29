import re
import joblib

MODEL_PATH = "artifacts/best_phishing_model.pkl"

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = re.sub(r"\s+", " ", text).strip()
    return text

def load_model(model_path):
    loaded_model = joblib.load(model_path)
    return loaded_model

def predict_email(email_text, model):
    cleaned_email_text = clean_text(email_text)
    pred = model.predict([cleaned_email_text])[0]
    label = "phishing" if int(pred) == 1 else "legitimate"
    result = {"prediction": int(pred), "label": label}
    return result

if __name__ == "__main__":
    model = load_model(MODEL_PATH)
    
    # placeholder for API input
    email_text = "URGENT: This is a test email. Please verify your account immediately by clicking the link below. http://secure-login.example.com "

    prediction = predict_email(email_text, model)
    print(prediction)
