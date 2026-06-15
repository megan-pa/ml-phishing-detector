import os
import asyncio
from pathlib import Path
from dotenv import load_dotenv

BASE_DIR = Path(__file__).resolve().parent
load_dotenv(BASE_DIR / "prod.env", override=True) 

import joblib
from explanation import get_chat_completion

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "best_phishing_model.pkl"

API_KEY = os.environ["OPENAI_KEY"]

def final_decision(email_text, model):
    ml_score = float(model.decision_function([email_text])[0])
    
    if ml_score >= 0.7:
        return {
            "label": "phishing",
            "reason": "ML prediction",
            "ml_score": ml_score,
        }

    return {
        "label": "legitimate",
        "reason": "Low risk score",
        "ml_score": ml_score,
    }

async def ai_result_explanation(email_text, result):
    response = await get_chat_completion(
        prompt = f"""
        You are a cybersecurity assistant. You have been tasked with determining whether an email you have received is either phishing or legitiamte. 
        The email below has been analysed using a machine learning model and a rule-based detection system. 

        Your tasked with explaining the classification outcome in a clear and user-friendly manner. 

        Here is the email text:
        \"\"\"{email_text}\"\"\"

        Classification result:
        {result}

        You should structure your response as follows with these subheadings:
        1. Final verdict, make use of ML and risk score here (phishing or legitimate)
        2. Key indicators in the email that influenced the decision, focus on the actual email text rather than the ML and risk score (bullet points)
        3. Why there indicators matter (1-2 sentences)
        """
    )

    return response

if __name__ == "__main__":
    async def main():
        best_model = joblib.load(MODEL_PATH)
        result = final_decision("This is a test email, urgent", best_model)
        explanation = await ai_result_explanation("This is a test email, urgent", result)

        print(result)
        print(explanation)
    
    asyncio.run(main())
