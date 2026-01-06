import os
from openai import OpenAI

client = OpenAI(api_key = os.environ["OPENAI_KEY"])

def get_chat_completion(prompt, model="gpt-4"):
    messages = [{"role": "user", "content": prompt}]
    
    response = client.chat.completions.create(
        model=model,
        messages=messages,
        temperature=0
    )

    return response.choices[0].message.content