import os
import asyncio
from openai import AsyncOpenAI

client = AsyncOpenAI(api_key = os.environ["OPENAI_KEY"])

async def get_chat_completion(prompt, model="gpt-4"):
    response = await client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}]
    )

    return response.choices[0].message.content
