from dotenv import load_dotenv
load_dotenv()

import os
from openai import OpenAI

endpoint = os.getenv("AZURE_OPENAI_ENDPOINT")
api_key = os.getenv("AZURE_OPENAI_KEY")

print("Endpoint =", endpoint)
print("Key Exists =", api_key is not None)

client = OpenAI(
    base_url=f"{endpoint}/openai/v1",
    api_key=api_key
)

response = client.chat.completions.create(
    model="Phi-4-mini-instruct",
    messages=[
        {
            "role": "user",
            "content": "What is Human-Computer Interaction?"
        }
    ],
    max_tokens=100
)

print("\nRESPONSE:")
print(response.choices[0].message.content)