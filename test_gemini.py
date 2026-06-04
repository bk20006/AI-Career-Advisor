from dotenv import load_dotenv
import os
import google.generativeai as genai

load_dotenv()

import os

genai.configure(
    api_key=os.getenv("GEMINI_API_KEY")
)

model = genai.GenerativeModel(
    "gemini-2.5-flash"
)

response = model.generate_content(
    "Say hello in one sentence."
)

print(response.text)