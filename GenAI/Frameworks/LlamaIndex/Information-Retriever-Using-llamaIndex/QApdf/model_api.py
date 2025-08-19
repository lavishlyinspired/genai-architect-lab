
from llama_index.llms.google_genai import GoogleGenAI
from dotenv import load_dotenv

import os

GEMINI_API_KEY=os.getenv("GEMINI_API_KEY")
load_dotenv()

def load_model():

    llm = GoogleGenAI(
        model="gemini-2.5-pro",
        api_key=GEMINI_API_KEY,  # uses GOOGLE_API_KEY env var by default
    )

    # resp = llm.complete("Who is Paul Graham?")
    # print(resp)
    return llm

if __name__ == "__main__":
    pass

