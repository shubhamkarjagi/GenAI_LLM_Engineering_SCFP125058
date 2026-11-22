# -*- coding: utf-8 -*-
"""Executable Python version of Lab3/exercise5.ipynb."""

from dotenv import load_dotenv
from groq import Groq
import os
load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY. Add it to the project .env file (see .env.example).")
client = Groq(api_key=GROQ_API_KEY)


def ask_llm(system_prompt,user_prompt):
    message = [
        {
            "role":"system",
            "content":system_prompt
        },
        {
            "role":"user",
            "content":user_prompt
        }
    ]

    chat_completion = client.chat.completions.create(messages=message,model="openai/gpt-oss-120b")
    response = chat_completion.choices[0].message.content
    return response

sentence = input("Enter sentence: ")
target_language = input("Enter target language: ")
formality = input("Enter Formality: ")


SYSTEM_PROMPT = """You are a professional translation assistant.
"""

user_prompt = """Sentence:{sentence} Target Language:{target_language} Formality level:{formality}"""


print(ask_llm(SYSTEM_PROMPT,user_prompt.format(sentence=sentence,target_language=target_language,formality=formality)))


