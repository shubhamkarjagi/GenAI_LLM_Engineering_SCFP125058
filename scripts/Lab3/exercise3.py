# -*- coding: utf-8 -*-
"""Executable Python version of Lab3/exercise3.ipynb."""

from groq import Groq
from dotenv import load_dotenv
import os
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY. Add it to the project .env file (see .env.example).")
client = Groq(api_key = GROQ_API_KEY)

SYSTEM_PROMPT = """You are an expert resume information extraction system. Extract **only the fields requested by the user** from the provided resume text."""


resume_text = input("Enter the resume text: ")
fields_to_extract = input("Enter the fields to extract: ")

user_prompt = f"Resume Text: {resume_text}\n Fields to Extract: {fields_to_extract}"

message = [
    {
        "role":"system",
        "content":SYSTEM_PROMPT
    },
    {
        "role":"user",
        "content":user_prompt
    }
]

chat_completion = client.chat.completions.create(messages = message, model="openai/gpt-oss-120b")
response = chat_completion.choices[0].message.content

print(response)


