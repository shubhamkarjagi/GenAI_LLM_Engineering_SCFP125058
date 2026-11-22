# -*- coding: utf-8 -*-
"""Executable Python version of Lab3/exercise4.ipynb."""

from groq import Groq
from dotenv import load_dotenv
import os
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY. Add it to the project .env file (see .env.example).")
client = Groq(api_key=GROQ_API_KEY)

customer_message = input("Enter the message: ")
company_name = input("Enter the company name: ")
max_words = input("Enter the maximum number of words for the reply: ")
user_prompt = "Customer Message: {customer_msg}\nCompany Name: {company_n}\nWord Limit: {max_w}"

SYSTEM_PROMPT = """You are a professional customer support assistant. Generate a helpful, clear, and empathetic reply to the customer's message."""


def ask_llm(SYSTEM_PROMPT, user_prompt):
    message = [
        {
            "role":"system",
            "content": SYSTEM_PROMPT
        },
        {
            "role":"user",
            "content": user_prompt
        }
    ]
    chat_completion = client.chat.completions.create(messages=message, model="openai/gpt-oss-120b")
    response = chat_completion.choices[0].message.content
    return response

print(ask_llm(SYSTEM_PROMPT, user_prompt.format(customer_msg=customer_message, company_n=company_name, max_w=max_words)))


