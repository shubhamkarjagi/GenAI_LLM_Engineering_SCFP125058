# -*- coding: utf-8 -*-
"""Executable Python version of Lab3/exercise2.ipynb."""

from groq import Groq
from dotenv import load_dotenv
import os
load_dotenv()


GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise RuntimeError("Missing GROQ_API_KEY. Add it to the project .env file (see .env.example).")
client = Groq(api_key= GROQ_API_KEY)

system_prompt = (
        "You are an e-commerce review classifier. "
        "Classify the given review into exactly one of these categories: "
        "Product Quality, Shipping & Delivery, Customer Support, or Pricing. "
        "Output ONLY the category name, nothing else."
    )

user_prompt = input("Give your reviews: ")

message = [
    {
        "role":"system",
        "content":system_prompt
    },
    {
        "role": "user",
        "content":user_prompt
    }
]
chat_completion = client.chat.completions.create(messages = message, model="openai/gpt-oss-120b")
response = chat_completion.choices[0].message.content


print(response)


