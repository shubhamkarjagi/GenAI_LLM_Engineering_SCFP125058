# -*- coding: utf-8 -*-
"""Executable Python version of Lab5/exercise1.ipynb."""

import os
from dotenv import load_dotenv
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser

load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("Missing GROQ_API_KEY. Add it to the project .env file (see .env.example).")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    temperature=0.0,
    api_key=os.environ.get("GROQ_API_KEY")
)


class ReviewData(BaseModel):
    complaint: str = Field(description="The primary issue or core complaint reported by the customer")
    product_or_feature: str = Field(description="The specific product, component, or feature mentioned")
    sentiment: str = Field(description="Overall sentiment of the review (e.g., Negative, Strongly Negative)")

step1_parser = PydanticOutputParser(pydantic_object=ReviewData)

step1_template = PromptTemplate(
    template="Extract the requested details from the customer review.\n{format_instructions}\n\nReview:\n{review_text}\n",
    input_variables=["review_text"],
    partial_variables={"format_instructions": step1_parser.get_format_instructions()}
)

step1_chain = step1_template | llm | step1_parser

step2_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an internal support ticket writer. Create a clear, concise, and actionable support ticket summary for the engineering/support team based only on the provided structured complaint data."),
    ("human", "Core Complaint: {complaint}\nProduct/Feature: {product_or_feature}\nSentiment: {sentiment}\n\nWrite the support ticket summary:")
])

step2_chain = step2_prompt | llm | StrOutputParser()

review_text = input("Enter customer review: ")

# Step 1 execution
structured_data = step1_chain.invoke({"review_text": review_text})
print("\n--- Step 1: Structured Complaint Data ---")
print(structured_data.model_dump_json(indent=2))

# Step 2 execution 
ticket_summary = step2_chain.invoke({
    "complaint": structured_data.complaint,
    "product_or_feature": structured_data.product_or_feature,
    "sentiment": structured_data.sentiment
})
print("\n--- Step 2: Support Ticket Summary ---")
print(ticket_summary)


