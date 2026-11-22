# -*- coding: utf-8 -*-
"""Executable Python version of Lab6/exercise5.ipynb."""

from langchain_groq import ChatGroq
from langchain_core.tools import tool
from dotenv import load_dotenv
import os

load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("Missing GROQ_API_KEY. Add it to the project .env file (see .env.example).")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.environ["GROQ_API_KEY"],
    temperature=0
)

from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

class EMIInput(BaseModel):
    principal: float = Field(gt=0, description="Loan principal amount")
    annual_rate: float = Field(ge=0, le=30, description="Annual interest rate percentage from 0 to 30")
    tenure_months: int = Field(gt=0, description="Loan tenure in months")

def calculate_emi(principal: float, annual_rate: float, tenure_months: int) -> str:
    """Calculate monthly EMI using principal, annual interest rate, and tenure in months."""
    monthly_rate = annual_rate / 100 / 12
    if monthly_rate == 0:
        emi = principal / tenure_months
    else:
        emi = principal * monthly_rate * (1 + monthly_rate) ** tenure_months / ((1 + monthly_rate) ** tenure_months - 1)
    return f"Monthly EMI: ₹{emi:.2f}"

emi_tool = StructuredTool.from_function(
    func=calculate_emi,
    name="calculate_emi",
    description="Calculate monthly loan EMI. Inputs: principal, annual_rate, tenure_months.",
    args_schema=EMIInput
)

# A complete loop is needed because the model may call multiple tools in successive turns.
tools = [emi_tool]
llm_with_tools = llm.bind_tools(tools)
tool_map = {available_tool.name: available_tool for available_tool in tools}

def run_query(user_query: str, max_rounds: int = 8):
    messages = [
        {"role": "system", "content": "Use the available tools as needed. Tools with sample data are only demos; do not present sample results as live information."},
        {"role": "user", "content": user_query},
    ]
    for turn in range(max_rounds):
        response = llm_with_tools.invoke(messages)
        messages.append(response)
        if not response.tool_calls:
            print("\nFinal answer:")
            print(response.content)
            return response.content
        for call in response.tool_calls:
            name = call["name"]
            if name not in tool_map:
                result = f"Unknown tool: {name}"
            else:
                try:
                    result = str(tool_map[name].invoke(call["args"]))
                except Exception as exc:
                    result = f"Tool error ({name}): {exc}"
            print(f"Tool {name} -> {result}")
            messages.append({"role": "tool", "tool_call_id": call["id"], "content": result})
    raise RuntimeError("Model exceeded the tool-call limit. Try a shorter query.")

run_query("What's my EMI for a 5 lakh loan at 9% for 24 months?")

