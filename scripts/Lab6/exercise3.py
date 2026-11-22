# -*- coding: utf-8 -*-
"""Executable Python version of Lab6/exercise3.ipynb."""

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

@tool
def convert_temperature(value: float, from_unit: str, to_unit: str) -> str:
    """Convert temperature between Celsius and Fahrenheit. Use units C or F."""
    from_unit = from_unit.upper()
    to_unit = to_unit.upper()
    if from_unit == "C" and to_unit == "F":
        result = value * 9 / 5 + 32
    elif from_unit == "F" and to_unit == "C":
        result = (value - 32) * 5 / 9
    else:
        return "Unsupported temperature conversion. Use C to F or F to C."
    return f"{value}°{from_unit} = {result:.2f}°{to_unit}"

@tool
def convert_distance(value: float, from_unit: str, to_unit: str) -> str:
    """Convert distance between kilometers and miles. Use units km or miles."""
    from_unit = from_unit.lower()
    to_unit = to_unit.lower()
    if from_unit == "km" and to_unit == "miles":
        result = value * 0.621371
    elif from_unit == "miles" and to_unit == "km":
        result = value / 0.621371
    else:
        return "Unsupported distance conversion. Use km to miles or miles to km."
    return f"{value} {from_unit} = {result:.2f} {to_unit}"

# A complete loop is needed because the model may call multiple tools in successive turns.
tools = [convert_temperature, convert_distance]
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

run_query("Convert 100 km to miles.")

