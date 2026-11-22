# -*- coding: utf-8 -*-
"""Executable Python version of Lab6/exercise2.ipynb."""

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
def get_weather(city: str) -> str:
    """Return sample current weather temperature and conditions for a city."""
    sample_weather = {
        "Mumbai": {"temperature": 30, "conditions": "humid and partly cloudy"},
        "Pune": {"temperature": 25, "conditions": "partly cloudy"},
        "Delhi": {"temperature": 22, "conditions": "cool and clear"}
    }
    data = sample_weather.get(city.title())
    if not data:
        return f"Weather data unavailable for {city}. Do not guess the weather."
    return str(data)

@tool
def recommend_outfit(temperature: float, conditions: str) -> str:
    """Recommend an outfit based on temperature in Celsius and weather conditions."""
    if temperature >= 30:
        outfit = "light breathable clothing"
    elif temperature <= 18:
        outfit = "a warm layer or jacket"
    else:
        outfit = "comfortable layered clothing"
    if "rain" in conditions.lower():
        outfit += " and bring an umbrella or raincoat"
    return f"Suggestion: wear {outfit}. Conditions: {conditions}."

# A complete loop is needed because the model may call multiple tools in successive turns.
tools = [get_weather, recommend_outfit]
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

run_query("What should I wear today in Mumbai?")

