# -*- coding: utf-8 -*-
"""Executable Python version of Lab6/exercise8.ipynb."""

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
def get_flight_cost(origin: str, destination: str) -> str:
    """Return a sample estimated round-trip flight cost for a route in INR."""
    sample = {("Delhi", "Goa"): 8500, ("Mumbai", "Goa"): 4500, ("Pune", "Delhi"): 7000}
    cost = sample.get((origin.title(), destination.title()))
    if cost is None:
        return f"Sample flight estimate unavailable for {origin} to {destination}."
    return f"Estimated round-trip flight cost: ₹{cost}."

@tool
def get_hotel_cost(city: str, nights: int) -> str:
    """Return a sample hotel cost estimate in INR for the requested number of nights."""
    if nights <= 0:
        return "Nights must be positive."
    nightly_rate = {"Goa": 3000, "Delhi": 3500, "Mumbai": 4000}.get(city.title())
    if nightly_rate is None:
        return f"Sample hotel estimate unavailable for {city}."
    return f"Estimated hotel cost for {nights} night(s): ₹{nightly_rate * nights}."

@tool
def get_food_budget(city: str, days: int) -> str:
    """Return a sample food budget estimate in INR for the requested number of days."""
    if days <= 0:
        return "Days must be positive."
    daily_budget = {"Goa": 1000, "Delhi": 1200, "Mumbai": 1300}.get(city.title())
    if daily_budget is None:
        return f"Sample food estimate unavailable for {city}."
    return f"Estimated food budget for {days} day(s): ₹{daily_budget * days}."

# A complete loop is needed because the model may call multiple tools in successive turns.
tools = [get_flight_cost, get_hotel_cost, get_food_budget]
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

run_query("I'm planning a 3-night trip from Delhi to Goa. What will flights, hotel, and food roughly cost?")

