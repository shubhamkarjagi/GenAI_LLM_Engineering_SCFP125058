# -*- coding: utf-8 -*-
"""Executable Python version of Lab6/exercise10.ipynb."""

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
def find_restaurants(cuisine: str, city: str) -> str:
    """Find sample restaurants matching a cuisine in a city."""
    data = {
        ("Italian", "Pune"): ["Bella Italia", "Little Italy"],
        ("Indian", "Pune"): ["Shabree", "Durvankur"],
        ("Italian", "Mumbai"): ["Example Italian Bistro"]
    }
    return str(data.get((cuisine.title(), city.title()), []))

@tool
def get_restaurant_rating(restaurant_name: str) -> str:
    """Return a sample rating out of 5 for a restaurant."""
    ratings = {"Bella Italia": 4.3, "Little Italy": 4.1, "Shabree": 4.4, "Durvankur": 4.2, "Example Italian Bistro": 4.0}
    rating = ratings.get(restaurant_name)
    return f"Sample rating: {rating}/5" if rating is not None else "Rating unavailable in sample data."

@tool
def check_restaurant_open(restaurant_name: str, day: str) -> str:
    """Check sample opening status for a restaurant on a day."""
    return f"Sample data: {restaurant_name} is listed as open on {day}."

@tool
def estimate_cost_for_two(restaurant_name: str) -> str:
    """Return a sample approximate cost for two people in INR."""
    prices = {"Bella Italia": 1800, "Little Italy": 1600, "Shabree": 900, "Durvankur": 700, "Example Italian Bistro": 2000}
    price = prices.get(restaurant_name)
    return f"Sample cost for two: ₹{price}" if price is not None else "Cost estimate unavailable in sample data."

# A complete loop is needed because the model may call multiple tools in successive turns.
tools = [find_restaurants, get_restaurant_rating, check_restaurant_open, estimate_cost_for_two]
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

run_query("Suggest a good Italian restaurant in Pune that's open today, with its rating and rough cost for two.")

