# -*- coding: utf-8 -*-
"""Executable Python version of Lab6/exercise9.ipynb."""

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

from datetime import datetime

@tool
def check_calendar_availability(person: str, date: str) -> str:
    """Check sample calendar availability for a person on a date."""
    return f"Sample calendar: {person} is available on {date} between 10:00 and 16:00."

@tool
def get_timezone_offset(city: str) -> str:
    """Return a sample timezone offset for a city."""
    zones = {"Mumbai": "UTC+05:30", "Pune": "UTC+05:30", "London": "UTC+00:00", "New York": "UTC-04:00"}
    return zones.get(city.title(), f"Timezone not found for {city}.")

@tool
def book_meeting(person: str, date: str, time: str) -> str:
    """Book a sample meeting for a person, date, and time."""
    return f"Sample booking confirmed: meeting with {person} on {date} at {time}."

# A complete loop is needed because the model may call multiple tools in successive turns.
tools = [check_calendar_availability, get_timezone_offset, book_meeting]
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

run_query("Is Alex available for a meeting on 2026-11-02?")
run_query("Schedule a sample meeting with Alex in Mumbai on 2026-11-02 at 14:00. Check availability and timezone first.")

