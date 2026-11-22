# -*- coding: utf-8 -*-
"""Executable Python version of Lab5/exercise4.ipynb."""


import os
import requests
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_core.prompts import PromptTemplate, ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from pydantic import BaseModel, Field

load_dotenv()
if not os.getenv("GROQ_API_KEY"):
    raise RuntimeError("Missing GROQ_API_KEY. Add it to the project .env file (see .env.example).")

llm = ChatGroq(
    model="openai/gpt-oss-20b",
    api_key=os.getenv("GROQ_API_KEY")
)

# STEP 1: Extract event details
class EventDetails(BaseModel):
    event_type: str = Field(description="Type of event")
    event_date: str = Field(description="Date in YYYY-MM-DD format")
    location: str = Field(description="City name")
    indoor_outdoor_preference: str = Field(description="Indoor or outdoor")


parser = PydanticOutputParser(pydantic_object=EventDetails)

prompt = PromptTemplate(
    template="""
Extract the event details from the description.
Convert dates to YYYY-MM-DD.
If the year is missing, use 2026.
Extract the city name.
{format_instructions}

Description: {event_description_text}
""",
    input_variables=["event_description_text"],
    partial_variables={
        "format_instructions": parser.get_format_instructions()
    }
)

step1_chain = prompt | llm | parser


# STEP 2: Get weather information
class WeatherData(BaseModel):
    temperature: str
    precipitation_chance: str
    conditions: str


def get_weather(location, event_date):
    api_key = os.getenv("WEATHER_API_KEY")

    if not api_key:
        raise ValueError("Add WEATHER_API_KEY to your .env file")

    url = "https://api.weatherapi.com/v1/forecast.json"
    params = {
        "key": api_key,
        "q": location,
        "dt": event_date
    }

    response = requests.get(url, params=params, timeout=20)
    data = response.json()

    if response.status_code != 200 or "forecast" not in data:
        message = data.get("error", {}).get("message", "Unknown API error")
        raise ValueError("Weather API error: " + message)

    day = data["forecast"]["forecastday"][0]["day"]

    return WeatherData(
        temperature=str(day["avgtemp_c"]) + " °C",
        precipitation_chance=str(day["daily_chance_of_rain"]) + "%",
        conditions=day["condition"]["text"]
    )


# STEP 3: Generate recommendation
step3_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful event planning assistant."),
    ("human", """
Event: {event_type}
Date: {event_date}
Location: {location}
Preference: {preference}

Temperature: {temperature}
Rain chance: {rain}
Conditions: {conditions}

Recommend whether to proceed, reschedule, or move indoors.
Give a short reason based on the weather.
""")
])

step3_chain = step3_prompt | llm | StrOutputParser()


# RUN ALL THREE STEPS
try:
    description = input("Describe your event: ")

    event = step1_chain.invoke({
        "event_description_text": description
    })

    print("\nSTEP 1: Event Details")
    print(event)

    if (event.location.lower() == "not specified"
            or event.event_date.lower() == "not specified"):
        raise ValueError("Please enter a valid city and event date.")

    weather = get_weather(event.location, event.event_date)

    print("\nSTEP 2: Weather Data")
    print(weather)

    result = step3_chain.invoke({
        "event_type": event.event_type,
        "event_date": event.event_date,
        "location": event.location,
        "preference": event.indoor_outdoor_preference,
        "temperature": weather.temperature,
        "rain": weather.precipitation_chance,
        "conditions": weather.conditions
    })

    print("\nSTEP 3: Recommendation")
    print(result)

except Exception as e:
    print("\nError:", e)



