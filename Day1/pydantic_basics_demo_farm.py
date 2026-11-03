# =====================================================================
# Extract fields from a sentence into a validated Pydantic model
# Version: client environment (LangChain AzureChatOpenAI, FARM gateway)
#
# Install once:
#   pip install langchain-core langchain-openai pydantic python-dotenv
#
# Run:
#   python pydantic_basics_demo_farm.py
# =====================================================================

import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI
from pydantic import BaseModel, Field

load_dotenv()

# ---------------------------------------------------------------------
# Connection settings for the FARM gateway, read from a .env file:
#   FARM_BASE_URL, FARM_API_KEY, FARM_DEPLOYMENT_NAME, MODEL_NAME, FARM_API_VERSION
# ---------------------------------------------------------------------
CONFIG = {
    "base_url": os.getenv("FARM_BASE_URL"),
    "api_key": os.getenv("FARM_API_KEY"),
    "chat_deployment": os.getenv("FARM_DEPLOYMENT_NAME"),
    "chat_model": os.getenv("MODEL_NAME"),
    "api_version": os.getenv("FARM_API_VERSION"),
}

missing = [k for k, v in CONFIG.items() if not v]
if missing:
    raise ValueError(f"Missing .env values: {missing}")

# The FARM gateway needs this extra header on every request
FARM_HEADERS = {"genaiplatform-farm-subscription-key": CONFIG["api_key"]}


class PersonDetails(BaseModel):
    name: str = Field(description="The person's full name.")
    age: int = Field(description="The person's age in years.")
    city: str = Field(description="The city where the person lives.")


prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "Extract the person's name, age, and city from the provided text."),
        ("human", "{text}"),
    ]
)

llm = AzureChatOpenAI(
    azure_endpoint=CONFIG["base_url"],
    azure_deployment=CONFIG["chat_deployment"],
    model=CONFIG["chat_model"],
    api_version=CONFIG["api_version"],
    api_key=CONFIG["api_key"],
    default_headers=FARM_HEADERS,
)

# method="function_calling" works on every Azure API version.
# (The default "json_schema" method needs API version 2024-08-01-preview or later.)
extraction_chain = prompt | llm.with_structured_output(PersonDetails, method="function_calling")

person = extraction_chain.invoke({"text": "My name is Maya, I am 30 years old, and I live in Seattle."})
print(person.name, person.age, person.city)
print(person.model_dump_json(indent=2))
