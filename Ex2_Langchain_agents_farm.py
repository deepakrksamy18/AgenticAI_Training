# =====================================================================
# Ex2 - A LangChain agent with an EMI calculator tool
# Version: client environment (LangChain AzureChatOpenAI, FARM gateway)
#
# Install once:
#   pip install langchain langchain-core langchain-openai python-dotenv
#
# Run:
#   python Ex2_Langchain_agents_farm.py
# =====================================================================

import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import AzureChatOpenAI

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


def emi_calculator(principal: float, annual_interest_rate: float, duration_months: int):
    """
    Calculate EMI and monthly interest rate.

    Args:
        principal (float): Loan amount
        annual_interest_rate (float): Annual interest rate in percentage
        duration_months (int): Loan duration in months

    Returns:
        dict: EMI details
    """
    # Convert annual interest rate to monthly decimal rate
    monthly_rate = annual_interest_rate / 12 / 100

    # EMI Formula
    emi = principal * monthly_rate * (1 + monthly_rate) ** duration_months \
          / ((1 + monthly_rate) ** duration_months - 1)

    return {
        "Principal": principal,
        "Annual Interest Rate (%)": annual_interest_rate,
        "Monthly Interest Rate": round(monthly_rate, 6),
        "Duration (Months)": duration_months,
        "Monthly EMI": round(emi, 2),
    }


# Example usage of the plain function, no AI
result = emi_calculator(500000, 8.5, 60)
print(result)

# 1️⃣ Create the LLM: LangChain's Azure client with the same FARM values as the agno example
llm = AzureChatOpenAI(
    azure_endpoint=CONFIG["base_url"],
    azure_deployment=CONFIG["chat_deployment"],
    model=CONFIG["chat_model"],
    api_version=CONFIG["api_version"],
    api_key=CONFIG["api_key"],
    default_headers=FARM_HEADERS,
)

# 2️⃣ Create Agent
emi_agent = create_agent(
    model=llm,
    tools=[emi_calculator],
    system_prompt="You are a helpful math assistant. The emi_calculator tool helps to compute the emi for a loan "
                  "with principle, interest rate and loan tenure are inputs and returns the monthly EMI applicable.",
)

# 3️⃣ Invoke EMI Agent
response = emi_agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "I am going to get a house which requires 50 lakhs, I have only 10 lakhs in "
                                        "my hand and I have a salary of 1 lakh, can I afford to buy this house and "
                                        "bank is charging 12 percent interest and my loan duration is 10 years?"}
        ]
    }
)
print(response["messages"][-1].content)
