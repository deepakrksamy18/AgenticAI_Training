
import os
from dotenv import load_dotenv

from langchain.agents import create_agent
from langchain_openai import ChatOpenAI

# Load environment variables
load_dotenv()

# Bosch LLM configuration
MODEL_NAME = "gpt-5-nano-2025-08-07"

BASE_URL = (
    "https://aoai-farm.bosch-temp.com/api/openai/"
    "deployments/gpt-5-nano-2025-08-07"
)

API_VERSION = "2024-05-01-preview"

# Correct API key retrieval
API_KEY = os.getenv("BOSCH_API_KEY")

if not API_KEY:
    raise ValueError(
        "BOSCH_API_KEY not found. Check your .env file."
    )


# EMI Calculator Tool
def emi_calculator(
    principal: float,
    annual_interest_rate: float,
    duration_months: int
):
    """
    Calculate monthly EMI for a loan.

    Args:
        principal: Loan principal amount.
        annual_interest_rate: Annual interest rate in percentage.
        duration_months: Loan tenure in months.

    Returns:
        Dictionary containing EMI details.
    """

    monthly_rate = annual_interest_rate / 12 / 100

    emi = (
        principal * monthly_rate *
        (1 + monthly_rate) ** duration_months
        /
        ((1 + monthly_rate) ** duration_months - 1)
    )

    return {
        "Principal": principal,
        "Annual Interest Rate (%)": annual_interest_rate,
        "Monthly Interest Rate": round(monthly_rate, 6),
        "Duration (Months)": duration_months,
        "Monthly EMI": round(emi, 2)
    }


# Initialize LLM
llm = ChatOpenAI(
    model=MODEL_NAME,
    base_url=BASE_URL,
    api_key=API_KEY,
    default_query={"api-version": API_VERSION}
)


# Create Agent
emi_agent = create_agent(
    model=llm,
    tools=[emi_calculator],
    system_prompt="""
    You are a helpful financial calculation assistant.

    You have access to an EMI calculator tool.

    Instructions:
    1. Identify the loan principal, interest rate and tenure.
    2. Always use the emi_calculator tool to calculate EMI.
    3. Explain the monthly EMI clearly.
    4. Evaluate affordability based on monthly salary.
    5. Consider the EMI-to-income ratio.
    6. Mention that other expenses and existing loans
       can affect affordability.
    7. Never invent financial calculations.
    """
)


# Invoke Agent
response = emi_agent.invoke(
    {
        "messages": [
            {
                "role": "user",
                "content": """
                I am planning to buy a house costing 50 lakhs.
                I have 10 lakhs available as down payment.
                My monthly salary is 1 lakh.
                The bank offers 12% annual interest.
                Loan duration is 10 years.

                Calculate my EMI and tell me whether
                I can afford this house.
                """
            }
        ]
    }
)


# Print Agent Response
print("\n----- EMI AGENT RESPONSE -----\n")

print(response["messages"][-1].content)
