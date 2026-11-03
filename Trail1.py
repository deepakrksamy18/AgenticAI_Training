import os
from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
 
load_dotenv()
 
# Same endpoint, key and api-version as check.py
MODEL_NAME = "gpt-5-nano-2025-08-07"
BASE_URL = "https://aoai-farm.bosch-temp.com/api/openai/deployments/gpt-5-nano-2025-08-07"
API_VERSION = "2024-05-01-preview"
API_KEY = os.getenv("MODEL_FARM_API_KEY")
 
 
 
 
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
        "Monthly EMI": round(emi, 2)
    }
 
# Example usage
result = emi_calculator(500000, 8.5, 60)
print(result)
llm = ChatOpenAI(
    model=MODEL_NAME,
    base_url=BASE_URL,
    api_key=API_KEY,
    default_query={"api-version": API_VERSION},
)
 
# 2️⃣ Create Agent
emi_agent = create_agent(
    model=llm,
    tools=[emi_calculator],
    system_prompt="You are a helpful math assistant. The emi_calculator tool helps to compute the emi for a loan with principle, interest rate and loan tenure are inputs and returns the monthly EMI applicable."
)
 
# 3️⃣ Invoke EMI Agent
response = emi_agent.invoke(
    {
        "messages": [
            {"role": "user", "content": "I am going to get a house which requires 50 lakhs, I have only 10 lakhs in my hand and I have a salary of 1 lakh, can I afford to buy this house and bank is charging 12 percent interest and my laon duration is 10 years?"}
        ]
    }
)
print(response["messages"][-1].content)