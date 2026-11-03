import ast
import math
import operator
import os

from dotenv import load_dotenv
from langchain.agents import create_agent
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import InMemorySaver

load_dotenv()

MODEL_NAME = "gpt-5-nano-2025-08-07"
BASE_URL = "https://aoai-farm.bosch-temp.com/api/openai/deployments/gpt-5-nano-2025-08-07"
API_VERSION = "2024-05-01-preview"
API_KEY = os.getenv("MODEL_FARM_API_KEY") or os.getenv("BOSCH_API_KEY")

if not API_KEY:
    raise ValueError("MODEL_FARM_API_KEY (or BOSCH_API_KEY) not found. Check your .env file.")

_BINARY_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}

_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}

_FUNCTIONS = {
    "abs": abs, "round": round, "min": min, "max": max, "pow": pow,
    "sqrt": math.sqrt, "cbrt": math.cbrt, "exp": math.exp,
    "log": math.log, "log10": math.log10, "log2": math.log2,
    "sin": math.sin, "cos": math.cos, "tan": math.tan,
    "asin": math.asin, "acos": math.acos, "atan": math.atan, "atan2": math.atan2,
    "sinh": math.sinh, "cosh": math.cosh, "tanh": math.tanh,
    "degrees": math.degrees, "radians": math.radians,
    "floor": math.floor, "ceil": math.ceil, "trunc": math.trunc,
    "factorial": math.factorial, "gcd": math.gcd, "lcm": math.lcm,
    "comb": math.comb, "perm": math.perm, "hypot": math.hypot,
}

_CONSTANTS = {"pi": math.pi, "e": math.e, "tau": math.tau, "inf": math.inf}

# Guards against expressions that would hang the process.
_MAX_EXPONENT = 10_000
_MAX_FACTORIAL_ARG = 5_000


def _eval_node(node):
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.Name) and node.id in _CONSTANTS:
        return _CONSTANTS[node.id]
    if isinstance(node, ast.BinOp) and type(node.op) in _BINARY_OPS:
        left, right = _eval_node(node.left), _eval_node(node.right)
        if isinstance(node.op, ast.Pow) and abs(right) > _MAX_EXPONENT:
            raise ValueError(f"Exponent too large (max {_MAX_EXPONENT}).")
        return _BINARY_OPS[type(node.op)](left, right)
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_eval_node(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and not node.keywords:
        name = node.func.id
        if name not in _FUNCTIONS:
            raise ValueError(f"Unsupported function: {name}")
        args = [_eval_node(arg) for arg in node.args]
        if name in ("factorial", "comb", "perm") and any(abs(a) > _MAX_FACTORIAL_ARG for a in args):
            raise ValueError(f"Argument too large for {name} (max {_MAX_FACTORIAL_ARG}).")
        return _FUNCTIONS[name](*args)
    raise ValueError(f"Unsupported expression element: {type(node).__name__}")


def calculate(expression: str) -> str:
    """
    Evaluate a mathematical expression and return the result.

    Supports + - * / // % ** (and ^ for power), parentheses, constants (pi, e, tau),
    and functions: abs, round, min, max, pow, sqrt, cbrt, exp, log (natural or log(x, base)),
    log10, log2, sin, cos, tan, asin, acos, atan, atan2, sinh, cosh, tanh (trig uses radians;
    use radians()/degrees() to convert), floor, ceil, trunc, factorial, gcd, lcm, comb, perm, hypot.

    Args:
        expression: Expression string, e.g. "sqrt(16) + 2**3 * sin(pi/2)".
    """
    try:
        tree = ast.parse(expression.replace("^", "**").strip(), mode="eval")
        return str(_eval_node(tree.body))
    except ZeroDivisionError:
        return "Error: division by zero."
    except Exception as exc:
        return f"Error: {exc}"


def emi_calculator(principal: float, annual_interest_rate: float, duration_months: int) -> dict:
    """
    Calculate the monthly EMI for a loan.

    Args:
        principal: Loan amount.
        annual_interest_rate: Annual interest rate in percentage.
        duration_months: Loan duration in months.
    """
    try:
        monthly_rate = annual_interest_rate / 12 / 100
        if monthly_rate == 0:
            emi = principal / duration_months
        else:
            factor = (1 + monthly_rate) ** duration_months
            emi = principal * monthly_rate * factor / (factor - 1)
        return {
            "Principal": principal,
            "Annual Interest Rate (%)": annual_interest_rate,
            "Monthly Interest Rate": round(monthly_rate, 6),
            "Duration (Months)": duration_months,
            "Monthly EMI": round(emi, 2),
        }
    except Exception as exc:
        return {"error": str(exc)}


llm = ChatOpenAI(
    model=MODEL_NAME,
    base_url=BASE_URL,
    api_key=API_KEY,
    default_query={"api-version": API_VERSION},
)

calculator_agent = create_agent(
    model=llm,
    tools=[calculate, emi_calculator],
    system_prompt=(
        "You are a helpful math assistant. Always use the calculate tool for any arithmetic "
        "or scientific computation instead of computing in your head. Use emi_calculator for "
        "loan EMI questions. If a tool returns an error, explain it and suggest a correction."
    ),
    checkpointer=InMemorySaver(),
)


def _print_step(message) -> None:
    kind = getattr(message, "type", "")
    if kind == "ai" and getattr(message, "tool_calls", None):
        for call in message.tool_calls:
            print(f"  [Step] Calling tool '{call['name']}' with {call['args']}")
    elif kind == "tool":
        print(f"  [Step] Tool result: {message.content}")
    elif kind == "ai" and message.content:
        print(f"\nAgent: {message.content}\n")


def main() -> None:
    config = {"configurable": {"thread_id": "calculator-chat"}}
    print("Calculator Agent ready. Type 'exit' to quit.\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not user_input:
            continue
        if user_input.lower() in {"exit", "quit"}:
            break

        try:
            for update in calculator_agent.stream(
                {"messages": [{"role": "user", "content": user_input}]},
                config,
                stream_mode="updates",
            ):
                for node_output in update.values():
                    for message in node_output.get("messages", []):
                        _print_step(message)
        except Exception as exc:
            print(f"Error: {exc}\n")


if __name__ == "__main__":
    main()
