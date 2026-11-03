"""Module 2, Lab 4. Reflexion: custom code generator agent.

Pattern: an actor attempts the task, an evaluator scores it, a reflector writes a short lesson
about why it failed, the lesson goes into memory, and the actor tries again with the lessons.
The model's weights do not change; the improvement comes from text the agent wrote for itself.

Points for an experienced audience
  1. The evaluator must be grounded. Here it runs real tests. A model asked whether its own
     code looks right mostly agrees with itself.
  2. Memory holds lessons, not transcripts.
  3. It is bounded: MAX_ATTEMPTS and a "not solved" exit are part of the pattern.

The specification given to the actor is deliberately short. The team's conventions live only
in TEST_CASES, so the first attempt usually fails and the agent learns them from the failures.

SAFETY: this lab runs model-written code in a subprocess with a timeout. That is acceptable for
a pure function in a classroom. In production, run generated code in a container or sandbox
with no network and no credentials.

Variations to try
  - Put every convention into CODE_TASK: it usually passes first time. Reflection is the
    expensive substitute for a missing specification.
  - Give the coder the raw failures and skip the reflector: plain retry, the control experiment.
  - Set MAX_ATTEMPTS = 1 for the baseline pass rate.
"""
import json
import operator
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Annotated

from pydantic import BaseModel, Field
from typing_extensions import TypedDict

from langchain.agents import create_agent
from langchain.agents.structured_output import ToolStrategy
from langgraph.graph import END, START, StateGraph

from lab_common import banner, clip, llm


CODE_TASK = "Write a Python function parse_duration(text) that converts a duration string such as '1h30m' to a whole number of seconds."

# Each case is (expression, expected value). An expected value that is a string names the exception type.
TEST_CASES = [
    ("parse_duration('45s')", 45),
    ("parse_duration('1h30m')", 5400),
    ("parse_duration('2d')", 172800),
    ("parse_duration('90')", 90),                 # a bare number means seconds
    ("parse_duration(' 1H 30M ')", 5400),         # case and whitespace are ignored
    ("parse_duration('1.5h')", 5400),             # decimals are allowed
    ("parse_duration('0.5s')", 1),                # fractions of a second round half up
    ("parse_duration('1w')", 604800),             # weeks are supported
    ("parse_duration('')", "ValueError"),
    ("parse_duration('10x')", "ValueError"),
    ("parse_duration('30m1h')", "ValueError"),    # units must go from largest to smallest
    ("parse_duration('1h1h')", "ValueError"),     # a unit may appear once
    ("parse_duration('-5m')", "ValueError"),
    ("parse_duration(None)", "TypeError"),        # a non-string is a type error
]

TEST_RUNNER = '''
import json
failures = []
for expr, expected in CASES:
    try:
        got = eval(expr)
    except Exception as exc:
        got = type(exc).__name__
    if got != expected or type(got) is not type(expected):
        failures.append(f"{expr}: expected {expected!r}, got {got!r}")
print(json.dumps(failures))
'''


def run_tests(code: str, cases=TEST_CASES, timeout=10):
    """Run candidate code against the cases in a separate process. Returns (passed, list of failures)."""
    with tempfile.TemporaryDirectory() as tmp:
        script = Path(tmp) / "candidate_test.py"
        script.write_text(f"{code}\n\nCASES = {cases!r}\n{TEST_RUNNER}")
        try:
            proc = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, timeout=timeout, cwd=tmp)
        except subprocess.TimeoutExpired:
            return False, [f"timed out after {timeout}s (possible infinite loop)"]
    if proc.returncode != 0:
        return False, ["code did not run: " + (proc.stderr.strip().splitlines() or ["unknown error"])[-1]]
    failures = json.loads(proc.stdout.strip().splitlines()[-1])
    return not failures, failures


CODER_PROMPT = """You are a Python developer writing one small, pure function to a specification.

- Return the complete source of the function in `code`: plain Python, standard library only, no markdown
  fences, no tests, no printing, no input and no file or network access.
- If you are given lessons from earlier attempts, they describe how this team's test suite expects the
  function to behave. Treat every lesson as part of the specification, and keep satisfying the earlier lessons
  when you apply a new one.
- If you are given a previous attempt, revise it. Do not start again from nothing unless a lesson says the
  approach itself is wrong.
- In `notes`, say in one or two sentences what you changed and why.
"""

REFLECTOR_PROMPT = """You review a failed attempt at a small Python function. You are given the task, the code
and the test cases that failed. Write what the next attempt needs to know.

- Work out the rule behind the failures. Several failing cases often share one cause, so group them.
- Write each lesson as a requirement about behaviour that would still be true if the code were rewritten from
  scratch, for example "a bare number with no unit is a count of seconds". Do not write code and do not
  describe a line-by-line patch.
- Include only what the failures support. Do not guess at requirements that no failing case shows.
- If a lesson from an earlier attempt was not followed, say so plainly.
"""


class CodeSolution(BaseModel):
    code: str = Field(description="Complete Python source, no markdown fences")
    notes: str


class Reflection(BaseModel):
    diagnosis: str = Field(description="Why the attempt failed, in one or two sentences")
    lessons: list[str] = Field(description="Behavioural requirements learned from the failures")


coder = create_agent(model=llm, tools=[], system_prompt=CODER_PROMPT, response_format=ToolStrategy(CodeSolution))
reflector = create_agent(model=llm, tools=[], system_prompt=REFLECTOR_PROMPT, response_format=ToolStrategy(Reflection))


class ReflexionState(TypedDict, total=False):
    task: str
    code: str
    attempt: int
    passed: bool
    failures: list[str]
    lessons: Annotated[list[str], operator.add]   # episodic memory: grows across attempts
    history: Annotated[list[dict], operator.add]


MAX_ATTEMPTS = 4


def generate_node(state: ReflexionState):
    parts = [f"Task:\n{state['task']}"]
    if state.get("lessons"):
        parts.append("Lessons from earlier attempts:\n" + "\n".join(f"- {l}" for l in state["lessons"]))
    if state.get("code"):
        parts.append(f"Previous attempt:\n{state['code']}")
    solution = coder.invoke({"messages": [{"role": "user", "content": "\n\n".join(parts)}]})["structured_response"]
    attempt = state.get("attempt", 0) + 1
    print(f"\nATTEMPT {attempt}: {clip(solution.notes, 200)}")
    return {"code": solution.code, "attempt": attempt}


def evaluate_node(state: ReflexionState):
    passed, failures = run_tests(state["code"])
    print(f"  tests: {'all passed' if passed else str(len(failures)) + ' failing'}")
    for failure in failures[:6]:
        print(f"    - {failure}")
    return {"passed": passed, "failures": failures,
            "history": [{"attempt": state["attempt"], "failing": len(failures)}]}


def reflect_node(state: ReflexionState):
    brief = (f"Task:\n{state['task']}\n\nCode:\n{state['code']}\n\nFailing cases:\n" + "\n".join(state["failures"])
             + "\n\nLessons already recorded:\n" + ("\n".join(state.get("lessons", [])) or "none"))
    reflection = reflector.invoke({"messages": [{"role": "user", "content": brief}]})["structured_response"]
    print(f"  reflection: {clip(reflection.diagnosis, 200)}")
    for lesson in reflection.lessons:
        print(f"    lesson: {lesson}")
    return {"lessons": reflection.lessons}


def after_evaluation(state: ReflexionState):
    return END if state["passed"] or state["attempt"] >= MAX_ATTEMPTS else "reflect"


builder = StateGraph(ReflexionState)
builder.add_node("generate", generate_node)
builder.add_node("evaluate", evaluate_node)
builder.add_node("reflect", reflect_node)
builder.add_edge(START, "generate")
builder.add_edge("generate", "evaluate")
builder.add_conditional_edges("evaluate", after_evaluation, [END, "reflect"])
builder.add_edge("reflect", "generate")
code_generator = builder.compile()


def main():
    banner("Reflexion loop")
    result = code_generator.invoke({"task": CODE_TASK})
    print("\n=== Outcome ===")
    print("Solved" if result["passed"] else f"Not solved after {result['attempt']} attempts", "| failing tests per attempt:",
          [h["failing"] for h in result["history"]])
    print("\nLessons in memory:")
    for lesson in result.get("lessons", []):
        print(" -", lesson)
    print("\nFinal code:\n" + result["code"])


if __name__ == "__main__":
    main()
