"""Extract fields from a prompt into a validated Pydantic model."""

import os

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class PersonDetails(BaseModel):
    name: str = Field(description="The person's full name.")
    age: int = Field(description="The person's age in years.")
    city: str = Field(description="The city where the person lives.")


def main() -> None:

    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Extract the person's name, age, and city from the provided text."),
            ("human", "{text}"),
        ]
    )
    llm = ChatOpenAI(
        model=os.getenv("AZURE_OPENAI_DEPLOYMENT", ""),
        base_url=os.getenv(
            "AZURE_OPENAI_BASE_URL",
            "",
        ),
        api_key="",
    )
    extraction_chain = prompt | llm.with_structured_output(PersonDetails)

    person = extraction_chain.invoke(
        {"text": "My name is Maya, I am 30 years old, and I live in Seattle."}
    )
    print(person.name, person.age, person.city)
    print(person.model_dump_json(indent=2))


if __name__ == "__main__":
    main()