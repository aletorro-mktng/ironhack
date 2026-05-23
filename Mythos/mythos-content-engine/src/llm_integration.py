import os
from dotenv import load_dotenv
from openai import OpenAI


load_dotenv()


DEFAULT_MODEL = os.getenv("LLM_MODEL", "gpt-5.4-mini")

def generate_text(prompt: str, model: str = DEFAULT_MODEL) -> str:
    """
    Send a prompt to the selected LLM provider and return generated text.
    """

    if not os.getenv("OPENAI_API_KEY"):
        raise EnvironmentError(
            "OPENAI_API_KEY was not found. Add it to your .env file."
        )

    client = OpenAI()

    response = client.responses.create(
        model=model,
        input=prompt
    )

    return response.output_text


if __name__ == "__main__":
    test_prompt = "Write one sentence confirming the Mythos Content Engine is working."

    result = generate_text(test_prompt)

    print("LLM test successful.")
    print(result)
