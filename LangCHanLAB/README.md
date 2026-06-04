# LangChain Normal Objects Lab

## How to Run

1. Install the required packages:

```bash
pip install langchain langchain-openai python-dotenv
```

2. Create a `.env` file in the repository root:

```env
OPENAI_API_KEY=your_api_key_here
```

3. Run the script:

```bash
python normalobjects_langchain.py
```

The script creates themed LangChain tools, builds an agent with those tools, tests three sample complaints, and prints tool usage statistics.

## File Map

- `normalobjects_langchain.py`: Main lab script. Defines the LLM, creative tools, LangChain agent, sample complaints, complaint handler, and tool usage tracker.
- `.env`: Local environment file for `OPENAI_API_KEY`. This file should not be shared or committed.
- `lab_summary.md`: Short narrative summary of the lab.
- `__pycache__/`: Python-generated cache folder created when the script runs.
