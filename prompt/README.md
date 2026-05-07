# Prompt Engineering Lab

This folder contains a prompt engineering lab using the OpenAI Python SDK. The notebook tests prompts for sentiment analysis, product description generation, and data extraction across multiple runs, then compares improved prompt versions for consistency.

## How to Run

1. Open `prompt_engineering_lab.ipynb` in Jupyter, VS Code, or another notebook editor.
2. Install the required packages if they are not already installed:

   ```python
   %pip install openai python-dotenv
   ```

3. Add your OpenAI API key when the notebook asks for it. The setup cell uses a hidden prompt:

   ```python
   import getpass
   import os

   if not os.getenv("OPENAI_API_KEY"):
       os.environ["OPENAI_API_KEY"] = getpass.getpass("Paste your OpenAI API key: ")
   ```

   You can also create a local `.env` file with:

   ```bash
   OPENAI_API_KEY=your-api-key-here
   ```

4. Run the notebook cells from top to bottom.
5. Review the output tables and written observations in the notebook.
6. Read `lab_summary.md` for the final concise lab reflection.

## File Map

```text
.
├── README.md
├── prompt_engineering_lab.ipynb
├── lab_summary.md
└──  prompt_engineering_lab.ipynb
```

## Files

`README.md`  
Project instructions and file map.

`prompt_engineering_lab.ipynb`  
Main notebook for the lab. It includes the OpenAI client setup, helper functions for running prompts multiple times, prompt tests, consistency comparisons, failure analysis, and prompt iterations.

`lab_summary.md`  
Short written summary of the lab results and lessons learned.

` prompt_engineering_lab.ipynb`  
Older/duplicate notebook file with a leading space in the filename. Keep only if you still need it for reference.

## Notes

Do not commit or share your OpenAI API key. If you use a `.env` file, keep it local and add it to `.gitignore` before publishing the project.
