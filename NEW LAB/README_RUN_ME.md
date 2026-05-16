# ChatGPT API Vision Lab — Run Instructions

This kit gives you a submit-ready structure for the lab.

## Files

- `product_listing_generator.py` — main Python script
- `requirements.txt` — packages to install
- `sample_products.json` — example metadata format
- `bad_products_for_error_test.json` — intentionally broken product file for error-handling screenshots
- `lab_report_template.md` — 1-page report template
- `mock_generated_listings.json` — mock output example, clearly not real API output

## 1. Create a project folder

Put these files in one folder.

## 2. Create and activate a virtual environment

Mac/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

Windows PowerShell:

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

## 3. Install packages

```bash
pip install -r requirements.txt
```

## 4. Set your OpenAI API key

Do NOT put your key inside the Python file.

Mac/Linux:

```bash
export OPENAI_API_KEY="your_key_here"
```

Windows PowerShell:

```powershell
$env:OPENAI_API_KEY="your_key_here"
```

Alternative: create a local `.env` file:

```bash
OPENAI_API_KEY="your_key_here"
```

Do not submit your `.env` file.

## 5. Prepare product data

Recommended lab path using HuggingFace:

```bash
python product_listing_generator.py --prepare-hf --hf-limit 10 --products products.json --images-dir product_images
```

This creates:

- `products.json`
- a `product_images/` folder

## 6. Validate setup before spending API credits

```bash
python product_listing_generator.py --products products.json --validate-only
```

You want to see:

- product file loaded
- first image encoded successfully
- API key is set

## 7. Generate real listings for at least 3 products

```bash
python product_listing_generator.py --products products.json --output generated_listings.json --limit 3
```

That creates `generated_listings.json`.

## 8. Run mock mode for testing without API credits

```bash
python product_listing_generator.py --products sample_products.json --output mock_generated_listings.json --mock --limit 3
```

Mock mode is only for testing. For the final lab, submit real API-generated results.

## 9. Error-handling screenshot

Run the intentionally broken test:

```bash
python product_listing_generator.py --products bad_products_for_error_test.json --output error_test_results.json --mock
```

This demonstrates image/file error handling. For a real API error screenshot, you can temporarily unset your key and run without `--mock`.

Mac/Linux:

```bash
unset OPENAI_API_KEY
python product_listing_generator.py --products products.json --output api_key_error.json --limit 1
```

Then set your key again before real work.

## Suggested screenshots for submission

Capture terminal output showing:

1. Dataset preparation success
2. Validation success
3. Successful generation of at least 3 listings
4. One error-handling example

## Important submission warning

Do not submit:

- `.env`
- API keys
- screenshots where your API key is visible
