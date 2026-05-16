# Product Listing Generator Refactoring Lab

## How to Run

Create and activate a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Run the refactored script in mock mode:

```bash
python product_listing_generator_refactored.py --products valid.json --output test_generated_listings.json --mock --limit 1
```

Prepare a compatible image-based product dataset:

```bash
python product_listing_generator_refactored.py --prepare-hf --hf-limit 10 --products products.json --images-dir product_images
```

Validate the compatible product dataset:

```bash
python product_listing_generator_refactored.py --products products.json --validate-only
```

Run real OpenAI generation:

```bash
export OPENAI_API_KEY="your_key_here"
python product_listing_generator_refactored.py --products products.json --output generated_listings.json --limit 3
```

Run error-handling checks:

```bash
python product_listing_generator_refactored.py --products missing_products.json --mock
python product_listing_generator_refactored.py --products malformed.json --mock
python product_listing_generator_refactored.py --products invalid_products.json --mock
```

Open the refactoring notebook:

```bash
jupyter notebook refactoring.ipynb
```

## File and Folder Map

- `README.md` - run instructions and file map.
- `lab_summary.md` - short narrative summary for the lab.
- `before_after_comparison.md` - before/after comparison of the refactoring.
- `requirements.txt` - Python package dependencies.
- `product_listing_generator.py` - original product listing generator.
- `product_listing_generator_refactored.py` - modularized/refactored version.
- `refactoring.ipynb` - refactoring checklist, tests, and notebook work.
- `codex_refactor.diff` - diff between the original and refactored Python files.
- `products.json` - simple product fixture using a top-level `products` object.
- `valid.json` - valid image-based fixture for mock testing.
- `invalid_products.json` - valid JSON with product validation errors.
- `malformed.json` - intentionally invalid JSON syntax.
- `invalid.json` - intentionally invalid JSON syntax used in notebook testing.
- `invalid_data.json` - invalid product data used in notebook testing.
- `sample_products.json` - sample image-based product input format.
- `bad_products_for_error_test.json` - broken input for error-handling tests.
- `generated_listings.json` - generated listing output example.
- `mock_generated_listings.json` - mock listing output example.
- `test_generated_listings.json` - local test output.
- `modular_test_output.json` - local modular workflow test output.
- `test_output.json` - local test output.
- `lab_report_template.md` - original lab report template.
- `README_RUN_ME.md` - original run instructions supplied with the lab.
- `product_images/` - local product images used by image-based fixtures.
- `venv/` - local virtual environment.
- `__pycache__/` - Python cache files.
