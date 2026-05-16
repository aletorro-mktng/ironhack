#!/usr/bin/env python3
"""
LAB | API Calling to ChatGPT
Automated Product Listing Generator

What this script does:
1. Loads product metadata from JSON or CSV.
2. Encodes product images as base64 data URLs.
3. Sends the product image + metadata to OpenAI's vision-capable API.
4. Receives a structured product listing.
5. Saves all generated listings to JSON.
6. Handles common API and file errors gracefully.

Security:
- DO NOT paste your API key into this file.
- Use an environment variable called OPENAI_API_KEY or a local .env file.
"""

from __future__ import annotations

import argparse
import base64
import csv
import json
import mimetypes
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    # python-dotenv is helpful but not required.
    pass

from pydantic import BaseModel, Field, ValidationError


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
DEFAULT_PRICE = 29.99
DEFAULT_DELAY_SECONDS = 1.0
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_BASE_DELAY = 2.0
DEFAULT_TEMPERATURE = 0.7
DEFAULT_HF_DATASET = "ashraq/fashion-product-images-small"
DEFAULT_HF_PRICE_MIN = 19.99
DEFAULT_HF_PRICE_MAX = 149.99
SUPPORTED_IMAGE_TYPES = {"image/jpeg", "image/png", "image/webp", "image/gif"}
PRODUCT_NAME_FIELDS = ("name", "productDisplayName", "product_name", "title")
CATEGORY_FIELDS = ("category", "masterCategory", "subCategory", "articleType")
IMAGE_PATH_FIELDS = ("image_path", "image", "path")
PRODUCT_ID_FIELDS = ("id", "product_id", "sku")
ADDITIONAL_INFO_FIELDS = ("gender", "baseColour", "season", "usage", "articleType", "additional_info")
SYSTEM_PROMPT = (
    "You generate accurate, ethical, conversion-focused e-commerce listings. "
    "You must not invent product specs that are not visible or provided."
)


class ProductInput(BaseModel):
    """Validated product data used by the listing workflow."""
    id: Any
    name: str = Field(min_length=1)
    price: float = Field(gt=0)
    category: str = Field(min_length=1)
    image_path: str = Field(min_length=1)
    additional_info: Optional[str] = None


class ProductListing(BaseModel):
    """Structured output expected from the OpenAI API."""
    title: str = Field(description="Catchy SEO-friendly product title, max 60 characters.")
    description: str = Field(description="Detailed product description, roughly 150-200 words.")
    features: List[str] = Field(description="5 to 7 product feature bullets.")
    keywords: str = Field(description="10 to 15 comma-separated SEO keywords.")


def format_error_message(
    function_name: str,
    error_type: str,
    context: str,
    error_message: str,
    helpful_tip: str,
) -> str:
    """Create a consistent error message for debugging."""
    return (
        f"ERROR in {function_name}(): {error_type}\n"
        f"  Location: {context}\n"
        f"  Message: {error_message}\n"
        f"  Suggestion: {helpful_tip}"
    )


def get_first_value(raw: Dict[str, Any], fields: tuple[str, ...], default: Any = None) -> Any:
    """Return the first non-empty value from a list of possible field names."""
    for field in fields:
        value = raw.get(field)
        if value not in [None, "", "nan"]:
            return value
    return default


def collect_additional_info(raw: Dict[str, Any]) -> Optional[str]:
    """Format optional product metadata into one readable string."""
    additional_parts = []

    for field in ADDITIONAL_INFO_FIELDS:
        value = raw.get(field)
        if value not in [None, "", "nan"]:
            additional_parts.append(f"{field}: {value}")

    return "; ".join(additional_parts) or None


def validate_product_data(product: Dict[str, Any]) -> Dict[str, Any]:
    """Validate normalized product data with Pydantic."""
    try:
        return ProductInput(**product).model_dump()
    except ValidationError as error:
        print(format_error_message(
            function_name="validate_product_data",
            error_type="ValidationError",
            context=f"Product data: {product}",
            error_message=str(error),
            helpful_tip="Check required fields: name, positive price, category, and image_path.",
        ))
        raise


def validate_image_file(image_path: str | Path) -> str:
    """
    Validate an image file and return its MIME type.

    Raises FileNotFoundError or ValueError when the image cannot be used.
    """
    path = Path(image_path)

    if not path.exists():
        print(format_error_message(
            function_name="validate_image_file",
            error_type="FileNotFoundError",
            context=f"Image file '{path}'",
            error_message="Image file was not found.",
            helpful_tip="Check that image_path points to an existing JPG, PNG, WEBP, or GIF file.",
        ))
        raise FileNotFoundError(f"Image file not found: {path}")

    if not path.is_file():
        print(format_error_message(
            function_name="validate_image_file",
            error_type="ValueError",
            context=f"Image path '{path}'",
            error_message="Image path is not a file.",
            helpful_tip="Use a direct file path, not a folder path.",
        ))
        raise ValueError(f"Image path is not a file: {path}")

    if path.stat().st_size == 0:
        print(format_error_message(
            function_name="validate_image_file",
            error_type="ValueError",
            context=f"Image file '{path}'",
            error_message="Image file is empty.",
            helpful_tip="Replace the image with a valid non-empty image file.",
        ))
        raise ValueError(f"Image file is empty: {path}")

    mime_type, _ = mimetypes.guess_type(path)

    if mime_type not in SUPPORTED_IMAGE_TYPES:
        print(format_error_message(
            function_name="validate_image_file",
            error_type="ValueError",
            context=f"Image file '{path}'",
            error_message=f"Unsupported MIME type: {mime_type}",
            helpful_tip="Use JPG, PNG, WEBP, or GIF.",
        ))
        raise ValueError(
            f"Unsupported image type for {path}. "
            f"Detected: {mime_type}. Use JPG, PNG, WEBP, or GIF."
        )

    return mime_type


def encode_image_as_data_url(image_path: str | Path) -> str:
    """
    Convert an image file into a base64 data URL for the OpenAI API.
    """
    path = Path(image_path)
    mime_type = validate_image_file(path)

    with path.open("rb") as image_file:
        b64_image = base64.b64encode(image_file.read()).decode("utf-8")

    return f"data:{mime_type};base64,{b64_image}"


def create_product_listing_prompt(
    product_name: str,
    price: float,
    category: str,
    additional_info: Optional[str] = None
) -> str:
    """
    Create a strong prompt for generating product listings.
    """
    extra_line = f"- Additional Info: {additional_info}" if additional_info else ""

    return f"""You are an expert e-commerce copywriter. Analyze the product image and create a compelling product listing.

Product Information:
- Name: {product_name}
- Price: ${price:.2f}
- Category: {category}
{extra_line}

Create a professional product listing with:

1. Product Title:
   - Catchy, SEO-friendly, 60 characters max.

2. Product Description:
   - 150-200 words.
   - Highlight key features and benefits.
   - Use persuasive but honest language.
   - Include details visible in the image, such as color, material, design, shape, visible pattern, and style.

3. Key Features:
   - 5-7 concise bullet points.
   - Focus on practical buyer benefits.

4. SEO Keywords:
   - 10-15 comma-separated keywords.
   - Relevant to the product, category, design, and use case.

Important:
- Be specific about what is visible in the image.
- Do not invent technical specifications that are not shown or provided.
- If something cannot be confirmed from the image or metadata, avoid claiming it as fact.
"""


def normalize_product(raw: Dict[str, Any], index: int) -> Dict[str, Any]:
    """
    Normalize products from local JSON/CSV or HuggingFace-style fields.
    """
    return {
        "id": get_first_value(raw, PRODUCT_ID_FIELDS, index + 1),
        "name": str(get_first_value(raw, PRODUCT_NAME_FIELDS, f"Product {index + 1}")),
        "price": get_first_value(raw, ("price",), DEFAULT_PRICE),
        "category": str(get_first_value(raw, CATEGORY_FIELDS, "General")),
        "image_path": str(get_first_value(raw, IMAGE_PATH_FIELDS, "")),
        "additional_info": collect_additional_info(raw),
    }


def validate_products_path(products_path: str | Path) -> Path:
    """Validate product file path and supported extension."""
    path = Path(products_path)

    if not path.exists():
        print(format_error_message(
            function_name="validate_products_path",
            error_type="FileNotFoundError",
            context=f"Product file '{path}'",
            error_message="Product file was not found.",
            helpful_tip="Check that the file path is correct and the file exists.",
        ))
        raise FileNotFoundError(f"Product file not found: {path}")

    if path.suffix.lower() not in {".json", ".csv"}:
        print(format_error_message(
            function_name="validate_products_path",
            error_type="ValueError",
            context=f"Product file '{path}'",
            error_message=f"Unsupported file extension: {path.suffix}",
            helpful_tip="Use a .json or .csv product file.",
        ))
        raise ValueError("Products file must be .json or .csv")

    return path


def load_product_rows(products_path: str | Path) -> List[Dict[str, Any]]:
    """Load raw product rows from JSON or CSV."""
    path = Path(products_path)

    if path.suffix.lower() == ".json":
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as error:
            print(format_error_message(
                function_name="load_product_rows",
                error_type="JSONDecodeError",
                context=f"File '{path}', line {error.lineno}, column {error.colno}",
                error_message=error.msg,
                helpful_tip="Check JSON syntax at the indicated line and column.",
            ))
            raise

        if not isinstance(data, list):
            print(format_error_message(
                function_name="load_product_rows",
                error_type="ValueError",
                context=f"File '{path}'",
                error_message="JSON products file must contain a list of product objects.",
                helpful_tip="Wrap product objects in a JSON array: [{...}, {...}].",
            ))
            raise ValueError("JSON products file must contain a list of product objects.")
        return data

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            return list(reader)

    raise ValueError("Products file must be .json or .csv")


def process_product_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Normalize raw product rows into the internal product shape."""
    return [normalize_product(item, i) for i, item in enumerate(rows)]


def validate_products(products: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Validate all normalized products."""
    return [validate_product_data(product) for product in products]


def load_products(products_path: str | Path) -> List[Dict[str, Any]]:
    """Load, normalize, and validate products for backward compatibility."""
    path = validate_products_path(products_path)
    rows = load_product_rows(path)
    products = process_product_rows(rows)
    return validate_products(products)


def save_json(data: Any, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def create_mock_listing(product: Dict[str, Any]) -> Dict[str, Any]:
    """
    Offline/demo output for testing file writing and screenshots without spending API credits.
    Mark clearly as mock output in your submission if you use this.
    """
    name = product["name"]
    category = product["category"]

    return {
        "title": f"{name[:45]} | Stylish {category}",
        "description": (
            f"Bring a polished upgrade to your everyday essentials with {name}. "
            f"Designed for shoppers who want style, practicality, and a clean product presentation, "
            f"this {category.lower()} item is positioned as a versatile choice for modern buyers. "
            f"The listing generator would normally analyze the product image to describe visible details "
            f"such as color, silhouette, material cues, and design elements. This mock entry confirms that "
            f"the batch workflow, JSON formatting, and save process are working before using real API calls."
        ),
        "features": [
            "Generated in offline mock mode for testing",
            "Structured JSON output",
            "SEO-friendly title format",
            "Buyer-focused product description",
            "Batch processing compatible",
            "Ready for API replacement"
        ],
        "keywords": f"{name}, {category}, online shopping, product listing, ecommerce, product description, SEO listing, retail, catalog, product copy"
    }


def get_openai_client():
    """
    Initialize the OpenAI client using OPENAI_API_KEY from environment variables.
    """
    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:
        raise EnvironmentError(
            "OPENAI_API_KEY is not set. Set it before running real API calls.\n"
            "Mac/Linux: export OPENAI_API_KEY='your_key_here'\n"
            "Windows PowerShell: $env:OPENAI_API_KEY='your_key_here'\n"
            "Or create a local .env file with OPENAI_API_KEY='your_key_here'"
        )

    from openai import OpenAI
    return OpenAI(api_key=api_key)


def build_openai_input(prompt: str, image_data_url: str) -> List[Dict[str, Any]]:
    """Build the message input sent to the OpenAI API."""
    return [
        {
            "role": "system",
            "content": SYSTEM_PROMPT,
        },
        {
            "role": "user",
            "content": [
                {"type": "input_text", "text": prompt},
                {"type": "input_image", "image_url": image_data_url, "detail": "auto"},
            ],
        },
    ]


def parse_openai_listing_response(response: Any) -> Dict[str, Any]:
    """Parse a structured OpenAI response into a plain dictionary."""
    listing = response.output_parsed
    return listing.model_dump()


def call_openai_vision_api(
    client: Any,
    prompt: str,
    image_data_url: str,
    model: str = DEFAULT_MODEL,
    temperature: float = DEFAULT_TEMPERATURE,
) -> Dict[str, Any]:
    """
    Call the OpenAI API and return a structured listing.
    """
    response = client.responses.parse(
        model=model,
        temperature=temperature,
        input=build_openai_input(prompt, image_data_url),
        text_format=ProductListing,
    )
    return parse_openai_listing_response(response)


def generate_product_listing(client: Any, product: Dict[str, Any], model: str) -> Dict[str, Any]:
    """Prepare product inputs and generate one listing."""
    image_data_url = encode_image_as_data_url(product["image_path"])
    prompt = create_product_listing_prompt(
        product_name=product["name"],
        price=product["price"],
        category=product["category"],
        additional_info=product.get("additional_info"),
    )
    return call_openai_vision_api(client, prompt, image_data_url, model=model)


def is_retryable_error(error: Exception) -> bool:
    """
    Identify errors that are worth retrying.
    Uses class-name checks so the script remains robust across SDK versions.
    """
    retryable_names = {
        "RateLimitError",
        "APITimeoutError",
        "APIConnectionError",
        "InternalServerError",
    }
    return error.__class__.__name__ in retryable_names


def generate_listing_with_retries(
    client: Any,
    product: Dict[str, Any],
    model: str,
    max_retries: int = DEFAULT_MAX_RETRIES,
    base_delay: float = DEFAULT_RETRY_BASE_DELAY,
) -> Dict[str, Any]:
    """
    Generate a product listing with exponential backoff for retryable errors.
    """
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            return generate_product_listing(client, product, model=model)
        except Exception as error:
            last_error = error

            if not is_retryable_error(error) or attempt == max_retries:
                raise

            wait_time = base_delay * (2 ** (attempt - 1)) + random.uniform(0, 0.5)
            print(
                f"⚠ Retryable API error on attempt {attempt}/{max_retries}: "
                f"{error.__class__.__name__}. Waiting {wait_time:.1f}s..."
            )
            time.sleep(wait_time)

    raise last_error  # Defensive fallback


def select_products(products: List[Dict[str, Any]], limit: Optional[int]) -> List[Dict[str, Any]]:
    """Select the products to process."""
    return products[:limit] if limit else products


def build_success_result(
    product: Dict[str, Any],
    listing: Dict[str, Any],
    started_at: float,
) -> Dict[str, Any]:
    """Format a successful product result."""
    return {
        "status": "success",
        "product": product,
        "listing": listing,
        "processing_seconds": round(time.time() - started_at, 2),
    }


def build_failure_result(
    product: Dict[str, Any],
    error: Exception,
    started_at: float,
) -> Dict[str, Any]:
    """Format a failed product result."""
    return {
        "status": "failed",
        "product": product,
        "error_type": error.__class__.__name__,
        "error_message": str(error),
        "processing_seconds": round(time.time() - started_at, 2),
    }


def process_single_product(
    product: Dict[str, Any],
    client: Any,
    model: str,
    mock: bool,
) -> Dict[str, Any]:
    """Process one product and return its listing."""
    if mock:
        return create_mock_listing(product)
    return generate_listing_with_retries(client, product, model=model)


def summarize_results(results: List[Dict[str, Any]]) -> Dict[str, int]:
    """Count successful and failed product results."""
    successful = sum(1 for item in results if item["status"] == "success")
    return {
        "successful": successful,
        "failed": len(results) - successful,
    }


def print_processing_header(selected_count: int, model: str, mock: bool) -> None:
    """Print the batch processing header."""
    print(f"\nProcessing {selected_count} product(s)...")
    print(f"Mode: {'MOCK / offline' if mock else 'REAL OpenAI API'}")
    print(f"Model: {model if not mock else 'N/A'}\n")


def print_processing_summary(results: List[Dict[str, Any]], output_path: str | Path) -> None:
    """Print the batch processing summary."""
    summary = summarize_results(results)
    print("\nDone.")
    print(f"Successful listings: {summary['successful']}")
    print(f"Failed listings: {summary['failed']}")
    print(f"Saved to: {output_path}")


def process_product_batch(
    products: List[Dict[str, Any]],
    client: Any,
    model: str,
    delay_seconds: float,
    mock: bool,
) -> List[Dict[str, Any]]:
    """Process selected products and return result records."""
    results = []

    for i, product in enumerate(products, start=1):
        print(f"[{i}/{len(products)}] {product['name']}")
        started_at = time.time()

        try:
            listing = process_single_product(product, client, model=model, mock=mock)
            result = build_success_result(product, listing, started_at)

            if not mock:
                time.sleep(delay_seconds)

            print("  ✓ Listing generated")

        except Exception as error:
            result = build_failure_result(product, error, started_at)
            print(format_error_message(
                function_name="process_product_batch",
                error_type=error.__class__.__name__,
                context=f"Product '{product.get('name', 'unknown')}'",
                error_message=str(error),
                helpful_tip="Check product data, image path, API key, network connection, and retryable API errors.",
            ))

        results.append(result)

    return results


def save_results(results: List[Dict[str, Any]], output_path: str | Path) -> None:
    """Save generated product listing results."""
    save_json(results, output_path)


def process_products(
    products: List[Dict[str, Any]],
    output_path: str | Path,
    model: str,
    limit: Optional[int] = None,
    delay_seconds: float = DEFAULT_DELAY_SECONDS,
    mock: bool = False,
) -> List[Dict[str, Any]]:
    """
    Process products in batch and save results.
    """
    selected = select_products(products, limit)
    client = None if mock else get_openai_client()

    print_processing_header(len(selected), model, mock)
    results = process_product_batch(selected, client, model, delay_seconds, mock)
    save_results(results, output_path)
    print_processing_summary(results, output_path)
    return results


def load_huggingface_dataset(limit: int) -> Any:
    """Load product rows from the HuggingFace dataset."""
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "To use --prepare-hf, install datasets and pillow:\n"
            "pip install datasets pillow"
        ) from exc

    return load_dataset(DEFAULT_HF_DATASET, split=f"train[:{limit}]")


def create_images_directory(images_dir: str | Path) -> Path:
    """Create and return the product image output directory."""
    images_path = Path(images_dir)
    images_path.mkdir(parents=True, exist_ok=True)
    return images_path


def save_huggingface_image(row: Dict[str, Any], images_path: Path, product_id: Any) -> Optional[Path]:
    """Save one HuggingFace product image and return its path."""
    image = row.get("image")

    if image is None:
        return None

    image_file = images_path / f"product_{product_id}.jpg"
    image.save(image_file)
    return image_file


def build_huggingface_product(row: Dict[str, Any], index: int, image_file: Path) -> Dict[str, Any]:
    """Format one HuggingFace row as a product dictionary."""
    raw_product = {
        **row,
        "id": row.get("id", index + 1),
        "name": get_first_value(row, ("productDisplayName", "articleType"), f"Fashion Product {index + 1}"),
        "price": round(random.uniform(DEFAULT_HF_PRICE_MIN, DEFAULT_HF_PRICE_MAX), 2),
        "category": get_first_value(row, ("articleType", "subCategory", "masterCategory"), "Fashion"),
        "image_path": str(image_file),
    }
    return validate_product_data(normalize_product(raw_product, index))


def process_huggingface_rows(dataset: Any, images_path: Path) -> List[Dict[str, Any]]:
    """Convert HuggingFace dataset rows into validated local products."""
    products = []

    for i, row in enumerate(dataset):
        product_id = row.get("id", i + 1)
        image_file = save_huggingface_image(row, images_path, product_id)

        if image_file is None:
            print(f"Skipping product {product_id}: no image found")
            continue

        products.append(build_huggingface_product(row, i, image_file))

    return products


def prepare_huggingface_dataset(limit: int, output_products_path: str, images_dir: str) -> List[Dict[str, Any]]:
    """
    Optional helper for the lab dataset.

    Downloads the first N products from HuggingFace, saves images locally,
    and creates products.json.
    """
    print(f"Loading HuggingFace dataset with {limit} product(s)...")
    dataset = load_huggingface_dataset(limit)
    images_path = create_images_directory(images_dir)
    products = process_huggingface_rows(dataset, images_path)
    save_json(products, output_products_path)

    print("\n✓ HuggingFace dataset prepared")
    print(f"  Products file: {output_products_path}")
    print(f"  Images folder: {images_dir}")
    print(f"  Total products: {len(products)}")

    return products


def load_validate_and_process_products(products_path: str | Path) -> List[Dict[str, Any]]:
    """Run the product input workflow as separate steps."""
    path = validate_products_path(products_path)
    rows = load_product_rows(path)
    products = process_product_rows(rows)
    return validate_products(products)


def validate_first_product_image(products: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    """Validate the first product image when products are available."""
    if not products:
        return None

    first = products[0]
    validate_image_file(first["image_path"])
    return first


def is_api_key_configured() -> bool:
    """Return whether OPENAI_API_KEY is configured."""
    return bool(os.getenv("OPENAI_API_KEY"))


def validate_setup(products_path: str | Path) -> None:
    """Quick setup validation before running paid API calls."""
    print("\nValidating setup...")

    products = load_validate_and_process_products(products_path)
    print(f"✓ Product file loaded: {len(products)} product(s)")

    first = validate_first_product_image(products)
    if first:
        print(f"✓ First product: {first['name']}")
        print("✓ First image validated successfully")

    if is_api_key_configured():
        print("✓ OPENAI_API_KEY is set")
    else:
        print("⚠ OPENAI_API_KEY is not set. Mock mode will work, real API mode will not.")

    print("Validation complete.\n")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Generate e-commerce product listings from product images using OpenAI vision models."
    )

    parser.add_argument("--products", default="products.json", help="Path to products JSON or CSV file.")
    parser.add_argument("--output", default="generated_listings.json", help="Where to save generated listings.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI model name.")
    parser.add_argument("--limit", type=int, default=None, help="Maximum number of products to process.")
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY_SECONDS, help="Delay between successful API calls.")
    parser.add_argument("--mock", action="store_true", help="Run without calling OpenAI API. Useful for testing.")
    parser.add_argument("--validate-only", action="store_true", help="Validate product file, image encoding, and API key.")
    parser.add_argument("--prepare-hf", action="store_true", help="Download and prepare HuggingFace fashion products dataset.")
    parser.add_argument("--hf-limit", type=int, default=10, help="Number of HuggingFace products to prepare.")
    parser.add_argument("--images-dir", default="product_images", help="Directory for prepared/downloaded product images.")

    return parser


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()

    if args.prepare_hf:
        prepare_huggingface_dataset(
            limit=args.hf_limit,
            output_products_path=args.products,
            images_dir=args.images_dir,
        )
        return

    if args.validate_only:
        validate_setup(args.products)
        return

    products = load_validate_and_process_products(args.products)

    process_products(
        products=products,
        output_path=args.output,
        model=args.model,
        limit=args.limit,
        delay_seconds=args.delay,
        mock=args.mock,
    )


if __name__ == "__main__":
    main()
