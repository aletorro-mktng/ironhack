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

from pydantic import BaseModel, Field


DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


class ProductListing(BaseModel):
    """Structured output expected from the OpenAI API."""
    title: str = Field(description="Catchy SEO-friendly product title, max 60 characters.")
    description: str = Field(description="Detailed product description, roughly 150-200 words.")
    features: List[str] = Field(description="5 to 7 product feature bullets.")
    keywords: str = Field(description="10 to 15 comma-separated SEO keywords.")


def encode_image_as_data_url(image_path: str | Path) -> str:
    """
    Convert an image file into a base64 data URL for the OpenAI API.

    Parameters:
    - image_path: Path to JPG, PNG, WEBP, or GIF image.

    Returns:
    - data URL string, e.g. data:image/jpeg;base64,...

    Raises:
    - FileNotFoundError if the image does not exist.
    - ValueError if the file is empty or unsupported.
    """
    path = Path(image_path)

    if not path.exists():
        raise FileNotFoundError(f"Image file not found: {path}")

    if not path.is_file():
        raise ValueError(f"Image path is not a file: {path}")

    if path.stat().st_size == 0:
        raise ValueError(f"Image file is empty: {path}")

    mime_type, _ = mimetypes.guess_type(path)
    supported = {"image/jpeg", "image/png", "image/webp", "image/gif"}

    if mime_type not in supported:
        raise ValueError(
            f"Unsupported image type for {path}. "
            f"Detected: {mime_type}. Use JPG, PNG, WEBP, or GIF."
        )

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
    product_name = (
        raw.get("name")
        or raw.get("productDisplayName")
        or raw.get("product_name")
        or raw.get("title")
        or f"Product {index + 1}"
    )

    category = (
        raw.get("category")
        or raw.get("masterCategory")
        or raw.get("subCategory")
        or raw.get("articleType")
        or "General"
    )

    image_path = raw.get("image_path") or raw.get("image") or raw.get("path")

    raw_price = raw.get("price", 29.99)
    try:
        price = float(raw_price)
    except Exception:
        price = 29.99

    additional_parts = []
    for field in ["gender", "baseColour", "season", "usage", "articleType", "additional_info"]:
        value = raw.get(field)
        if value not in [None, "", "nan"]:
            additional_parts.append(f"{field}: {value}")

    return {
        "id": raw.get("id") or raw.get("product_id") or raw.get("sku") or index + 1,
        "name": str(product_name),
        "price": price,
        "category": str(category),
        "image_path": str(image_path) if image_path else "",
        "additional_info": "; ".join(additional_parts) or None,
    }


def load_products(products_path: str | Path) -> List[Dict[str, Any]]:
    """
    Load products from JSON or CSV.

    JSON format:
    [
      {
        "id": 1,
        "name": "Wireless Headphones",
        "price": 79.99,
        "category": "Electronics",
        "image_path": "product_images/headphones.jpg",
        "additional_info": "Noise cancelling, 30-hour battery"
      }
    ]
    """
    path = Path(products_path)

    if not path.exists():
        raise FileNotFoundError(f"Product file not found: {path}")

    if path.suffix.lower() == ".json":
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise ValueError("JSON products file must contain a list of product objects.")
        return [normalize_product(item, i) for i, item in enumerate(data)]

    if path.suffix.lower() == ".csv":
        with path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            return [normalize_product(row, i) for i, row in enumerate(reader)]

    raise ValueError("Products file must be .json or .csv")


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


def call_openai_vision_api(
    client: Any,
    product: Dict[str, Any],
    model: str = DEFAULT_MODEL,
    temperature: float = 0.7,
) -> Dict[str, Any]:
    """
    Send product image + metadata to OpenAI and return a structured listing.

    Uses the Responses API with a Pydantic structured output model.
    """
    image_data_url = encode_image_as_data_url(product["image_path"])

    prompt = create_product_listing_prompt(
        product_name=product["name"],
        price=product["price"],
        category=product["category"],
        additional_info=product.get("additional_info"),
    )

    response = client.responses.parse(
        model=model,
        temperature=temperature,
        input=[
            {
                "role": "system",
                "content": (
                    "You generate accurate, ethical, conversion-focused e-commerce listings. "
                    "You must not invent product specs that are not visible or provided."
                ),
            },
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {"type": "input_image", "image_url": image_data_url, "detail": "auto"},
                ],
            },
        ],
        text_format=ProductListing,
    )

    listing = response.output_parsed
    return listing.model_dump()


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
    max_retries: int = 3,
    base_delay: float = 2.0,
) -> Dict[str, Any]:
    """
    Generate a product listing with exponential backoff for retryable errors.
    """
    last_error = None

    for attempt in range(1, max_retries + 1):
        try:
            return call_openai_vision_api(client, product, model=model)
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


def process_products(
    products: List[Dict[str, Any]],
    output_path: str | Path,
    model: str,
    limit: Optional[int] = None,
    delay_seconds: float = 1.0,
    mock: bool = False,
) -> List[Dict[str, Any]]:
    """
    Process products in batch and save results.
    """
    selected = products[:limit] if limit else products
    client = None if mock else get_openai_client()

    results = []

    print(f"\nProcessing {len(selected)} product(s)...")
    print(f"Mode: {'MOCK / offline' if mock else 'REAL OpenAI API'}")
    print(f"Model: {model if not mock else 'N/A'}\n")

    for i, product in enumerate(selected, start=1):
        print(f"[{i}/{len(selected)}] {product['name']}")

        started_at = time.time()

        try:
            if mock:
                listing = create_mock_listing(product)
            else:
                listing = generate_listing_with_retries(client, product, model=model)
                time.sleep(delay_seconds)

            result = {
                "status": "success",
                "product": product,
                "listing": listing,
                "processing_seconds": round(time.time() - started_at, 2),
            }
            print("  ✓ Listing generated")

        except Exception as error:
            result = {
                "status": "failed",
                "product": product,
                "error_type": error.__class__.__name__,
                "error_message": str(error),
                "processing_seconds": round(time.time() - started_at, 2),
            }
            print(f"  ✗ Failed: {error.__class__.__name__}: {error}")

        results.append(result)

        # Save after every product so progress is not lost if the run stops.
        save_json(results, output_path)

    successful = sum(1 for item in results if item["status"] == "success")
    failed = len(results) - successful

    print("\nDone.")
    print(f"Successful listings: {successful}")
    print(f"Failed listings: {failed}")
    print(f"Saved to: {output_path}")

    return results


def prepare_huggingface_dataset(limit: int, output_products_path: str, images_dir: str) -> List[Dict[str, Any]]:
    """
    Optional helper for the lab dataset.

    Downloads the first N products from:
    ashraq/fashion-product-images-small

    Saves images locally and creates products.json.
    """
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise ImportError(
            "To use --prepare-hf, install datasets and pillow:\n"
            "pip install datasets pillow"
        ) from exc

    images_path = Path(images_dir)
    images_path.mkdir(parents=True, exist_ok=True)

    print(f"Loading HuggingFace dataset with {limit} product(s)...")
    dataset = load_dataset("ashraq/fashion-product-images-small", split=f"train[:{limit}]")

    products = []

    for i, row in enumerate(dataset):
        product_id = row.get("id", i + 1)
        name = row.get("productDisplayName") or row.get("articleType") or f"Fashion Product {i + 1}"
        category = row.get("articleType") or row.get("subCategory") or row.get("masterCategory") or "Fashion"

        image = row.get("image")
        if image is None:
            print(f"Skipping product {product_id}: no image found")
            continue

        image_file = images_path / f"product_{product_id}.jpg"
        image.save(image_file)

        products.append({
            "id": product_id,
            "name": name,
            "price": round(random.uniform(19.99, 149.99), 2),
            "category": category,
            "image_path": str(image_file),
            "additional_info": (
                f"gender: {row.get('gender')}; "
                f"baseColour: {row.get('baseColour')}; "
                f"season: {row.get('season')}; "
                f"usage: {row.get('usage')}"
            ),
        })

    save_json(products, output_products_path)

    print("\n✓ HuggingFace dataset prepared")
    print(f"  Products file: {output_products_path}")
    print(f"  Images folder: {images_dir}")
    print(f"  Total products: {len(products)}")

    return products


def validate_setup(products_path: str | Path) -> None:
    """
    Quick setup validation before running paid API calls.
    """
    print("\nValidating setup...")

    products = load_products(products_path)
    print(f"✓ Product file loaded: {len(products)} product(s)")

    if products:
        first = products[0]
        print(f"✓ First product: {first['name']}")
        data_url = encode_image_as_data_url(first["image_path"])
        print(f"✓ First image encoded successfully")
        print(f"  Data URL preview: {data_url[:50]}...")

    if os.getenv("OPENAI_API_KEY"):
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
    parser.add_argument("--delay", type=float, default=1.0, help="Delay between successful API calls.")
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

    products = load_products(args.products)

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
