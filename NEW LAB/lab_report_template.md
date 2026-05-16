# LAB Report: API Calling to ChatGPT — Product Listing Generator

## Overview

This project implements an automated product listing generator for an e-commerce workflow. The application takes product metadata such as name, price, category, and image path, encodes the product image into base64 format, sends the image and prompt to OpenAI’s vision-capable API, and saves the generated product listing as structured JSON.

## How the API Integration Works

The script uses the OpenAI Python library and authenticates through the `OPENAI_API_KEY` environment variable. This avoids hard-coding secret credentials in the source code. For each product, the script reads the image file, converts it into a base64 data URL, and submits it with a prompt that asks the model to generate a title, description, key features, and SEO keywords.

The response is handled as a structured output using a Pydantic model called `ProductListing`. This ensures that each generated listing follows the expected format:

- `title`
- `description`
- `features`
- `keywords`

The final results are saved to `generated_listings.json`.

## Error Handling

The script includes error handling for common production issues:

- Missing or invalid API key
- Missing image files
- Unsupported image formats
- Empty image files
- Rate limit or temporary API connection failures
- Failed product-level processing

The batch process continues even if one product fails, and each failure is saved in the output JSON with an error type and message. Retry logic with exponential backoff is used for retryable API errors.

## Quality of Generated Listings

The generated listings are professional, buyer-focused, and formatted consistently. The prompt requires the model to mention visible design details from the image while avoiding unsupported claims. This improves listing accuracy and reduces hallucinated product specifications.

The best outputs were produced when the product metadata included additional information such as color, usage, material, or intended audience.

## Challenges Faced

The main challenges were image preparation and JSON consistency. Product images must exist locally and be in a supported format such as JPG, PNG, WEBP, or GIF. JSON parsing can also be unreliable if the model is only prompted to “return JSON,” so the final version uses structured outputs to enforce the required schema.

Another challenge is cost and speed. Vision API calls are slower and more expensive than text-only requests, so the script supports limits, delays, and mock mode for testing before running real calls.

## Potential Improvements

Future improvements could include:

- Multi-image support for front, side, and detail shots
- Cost tracking based on token usage
- Listing quality scoring
- Automatic regeneration for weak listings
- Export to CSV for Shopify, WooCommerce, or another e-commerce platform
- A simple web interface for non-technical users

## Conclusion

This project demonstrates a practical AI automation workflow using API calls, image encoding, structured prompting, batch processing, and error handling. The result is a reusable system that can reduce manual copywriting work and help maintain consistent product listing quality across an e-commerce catalog.
