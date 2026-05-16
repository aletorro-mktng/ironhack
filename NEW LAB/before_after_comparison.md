# Before/After Comparison

## Before Refactoring

The original `product_listing_generator.py` worked, but several functions combined multiple responsibilities.

`process_products()` selected products, created the OpenAI client, processed each product, called the API or mock generator, handled errors, formatted result dictionaries, saved output, and printed the final summary.

`call_openai_vision_api()` encoded the image, created the prompt, built the API request, called OpenAI, and parsed the response.

`load_products()` checked the file path, loaded JSON or CSV data, parsed the file contents, normalized product records, and performed some validation.

Some values were hardcoded directly inside functions, such as default prices, retry counts, delay values, supported image types, and prompt text.

## After Refactoring

The refactored `product_listing_generator_refactored.py` separates the workflow into smaller helper functions with clearer responsibilities.

Loading is handled by:

- `validate_products_path()`
- `load_product_rows()`

Validation is handled by:

- `ProductInput`
- `validate_product_data()`
- `validate_products()`
- `validate_image_file()`

Processing is handled by:

- `process_product_rows()`
- `process_single_product()`
- `process_product_batch()`
- `load_validate_and_process_products()`

API-related work is handled by:

- `create_product_listing_prompt()`
- `build_openai_input()`
- `call_openai_vision_api()`
- `parse_openai_listing_response()`
- `generate_product_listing()`

Output formatting is handled by:

- `build_success_result()`
- `build_failure_result()`

Saving is handled by:

- `save_json()`
- `save_results()`

Hardcoded values were moved into constants near the top of the file, including:

- `DEFAULT_PRICE`
- `DEFAULT_DELAY_SECONDS`
- `DEFAULT_MAX_RETRIES`
- `DEFAULT_RETRY_BASE_DELAY`
- `DEFAULT_TEMPERATURE`
- `SUPPORTED_IMAGE_TYPES`
- `SYSTEM_PROMPT`

## Result

The refactored version keeps the same core behavior while making the code easier to test, debug, and maintain. Error messages are also clearer because they now include the function name, error type, location, message, and suggestion.
