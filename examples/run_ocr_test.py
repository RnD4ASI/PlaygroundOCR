"""
Example script to demonstrate the usage of the Generator's get_ocr method
for processing a PDF file with different Hugging Face OCR models.

Instructions:
1. Ensure the `generator.py` file is in the parent directory or adjust the import path.
2. Make sure the required Hugging Face models are downloaded and available in the
   local model directory specified in the Generator's configuration (e.g., `../model/`).
   Models to have locally:
     - nanonets/Nanonets-OCR-s
     - stepfun-ai/GOT-OCR2_0
     - The default Hugging Face OCR model specified in your `config_model.json`
       (if different from the ones above).
3. Place the test PDF file (e.g., `F2024L01074.pdf`) in the same directory as this
   script, or update the `pdf_path` variable.
4. Ensure all dependencies for `generator.py` are installed, including `pdf2image`
   and its system dependency `poppler`.
"""
import os
import sys
import logging

# Adjust the path to import Generator from the parent directory
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

try:
    from generator import Generator
except ImportError:
    print("Failed to import Generator. Ensure generator.py is in the parent directory or update sys.path.")
    sys.exit(1)

# Configure basic logging for the script
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def run_ocr_tests():
    """
    Runs OCR tests using different models and formats.
    """
    logging.info("Initializing Generator for OCR tests...")
    try:
        # Initialize the Generator.
        # The Generator's __init__ loads configurations.
        # OCR models themselves are loaded on-demand by the get_ocr method.
        gen = Generator()
        logging.info("Generator initialized successfully.")
    except Exception as e:
        logging.error(f"Fatal error initializing Generator: {e}", exc_info=True)
        print(f"Fatal error initializing Generator: {e}. Check logs for details. Ensure configurations are correct.")
        return

    # Define the path to the PDF file for testing.
    # Assumes the PDF is in the same directory as this script.
    # If your PDF is elsewhere, update this path.
    pdf_path = os.path.join(current_dir, "F2024L01074.pdf") # Assumes PDF is in examples/
    # If F2024L01074.pdf is in the root, use:
    # pdf_path = os.path.join(parent_dir, "F2024L01074.pdf")


    if not os.path.exists(pdf_path):
        logging.error(f"Test PDF not found at: {pdf_path}")
        print(f"Test PDF not found: {pdf_path}. Please place it in the correct location or update the path in the script.")
        return

    # Define test cases: (model_name, output_format, description)
    # The default_hf_ocr_model is loaded from the Generator's config.
    default_model_name = None
    try:
        default_model_name = gen.default_hf_ocr_model
        logging.info(f"Identified default HF OCR model from config: {default_model_name}")
    except AttributeError:
        logging.warning("Could not determine default_hf_ocr_model from Generator instance. Skipping default model test if not explicitly listed.")
    except Exception as e:
        logging.warning(f"Error accessing default_hf_ocr_model: {e}. Skipping default model test if not explicitly listed.")


    test_cases = [
        ("nanonets/Nanonets-OCR-s", "markdown", "Nanonets Markdown OCR"),
        ("nanonets/Nanonets-OCR-s", "text", "Nanonets Text OCR"),
        ("stepfun-ai/GOT-OCR2_0", "markdown", "GOT-OCR2_0 Markdown OCR"),
        ("stepfun-ai/GOT-OCR2_0", "text", "GOT-OCR2_0 Text OCR"),
    ]

    # Add a test for the actual default model if it's set and different from already listed models
    if default_model_name and default_model_name not in [tc[0] for tc in test_cases]:
        logging.info(f"Adding test case for default model: {default_model_name}")
        test_cases.append((default_model_name, "markdown", f"Default ({default_model_name}) Markdown OCR"))

    for model_name, out_format, desc in test_cases:
        logging.info(f"--- Starting Test: {desc} (Model: {model_name}, Format: {out_format}) ---")
        try:
            # The get_ocr function saves the output file and returns None.
            gen.get_ocr(pdf_file_path=pdf_path, model=model_name, output_format=out_format)

            # Determine the expected output file path based on get_ocr's logic
            pdf_basename = os.path.splitext(os.path.basename(pdf_path))[0]
            # get_ocr saves to: os.path.join(os.path.dirname(pdf_file_path), "db", "preprocess", f"{pdf_name}.{output_format}")
            # If pdf_path is in examples/, dirname is examples/. Output will be examples/db/preprocess/...
            expected_output_dir = os.path.join(os.path.dirname(pdf_path), "db", "preprocess")
            expected_output_file = os.path.join(expected_output_dir, f"{pdf_basename}.{out_format}")

            logging.info(f"Checking for output file at: {expected_output_file}")
            if os.path.exists(expected_output_file):
                logging.info(f"SUCCESS: Output file created for {desc}.")
                # Optionally, print a sample of the content
                # with open(expected_output_file, 'r', encoding='utf-8') as f:
                #     content_sample = f.read(300)
                #     logging.info(f"Content sample for {desc}:\n{content_sample}...\n")
                print(f"SUCCESS: Test '{desc}' completed. Output file: {expected_output_file}")
            else:
                logging.error(f"FAILURE: Output file NOT created for {desc} at {expected_output_file}")
                print(f"FAILURE: Test '{desc}'. Output file not found: {expected_output_file}")
        except Exception as e:
            logging.error(f"ERROR during test '{desc}': {e}", exc_info=True)
            print(f"ERROR: Test '{desc}' failed: {e}")
        logging.info(f"--- Finished Test: {desc} ---")

if __name__ == "__main__":
    # Note: Ensure your current working directory is the root of the project
    # if generator.py uses relative paths for its configs or model directory,
    # or if the PDF path is relative to the root.
    # This script assumes it's run from within the `examples` directory or that
    # paths are adjusted accordingly.

    # Best practice: Run from the project root directory: python examples/run_ocr_test.py
    # This makes relative paths in generator.py (like for ./config or ./model) work as expected.

    # If F2024L01074.pdf is in the project root, not in examples/:
    # Script needs adjustment for pdf_path = os.path.join(parent_dir, "F2024L01074.pdf")
    # And the output directory logic in get_ocr will be relative to that.
    # For simplicity, this script assumes F2024L01074.pdf is copied or placed into examples/
    # or the path is correctly set to its location relative to the script or absolute.

    print("Starting OCR demonstration script.")
    print("Please ensure models are downloaded and paths are correctly set.")
    print("Outputs will be saved in a 'db/preprocess' subdirectory relative to the PDF's location.")
    print("If PDF is in 'examples/', output is in 'examples/db/preprocess/'.")
    print("If PDF is in root, output is in 'db/preprocess/'.")
    print("This script assumes F2024L01074.pdf is in the 'examples/' directory for simplicity of output pathing.")

    run_ocr_tests()

    print("\nOCR demonstration script finished.")
    print("Please check the 'db/preprocess' directory (relative to your PDF's location) for output files and review their contents.")
