"""
Basic usage example for Smart Image Cropper library.

This example demonstrates how to use the SmartImageCropper with different input types.
"""

import os
from smart_image_cropper import SmartImageCropper, SmartCropperError
from PIL import Image


def main():
    # Initialize the cropper with your API credentials
    # Replace these with your actual API URL and key
    API_URL = os.getenv("SMART_CROPPER_API_URL",
                        "https://your-api-endpoint.com/detect")
    API_KEY = os.getenv("SMART_CROPPER_API_KEY", "your-api-key")

    if API_URL == "https://your-api-endpoint.com/detect" or API_KEY == "your-api-key":
        print("⚠️  Please set your API_URL and API_KEY environment variables or update this script")
        print("   export SMART_CROPPER_API_URL='your-api-url'")
        print("   export SMART_CROPPER_API_KEY='your-api-key'")
        return

    cropper = SmartImageCropper(api_url=API_URL, api_key=API_KEY)

    # Example 1: Process from URL
    print("🔗 Processing image from URL...")
    try:
        image_url = "https://example.com/sample-image.jpg"
        result_bytes = cropper.process_image(image_url)

        # Save the result
        with open("result_from_url.jpg", "wb") as f:
            f.write(result_bytes)
        print("✅ Saved result_from_url.jpg")

    except SmartCropperError as e:
        print(f"❌ Error processing URL: {e}")

    # Example 2: Process from file bytes
    print("\n📁 Processing image from file bytes...")
    try:
        # Make sure you have a test image file
        if os.path.exists("test_image.jpg"):
            with open("test_image.jpg", "rb") as f:
                image_bytes = f.read()

            result_bytes = cropper.process_image(image_bytes)

            # Save the result
            with open("result_from_bytes.jpg", "wb") as f:
                f.write(result_bytes)
            print("✅ Saved result_from_bytes.jpg")
        else:
            print("⚠️  test_image.jpg not found, skipping this example")

    except SmartCropperError as e:
        print(f"❌ Error processing bytes: {e}")

    # Example 3: Process from PIL Image
    print("\n🖼️  Processing PIL Image...")
    try:
        if os.path.exists("test_image.jpg"):
            pil_image = Image.open("test_image.jpg")
            result_bytes = cropper.process_image(pil_image)

            # Save the result
            with open("result_from_pil.jpg", "wb") as f:
                f.write(result_bytes)
            print("✅ Saved result_from_pil.jpg")
        else:
            print("⚠️  test_image.jpg not found, skipping this example")

    except SmartCropperError as e:
        print(f"❌ Error processing PIL image: {e}")

    print("\n🎉 Example completed! Check the output files.")


if __name__ == "__main__":
    main()
