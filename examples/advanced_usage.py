"""
Advanced usage example for Smart Image Cropper library.

This example demonstrates all the different modes of operation:
1. Polling mode (default) - waits for completion and returns bounding boxes
2. Webhook mode - returns job ID and processes results asynchronously
3. Single mode - just sends the request without waiting for results
"""

import os
import time
from flask import Flask, request, jsonify
from smart_image_cropper import SmartImageCropper, SmartCropperError
from smart_image_cropper.cropper import BoundingBox
from dotenv import load_dotenv

load_dotenv()

# Initialize Flask app for webhook handling
app = Flask(__name__)

# Store results from webhook
webhook_results = {}


def main():
    # Initialize the cropper with your API credentials
    API_URL = os.getenv("SMART_CROPPER_API_URL",
                        "https://your-api-endpoint.com/detect")
    API_KEY = os.getenv("SMART_CROPPER_API_KEY", "your-api-key")

    if API_URL == "https://your-api-endpoint.com/detect" or API_KEY == "your-api-key":
        print("⚠️  Please set your API_URL and API_KEY environment variables")
        print("   export SMART_CROPPER_API_URL='your-api-url'")
        print("   export SMART_CROPPER_API_KEY='your-api-key'")
        return

    cropper = SmartImageCropper(api_url=API_URL, api_key=API_KEY)

    # Example 1: Polling Mode (Default)
    print("\n1️⃣ Polling Mode Example")
    print("=====================")
    try:
        # Load test image
        with open("test_image.jpg", "rb") as f:
            image_bytes = f.read()

        # Get bounding boxes using polling mode
        print("🔍 Getting bounding boxes (polling mode)...")
        bboxes = cropper.get_bounding_boxes(image_bytes, mode="polling")

        print(f"✅ Found {len(bboxes)} bounding boxes")

        # Create collage from bounding boxes
        print("🎨 Creating collage...")
        result = cropper.create_collage(image_bytes, bboxes)

        # Show information about the result
        print(f"📊 Result info:")
        print(f"   - Is collage: {result.is_collage}")
        print(f"   - Number of images: {len(result.coordinates)}")

        # Display coordinates of each image in the final result
        for i, coord in enumerate(result.coordinates):
            print(
                f"   - Image {i+1}: x={coord.x}, y={coord.y}, w={coord.width}, h={coord.height}")

        # Save the result
        with open("result_polling.jpg", "wb") as f:
            f.write(result.image_bytes)
        print("✅ Saved result_polling.jpg")

    except SmartCropperError as e:
        print(f"❌ Error in polling mode: {e}")

    # Example 2: Webhook Mode
    print("\n2️⃣ Webhook Mode Example")
    print("=====================")
    try:
        # Start Flask server in a separate thread for webhook handling
        import threading
        server_thread = threading.Thread(target=lambda: app.run(port=5000))
        server_thread.daemon = True
        server_thread.start()

        # Wait for server to start
        time.sleep(1)

        # Get job ID using webhook mode
        webhook_url = "https://webhook.site/4a65ff9f-aca5-4855-8908-20f638dfd518"
        print(f"🔍 Getting job ID (webhook mode, callback to {webhook_url})...")
        job_id = cropper.get_bounding_boxes(
            image_bytes,
            mode="webhook",
            webhook_url=webhook_url
        )

        print(f"✅ Got job ID: {job_id}")
        print("⏳ Waiting for webhook callback...")

        # Wait for webhook callback (in real usage, this would be handled by the webhook)
        time.sleep(5)

        if job_id in webhook_results:
            bboxes = webhook_results[job_id]
            print(f"✅ Received {len(bboxes)} bounding boxes via webhook")

            # Create collage from bounding boxes
            print("🎨 Creating collage...")
            result = cropper.create_collage(image_bytes, bboxes)

            # Show information about the result
            print(f"📊 Result info:")
            print(f"   - Is collage: {result.is_collage}")
            print(f"   - Number of images: {len(result.coordinates)}")

            # Display coordinates of each image in the final result
            for i, coord in enumerate(result.coordinates):
                print(
                    f"   - Image {i+1}: x={coord.x}, y={coord.y}, w={coord.width}, h={coord.height}")

            # Save the result
            with open("result_webhook.jpg", "wb") as f:
                f.write(result.image_bytes)
            print("✅ Saved result_webhook.jpg")
        else:
            print("⚠️ No webhook callback received")

    except SmartCropperError as e:
        print(f"❌ Error in webhook mode: {e}")

    # Example 3: Single Mode
    print("\n3️⃣ Single Mode Example")
    print("=====================")
    try:
        print("🔍 Sending request (single mode)...")
        cropper.get_bounding_boxes(image_bytes, mode="single")
        print("✅ Request sent successfully")
        print("ℹ️  Note: In single mode, you need to check the job status manually")

        # Example of manual status checking
        print("\nChecking job status manually...")
        job_id = "your_job_id"  # In real usage, you would get this from the API
        status = cropper.api_client.get_job_status(job_id)
        if status:
            print(f"✅ Job completed with {len(status)} bounding boxes")

            result = cropper.create_collage(image_bytes, status)
            print(f"📊 Collage created with {len(result.coordinates)} images")

            with open("result_single.jpg", "wb") as f:
                f.write(result.image_bytes)

        else:
            print("⏳ Job still in progress")

    except SmartCropperError as e:
        print(f"❌ Error in single mode: {e}")

    print("\n🎉 Examples completed! Check the output files.")
    print("📄 Coordinate information is now included in the results!")

# Webhook endpoint


@app.route('/webhook', methods=['POST'])
def webhook():
    data = request.json
    job_id = data.get('id')
    if job_id and data.get('status') == 'COMPLETED':
        # Parse bounding boxes from the webhook data
        bboxes = [
            BoundingBox(
                x1=bbox['x1'],
                y1=bbox['y1'],
                x2=bbox['x2'],
                y2=bbox['y2']
            )
            for bbox in data.get('output', [])
        ]
        webhook_results[job_id] = bboxes
    return jsonify({'status': 'ok'})


if __name__ == "__main__":
    main()
