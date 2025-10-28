"""
Test script to demonstrate X-ray image verification feature
"""
from PIL import Image
import io

def create_test_images():
    """Create sample test images"""
    
    # Create a simple non-medical image (just a colored rectangle)
    img = Image.new('RGB', (400, 400), color='red')
    img.save('test_fake_image.jpg')
    print("✅ Created test_fake_image.jpg (non-medical image)")
    
    print("\n📝 To test the security feature:")
    print("1. Start the web app: python web_app.py")
    print("2. Go to http://localhost:5000")
    print("3. Upload 'test_fake_image.jpg' - should be rejected")
    print("4. Upload a real chest X-ray - should be accepted")
    print("\n⚠️ Make sure GOOGLE_API_KEY is set in your .env file for verification to work!")

if __name__ == "__main__":
    create_test_images()
