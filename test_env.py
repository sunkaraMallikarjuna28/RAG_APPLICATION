# test_env.py
from dotenv import load_dotenv
import os

def test_env_file():
    print("🔍 Testing .env file...")
    
    # Load environment variables
    load_dotenv()
    
    # Check Google API Key
    google_key = os.getenv("GOOGLE_API_KEY")
    if google_key and google_key != "your_google_api_key_here":
        print("✅ Google API Key: Found")
    else:
        print("❌ Google API Key: Missing or not set")
    
    # Check MongoDB URI
    mongo_uri = os.getenv("MONGODB_URI")
    if mongo_uri and mongo_uri != "your_mongodb_connection_string_here":
        print("✅ MongoDB URI: Found")
    else:
        print("❌ MongoDB URI: Missing or not set")
    
    print("\n📋 Your .env file status:")
    if google_key and mongo_uri:
        print("🎉 .env file is ready!")
    else:
        print("⚠️ Please add your API keys to .env file")

if __name__ == "__main__":
    test_env_file()
