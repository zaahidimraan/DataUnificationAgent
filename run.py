import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Check if key is loaded (for debugging)
if not os.getenv("GOOGLE_API_KEY"):
    print("❌ ERROR: GOOGLE_API_KEY not found. Check your .env file.")
else:
    print("✅ Google API Key detected.")

from app import create_app

app = create_app()

if __name__ == '__main__':
    app.run(debug=True, port=5000)