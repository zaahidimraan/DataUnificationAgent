import os
from dotenv import load_dotenv
from app import create_app

# Load environment variables
load_dotenv()

# Check key
if not os.getenv("GOOGLE_API_KEY"):
    print("❌ ERROR: GOOGLE_API_KEY not found. Check your .env file.")
else:
    print("✅ Google API Key detected.")

app = create_app()

if __name__ == '__main__':
    # CRITICAL FIX: use_reloader=False prevents restart on file upload
    app.run(debug=True, port=5000)