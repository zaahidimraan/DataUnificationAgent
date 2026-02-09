import os
import sys
from dotenv import load_dotenv

# CRITICAL: Set console encoding to UTF-8 for Windows (must be first)
if sys.platform == 'win32':
    try:
        # Enable UTF-8 mode for Windows console
        import ctypes
        kernel32 = ctypes.windll.kernel32
        kernel32.SetConsoleOutputCP(65001)  # UTF-8 code page
        # Reconfigure stdout/stderr
        sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None
        sys.stderr.reconfigure(encoding='utf-8') if hasattr(sys.stderr, 'reconfigure') else None
    except Exception as e:
        print(f"Warning: Could not set UTF-8 encoding: {e}")

from app import create_app

# Load environment variables
load_dotenv()

# Check key
if not os.getenv("GOOGLE_API_KEY"):
    print("ERROR: GOOGLE_API_KEY not found. Check your .env file.")
else:
    print("SUCCESS: Google API Key detected.")

app = create_app()

if __name__ == '__main__':
    # CRITICAL FIX: use_reloader=False prevents restart on file upload
    app.run(debug=True, port=5000, use_reloader=False)