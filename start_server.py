#!/usr/bin/env python3
"""
Start the Synthetica web server.
"""
import os
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

if __name__ == "__main__":
    import uvicorn

    # Check for API key
    if not os.getenv("ANTHROPIC_API_KEY"):
        print("=" * 60)
        print("WARNING: ANTHROPIC_API_KEY environment variable not set")
        print("=" * 60)
        print()
        print("The server will start, but API calls will fail.")
        print("Please set your API key:")
        print("  export ANTHROPIC_API_KEY='your-api-key-here'")
        print()
        print("=" * 60)
        print()

    print("Starting Synthetica Web Server...")
    print("Access the web interface at: http://localhost:8000")
    print("API documentation at: http://localhost:8000/docs")
    print()

    uvicorn.run(
        "synthetica.api.app:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    )
