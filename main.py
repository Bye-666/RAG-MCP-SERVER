# Main entry point for MCP Server

import sys
from src.core.settings import load_settings

if __name__ == "__main__":
    try:
        settings = load_settings()
        print("MCP Server initialized with valid configuration")
    except Exception as e:
        print(f"Configuration error: {e}")
        sys.exit(1)