"""
Dashboard startup script.

Launches the Streamlit dashboard application.
"""

import subprocess
import sys
from pathlib import Path


def main():
    """Start the dashboard"""
    # Get the project root directory
    project_root = Path(__file__).parent.parent
    dashboard_app = project_root / "src" / "observability" / "dashboard" / "app.py"

    if not dashboard_app.exists():
        print(f"Error: Dashboard app not found at {dashboard_app}")
        sys.exit(1)

    print("Starting RAG-MCP Dashboard...")
    print(f"App location: {dashboard_app}")
    print("Dashboard will open in your browser")
    print("Press Ctrl+C to stop\n")

    # Launch streamlit
    try:
        subprocess.run(
            ["streamlit", "run", str(dashboard_app)],
            check=True
        )
    except KeyboardInterrupt:
        print("\nDashboard stopped")
    except subprocess.CalledProcessError as e:
        print(f"Error starting dashboard: {e}")
        sys.exit(1)
    except FileNotFoundError:
        print("Error: streamlit not found. Please install it:")
        print("   pip install streamlit")
        sys.exit(1)


if __name__ == "__main__":
    main()
