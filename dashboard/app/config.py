import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


# Configuration settings
class Config:
    # Use environment variable or default to VM IP
    API_HOST = os.getenv("API_HOST", "34.203.240.184")  # Replace with your VM's IP
    API_PORT = os.getenv("API_PORT", "8000")
    DASHBOARD_HOST = os.getenv(
        "DASHBOARD_HOST", "localhost"
    )  # Use localhost for local testing
    DASHBOARD_PORT = os.getenv("DASHBOARD_PORT", "8050")

    @classmethod
    def get_api_url(cls):
        return f"http://{cls.API_HOST}:{cls.API_PORT}/predict"


# Create a .env file for environment variables
env_content = """
# API Configuration
API_HOST=34.203.240.184  # Replace with your VM's IP
API_PORT=8000

# Dashboard Configuration
DASHBOARD_HOST=localhost
DASHBOARD_PORT=8050
"""

# Save as .env
with open(".env", "w") as f:
    f.write(env_content)
