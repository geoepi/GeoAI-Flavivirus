import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Access the OSF token from the environment variable
osf_token = os.getenv("OSF_TOKEN")

if osf_token is None:
    raise ValueError("OSF_TOKEN environment variable is not set.")

# Define OSF project URL (replace with your project URL)
osf_url = "https://osf.io/dn5tc/"

# Headers with authorization
headers = {
    'Authorization': f'Bearer {osf_token}'
}

# Send GET request to OSF API
response = requests.get(osf_url, headers=headers)

if response.status_code == 200:
    print("Data fetched successfully!")
    # Here, you can process or save the response data (like downloading files)
    print(response.json())  # For example, print the JSON response
else:
    print(f"Failed to fetch data: {response.status_code}")
    print(response.text)
