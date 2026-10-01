import urllib.request
import json
import os

# Try to get the file list from Zenodo API
api_url = 'https://zenodo.org/api/records/2530835'
try:
    with urllib.request.urlopen(api_url) as response:
        data = json.load(response)
        for file_info in data.get('files', []):
            print(f"File: {file_info['key']}, Size: {file_info['size']}, URL: {file_info['links']['self']}")
except Exception as e:
    print(f'Error: {e}')