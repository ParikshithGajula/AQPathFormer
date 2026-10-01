import urllib.request
import json

# Try the known CRC-100K record
api_url = 'https://zenodo.org/api/records/2530835'
try:
    with urllib.request.urlopen(api_url) as response:
        data = json.load(response)
        print('Title:', data.get('metadata', {}).get('title', 'N/A'))
except Exception as e:
    print(f'Error: {e}')

# Try the other CRC dataset
api_url2 = 'https://zenodo.org/api/records/4739990'
try:
    with urllib.request.urlopen(api_url2) as response:
        data = json.load(response)
        print('Title:', data.get('metadata', {}).get('title', 'N/A'))
        for file_info in data.get('files', []):
            print(f"  File: {file_info['key']}, Size: {file_info['size']}")
except Exception as e:
    print(f'Error: {e}')