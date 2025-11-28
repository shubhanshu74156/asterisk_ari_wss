import requests

USERNAME = "ai_agent"
PASSWORD = "mysecurepassword"
HOST = "localhost:8088"
APP = "my_app"

BASE_URL = f"http://{HOST}/ari"

recording_name = "rec_59d9d78ac4e144798d7f0c65c15764a1"
url = f"{BASE_URL}/recordings/stored/{recording_name}/file"

# Use stream=True for large files
res = requests.get(url, auth=(USERNAME, PASSWORD), stream=True)

if res.status_code == 200:
    # Save to a local file
    with open(f"{recording_name}.wav", "wb") as f:
        for chunk in res.iter_content(chunk_size=1024):
            if chunk:  # filter out keep-alive chunks
                f.write(chunk)
    print(f"Recording saved as {recording_name}.wav")
else:
    print("Failed to fetch recording:", res.status_code, res.text)

