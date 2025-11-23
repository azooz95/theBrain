

from microsft_auth import app

from fastapi.testclient import TestClient

client = TestClient(app)
response = client.get(f"/?user_info=hi")

print(response.status_code)