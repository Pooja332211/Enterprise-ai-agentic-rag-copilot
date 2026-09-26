import asyncio
import json
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from httpx import ASGITransport, AsyncClient
from backend.app.main import app


async def verify_all():
    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://localhost:8000") as client:
        endpoints = ["/docs", "/health", "/health/ready", "/api/v1/health"]
        for endpoint in endpoints:
            response = await client.get(endpoint)
            print(f"=== ENDPOINT: {endpoint} ===")
            print(f"Status Code: {response.status_code}")
            content_type = response.headers.get("content-type", "")
            print(f"Content-Type: {content_type}")
            if "json" in content_type:
                print("Response JSON:")
                print(json.dumps(response.json(), indent=2))
            else:
                print(f"Content length: {len(response.text)} characters")
            print()


if __name__ == "__main__":
    asyncio.run(verify_all())
