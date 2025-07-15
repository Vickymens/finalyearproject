import requests
import json
import uuid
from datetime import datetime, timezone

url = "https://dfbpzicwgxpnwxqxlkgr.supabase.co/rest/v1/mobile"
api_key = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImRmYnB6aWN3Z3hwbnd4cXhsa2dyIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTI2MDc1MjIsImV4cCI6MjA2ODE4MzUyMn0.5-peIslm3m_8nn4LRLDKsRyRjfrHzPYbffP1waBAoBI"

headers = {
    "apikey": api_key,
    "Authorization": f"Bearer {api_key}",
    "Content-Type": "application/json",
    "Prefer": "return=representation"
}

data = {
    "created_at": datetime.now(timezone.utc).isoformat(),
    "content": "Now the stage is set for the next chapter of our journey.",
}

response = requests.post(url, headers=headers, data=json.dumps(data))
