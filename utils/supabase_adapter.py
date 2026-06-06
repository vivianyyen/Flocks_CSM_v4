# test_connection.py - Run this locally first to verify credentials

from supabase import create_client
import os


# Option 2: Use environment variables
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')

print(f"URL: {SUPABASE_URL[:20]}...")
print(f"Key: {SUPABASE_KEY[:20]}...")

try:
    client = create_client(SUPABASE_URL, SUPABASE_KEY)
    response = client.table('cyber_news').select('*').limit(1).execute()
    print("✅ Connection successful!")
    print(f"Data: {response.data}")
except Exception as e:
    print(f"❌ Connection failed: {e}")
