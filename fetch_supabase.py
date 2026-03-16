#!/usr/bin/env python3
"""
Fetch data from Supabase dataset_fidc table.
Run this script directly in your terminal (not through VS Code sandbox).
"""
import os
from supabase import create_client
import json

# Your credentials from the browser session
SUPABASE_URL = "https://ulxfhbyvbjsivbpcmyim.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsImtpZCI6IjFYa0szS3lYVGxZcElPS2ciLCJ0eXAiOiJKV1QifQ.eyJpc3MiOiJodHRwczovL3VseGZoYnl2YmpzaXZicGNteWltLnN1cGFiYXNlLmNvL2F1dGgvdjEiLCJzdWIiOiI3MThkNjRkZC0wMDkwLTQwN2ItYjMzYS1lODMzYmFhOTZjMGYiLCJhdWQiOiJhdXRoZW50aWNhdGVkIiwiZXhwIjoxNzczNjkwMjQ0LCJpYXQiOjE3NzM2ODY2NDQsImVtYWlsIjoicGVkcm90b2Rlc2NhbkBnbWFpbC5jb20iLCJwaG9uZSI6IiIsImFwcF9tZXRhZGF0YSI6eyJwcm92aWRlciI6ImVtYWlsIiwicHJvdmlkZXJzIjpbImVtYWlsIl19LCJ1c2VyX21ldGFkYXRhIjp7ImVtYWlsX3ZlcmlmaWVkIjp0cnVlfSwicm9sZSI6ImF1dGhlbnRpY2F0ZWQiLCJhYWwiOiJhYWwxIiwiYW1yIjpbeyJtZXRob2QiOiJwYXNzd29yZCIsInRpbWVzdGFtcCI6MTc3MzY4NjY0NH1dLCJzZXNzaW9uX2lkIjoiZDZjZDdlMDYtZmQyNy00NzQ3LTkwNDEtOGM3MDMxYjMyY2QzIiwiaXNfYW5vbnltb3VzIjpmYWxzZX0.jW6L37Lo-1NhH6Ws0qY47pA5btKH1-sIZuvUHNzP5LQ"

def main():
    print("Connecting to Supabase...")
    supabase = create_client(SUPABASE_URL, SUPABASE_KEY)
    
    try:
        # Fetch first 5 rows to test connection
        print("Fetching data from dataset_fidc table...")
        response = supabase.table("dataset_fidc").select("*").limit(5).execute()
        
        print(f"\n✓ Success! Retrieved {len(response.data)} rows\n")
        print("Sample data:")
        print(json.dumps(response.data, indent=2, ensure_ascii=False))
        
        # Get total count
        count_response = supabase.table("dataset_fidc").select("*", count="exact").execute()
        print(f"\n✓ Total rows in table: {count_response.count}")
        
    except Exception as e:
        print(f"\n✗ Error: {e}")
        print("\nTroubleshooting:")
        print("1. Check if your token expired (expires_at: 1773690244)")
        print("2. Verify table name is 'dataset_fidc'")
        print("3. Check if you have read permissions on this table")

if __name__ == "__main__":
    main()
