#!/usr/bin/env python3
"""
Fetch data from Supabase dataset_fidc table with automatic authentication.
Run this script directly in your terminal (not through VS Code sandbox).
"""
import os
import sys
import json
import asyncio
from pathlib import Path

try:
    from supabase import create_client
except ImportError:
    print("Error: supabase package not installed")
    print("Run: pip install supabase")
    sys.exit(1)

# Import the auth agent
from supabase_auth_agent import SupabaseAuthAgent


async def main():
    print("=== Supabase Data Fetcher with Auto-Auth ===\n")
    
    # Initialize auth agent
    agent = SupabaseAuthAgent()
    
    # Ensure we have valid credentials
    print("1. Checking authentication...")
    authenticated = await agent.ensure_authenticated()
    
    if not authenticated:
        print("❌ Failed to authenticate. Please check the error messages above.")
        sys.exit(1)
    
    print("\n2. Connecting to Supabase...")
    
    # Load credentials from .env
    token_data = agent._load_cached_token()
    if not token_data:
        print("❌ No cached token found after authentication")
        sys.exit(1)
    
    supabase = create_client(
        agent.supabase_url,
        token_data['access_token']
    )
    
    try:
        print("\n3. Fetching data from dataset_fidc table...")
        response = supabase.table("dataset_fidc").select("*").limit(5).execute()
        
        print(f"\n✓ Success! Retrieved {len(response.data)} rows\n")
        print("=" * 60)
        print("Sample data:")
        print("=" * 60)
        print(json.dumps(response.data, indent=2, ensure_ascii=False))
        
        # Get total count
        print("\n4. Getting total row count...")
        count_response = supabase.table("dataset_fidc").select("*", count="exact").execute()
        print(f"\n✓ Total rows in table: {count_response.count}")
        
        print("\n" + "=" * 60)
        print("✅ All operations completed successfully!")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error fetching data: {e}")
        print("\nTroubleshooting:")
        print("1. Run with --refresh flag: python fetch_supabase.py --refresh")
        print("2. Verify table name is 'dataset_fidc'")
        print("3. Check if you have read permissions on this table")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
