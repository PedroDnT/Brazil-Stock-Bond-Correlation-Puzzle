#!/usr/bin/env python3
"""
Supabase Authentication Agent
Automates browser-based authentication flow for painelfidc.com.br
and maintains fresh access tokens.
"""
import asyncio
import json
import os
import time
from pathlib import Path
from typing import Optional, Dict
import sys

try:
    from playwright.async_api import async_playwright, TimeoutError as PlaywrightTimeout
except ImportError:
    print("Error: playwright not installed.")
    print("Run: pip install playwright && playwright install chromium")
    sys.exit(1)


class SupabaseAuthAgent:
    """Agent that manages Supabase authentication via browser automation."""
    
    def __init__(self, base_dir: str = "."):
        self.base_dir = Path(base_dir)
        self.env_file = self.base_dir / ".env"
        self.cache_file = self.base_dir / ".supabase_cache.json"
        self.url = "https://www.painelfidc.com.br/dataset-fidc"
        self.supabase_url = "https://ulxfhbyvbjsivbpcmyim.supabase.co"
        
    def _load_cached_token(self) -> Optional[Dict]:
        """Load cached token from disk."""
        if not self.cache_file.exists():
            return None
            
        try:
            with open(self.cache_file, 'r') as f:
                data = json.load(f)
                # Check if token is still valid (with 5 min buffer)
                if data.get('expires_at', 0) > (time.time() + 300):
                    return data
        except (json.JSONDecodeError, IOError):
            pass
        return None
    
    def _save_token_cache(self, token_data: Dict):
        """Save token to cache file."""
        with open(self.cache_file, 'w') as f:
            json.dump(token_data, f, indent=2)
    
    def _update_env_file(self, token_data: Dict):
        """Update .env file with new credentials."""
        env_content = f"""SUPABASE_URL={self.supabase_url}
SUPABASE_KEY={token_data['access_token']}
SUPABASE_AUTH_TOKEN={token_data['access_token']}
SUPABASE_REFRESH_TOKEN={token_data.get('refresh_token', '')}
"""
        with open(self.env_file, 'w') as f:
            f.write(env_content)
        print(f"✓ Updated {self.env_file}")
    
    async def _capture_token_from_browser(self) -> Optional[Dict]:
        """Launch browser and capture auth token from localStorage."""
        print("🚀 Launching headless browser...")
        
        async with async_playwright() as p:
            browser = await p.chromium.launch(
                headless=True,
                args=['--disable-blink-features=AutomationControlled']
            )
            
            context = await browser.new_context(
                user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36",
                viewport={'width': 1920, 'height': 1080}
            )
            
            page = await context.new_page()
            
            # Capture network requests to get auth headers
            captured_token = None
            
            async def handle_request(request):
                nonlocal captured_token
                if self.supabase_url in request.url:
                    headers = request.headers
                    if "authorization" in headers and not captured_token:
                        auth = headers["authorization"]
                        if auth.startswith("Bearer "):
                            captured_token = auth.replace("Bearer ", "")
            
            page.on("request", handle_request)
            
            try:
                print(f"📡 Navigating to {self.url}...")
                await page.goto(self.url, wait_until="networkidle", timeout=30000)
                
                # Wait a bit for any async calls
                await asyncio.sleep(3)
                
                # Try to get token from localStorage
                storage_data = await page.evaluate("""
                    () => {
                        const key = Object.keys(localStorage).find(k => 
                            k.includes('supabase') && k.includes('auth-token')
                        );
                        if (key) {
                            return {key: key, value: localStorage.getItem(key)};
                        }
                        return null;
                    }
                """)
                
                if storage_data and storage_data.get('value'):
                    token_obj = json.loads(storage_data['value'])
                    print(f"✓ Found auth token in localStorage: {storage_data['key']}")
                    return token_obj
                elif captured_token:
                    print("✓ Captured token from network request")
                    # Mock structure if we only have the token
                    return {
                        'access_token': captured_token,
                        'expires_at': int(time.time()) + 3600,
                        'token_type': 'bearer'
                    }
                else:
                    print("⚠️  No token found in localStorage or network requests")
                    print("The site may require manual login first.")
                    return None
                    
            except PlaywrightTimeout:
                print("⏱️  Timeout waiting for page to load")
                return None
            except Exception as e:
                print(f"❌ Error during browser automation: {e}")
                return None
            finally:
                await browser.close()
    
    async def get_valid_token(self, force_refresh: bool = False) -> Optional[Dict]:
        """Get a valid token, refreshing if necessary."""
        
        if not force_refresh:
            cached = self._load_cached_token()
            if cached:
                print("✓ Using cached token (still valid)")
                return cached
        
        print("🔄 Fetching fresh token from browser...")
        token_data = await self._capture_token_from_browser()
        
        if token_data:
            self._save_token_cache(token_data)
            self._update_env_file(token_data)
            
            expires_at = token_data.get('expires_at', 0)
            if expires_at:
                expires_in = int(expires_at - time.time())
                print(f"✓ Token valid for {expires_in // 60} minutes")
            
            return token_data
        
        return None
    
    async def ensure_authenticated(self) -> bool:
        """Ensure we have a valid token, fetch if needed."""
        token = await self.get_valid_token()
        return token is not None


async def main():
    """CLI entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Supabase Auth Agent")
    parser.add_argument("--refresh", action="store_true", 
                       help="Force refresh token even if cached")
    parser.add_argument("--check", action="store_true",
                       help="Check if cached token is still valid")
    
    args = parser.parse_args()
    
    agent = SupabaseAuthAgent()
    
    if args.check:
        cached = agent._load_cached_token()
        if cached:
            expires_at = cached.get('expires_at', 0)
            remaining = int(expires_at - time.time())
            print(f"✓ Cached token valid for {remaining // 60} more minutes")
            return
        else:
            print("⚠️  No valid cached token")
            return
    
    success = await agent.ensure_authenticated()
    
    if success:
        print("\n✅ Authentication successful!")
        print(f"   Credentials saved to: {agent.env_file}")
    else:
        print("\n❌ Authentication failed")
        print("\nTroubleshooting:")
        print("1. Make sure you've logged into painelfidc.com.br at least once")
        print("2. Check your internet connection")
        print("3. Try running with --refresh flag")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
