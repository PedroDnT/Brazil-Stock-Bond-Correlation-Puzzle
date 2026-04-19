import asyncio
from playwright.async_api import async_playwright
from playwright_stealth import stealth_async
import json
import os

async def get_supabase_credentials():
    async with async_playwright() as p:
        # Launch browser - headless=False to allow potential manual intervention if site has anti-bot
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/145.0.0.0 Safari/537.36"
        )
        page = await context.new_page()
        await stealth_async(page)

        print("Navigating to painelfidc.com.br/dataset-fidc...")
        
        # We need to catch the network request that contains the apikey
        captured_data = {"apikey": None, "authorization": None}
        
        async def handle_request(request):
            if "supabase.co" in request.url:
                headers = request.headers
                if "apikey" in headers and not captured_data["apikey"]:
                    captured_data["apikey"] = headers["apikey"]
                if "authorization" in headers and not captured_data["authorization"]:
                    captured_data["authorization"] = headers["authorization"]

        page.on("request", handle_request)

        try:
            # Go to the site - this should trigger the Supabase calls
            await page.goto("https://www.painelfidc.com.br/dataset-fidc", wait_until="networkidle", timeout=60000)
            
            # Wait a bit more for background calls
            await asyncio.sleep(5)
            
            if captured_data["apikey"]:
                print("Successfully captured credentials!")
                # Save to .env or return
                with open(".env", "w") as f:
                    f.write(f"SUPABASE_URL=https://ulxfhbyvbjsivbpcmyim.supabase.co\n")
                    f.write(f"SUPABASE_KEY={captured_data['apikey']}\n")
                    f.write(f"SUPABASE_AUTH_TOKEN={captured_data['authorization']}\n")
                return captured_data
            else:
                print("Failed to capture apikey automatically. Trying to extract from LocalStorage...")
                storage = await page.evaluate("() => JSON.stringify(localStorage)")
                storage_json = json.loads(storage)
                # Look for supabase tokens in storage
                for key, value in storage_json.items():
                    if "supabase" in key and "auth-token" in key:
                        token_data = json.loads(value)
                        captured_data["authorization"] = f"Bearer {token_data.get('access_token')}"
                        captured_data["apikey"] = token_data.get('access_token') # Often the same for anon
                
                if captured_data["authorization"]:
                     with open(".env", "w") as f:
                        f.write(f"SUPABASE_URL=https://ulxfhbyvbjsivbpcmyim.supabase.co\n")
                        f.write(f"SUPABASE_KEY={captured_data['apikey']}\n")
                        f.write(f"SUPABASE_AUTH_TOKEN={captured_data['authorization']}\n")
                     return captured_data

        except Exception as e:
            print(f"Error: {e}")
        finally:
            await browser.close()

if __name__ == "__main__":
    asyncio.run(get_supabase_credentials())
