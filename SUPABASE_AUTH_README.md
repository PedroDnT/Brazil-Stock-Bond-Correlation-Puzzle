# Supabase Authentication Automation

Automated agent for managing Supabase authentication with `painelfidc.com.br`.

## Overview

The Supabase Auth Agent automatically:
- Launches a headless browser to capture authentication tokens
- Caches tokens locally (valid for ~1 hour)
- Auto-refreshes expired tokens
- Updates `.env` file with fresh credentials

## Setup

### 1. Install Dependencies

```bash
# Activate your virtual environment
source .venv/bin/activate

# Install required packages
pip install playwright supabase python-dotenv

# Install Chromium for Playwright
playwright install chromium
```

### 2. Initial Authentication

The agent captures tokens from the browser session. Make sure you've visited and logged into `https://www.painelfidc.com.br/dataset-fidc` at least once in your regular browser.

## Usage

### Quick Start

Run the agent to authenticate and fetch data:

```bash
python fetch_supabase.py
```

This will:
1. Check for a valid cached token
2. If expired, launch a headless browser to capture a new one
3. Save credentials to `.env`
4. Fetch sample data from the `dataset_fidc` table

### Manual Token Management

**Check token status:**
```bash
python supabase_auth_agent.py --check
```

**Force refresh token:**
```bash
python supabase_auth_agent.py --refresh
```

**Just authenticate (no data fetching):**
```bash
python supabase_auth_agent.py
```

## Files Created

- `.env` - Supabase credentials (gitignored)
- `.supabase_cache.json` - Token cache with expiry (gitignored)

## Token Lifecycle

- **Duration**: Tokens typically last 1 hour
- **Auto-refresh**: Agent checks expiry and refreshes automatically
- **Cache**: 5-minute buffer before expiry to ensure reliability

## Integration with Notebooks

To use in your Jupyter notebooks:

```python
import asyncio
from supabase_auth_agent import SupabaseAuthAgent
from supabase import create_client

# Ensure authenticated
agent = SupabaseAuthAgent()
await agent.ensure_authenticated()

# Create client
token_data = agent._load_cached_token()
supabase = create_client(
    agent.supabase_url,
    token_data['access_token']
)

# Fetch data
data = supabase.table("dataset_fidc").select("*").execute()
```

## Troubleshooting

**"No token found in localStorage"**
- Visit https://www.painelfidc.com.br/dataset-fidc in your browser first
- Make sure you're logged in
- Try running with `--refresh` flag

**Network errors in VS Code**
- The sandbox blocks external network calls
- Run scripts in a regular Terminal (outside VS Code)

**Playwright installation issues**
- Make sure you ran `playwright install chromium`
- Check that Chromium downloaded successfully: `playwright install --help`

**Token expired immediately**
- The site may have changed its authentication flow
- Check the browser's Developer Tools -> Application -> Local Storage for the key pattern `sb-*-auth-token`

## Security Notes

- `.env` and `.supabase_cache.json` are gitignored by default
- Tokens are stored locally and never transmitted except to the Supabase API
- Consider using environment-specific `.env` files for production deployments
