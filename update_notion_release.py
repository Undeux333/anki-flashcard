#!/usr/bin/env python3
import json, os, sys
import gspread
from google.oauth2.service_account import Credentials

SPREADSHEET_ID    = os.environ.get("SPREADSHEET_ID", "")
GOOGLE_CREDS_JSON = os.environ.get("GOOGLE_CREDENTIALS", "")
DOWNLOAD_URL      = os.environ.get("DOWNLOAD_URL", "")
COL_RELEASE_URL   = 6  # F列

done_pages_path = "output/done_pages.json"
if not os.path.exists(done_pages_path):
    print("done_pages.json not found, skipping.")
    sys.exit(0)

rows = json.load(open(done_pages_path))

creds_info = json.loads(GOOGLE_CREDS_JSON)
scopes = ["https://www.googleapis.com/auth/spreadsheets"]
creds = Credentials.from_service_account_info(creds_info, scopes=scopes)
gc = gspread.authorize(creds)
sheet = gc.open_by_key(SPREADSHEET_ID).sheet1

for row in rows:
    try:
        sheet.update_cell(row, COL_RELEASE_URL, DOWNLOAD_URL)
    except Exception as e:
        print(f"Update failed (row {row}): {e}")

print(f"Updated {len(rows)} rows with Release URL: {DOWNLOAD_URL}")
