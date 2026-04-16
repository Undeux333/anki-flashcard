#!/usr/bin/env python3
import json, requests, os, sys

page_ids_path = "output/done_pages.json"
if not os.path.exists(page_ids_path):
    print("done_pages.json not found, skipping.")
    sys.exit(0)

page_ids     = json.load(open(page_ids_path))
notion_token = os.environ["NOTION_TOKEN"]
download_url = os.environ["DOWNLOAD_URL"]

headers = {
    "Authorization": f"Bearer {notion_token}",
    "Notion-Version": "2022-06-28",
    "Content-Type": "application/json"
}

for pid in page_ids:
    try:
        res = requests.patch(
            f"https://api.notion.com/v1/pages/{pid}",
            headers=headers,
            json={"properties": {"Release URL": {"url": download_url}}},
            timeout=10
        )
        res.raise_for_status()
    except Exception as e:
        print(f"Notion update failed ({pid}): {e}")

print(f"Updated {len(page_ids)} pages with Release URL: {download_url}")
