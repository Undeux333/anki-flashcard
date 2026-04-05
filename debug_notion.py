import os
import requests

NOTION_TOKEN       = os.environ.get("NOTION_TOKEN", "")
NOTION_DATABASE_ID = os.environ.get("NOTION_DATABASE_ID", "")
NOTION_VERSION     = "2022-06-28"

print(f"TOKEN先頭10文字: {NOTION_TOKEN[:10]}...")
print(f"DATABASE_ID: {NOTION_DATABASE_ID}")
print(f"DATABASE_ID文字数: {len(NOTION_DATABASE_ID)}")

headers = {
    "Authorization": f"Bearer {NOTION_TOKEN}",
    "Notion-Version": NOTION_VERSION,
}

# データベース情報を直接取得
url = f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}"
res = requests.get(url, headers=headers, timeout=15)
print(f"\nステータスコード: {res.status_code}")
print(f"レスポンス: {res.text[:500]}")
