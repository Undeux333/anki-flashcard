#!/usr/bin/env python3
import os, json, hashlib, asyncio, tempfile, requests, io, re, time
from pathlib import Path
from datetime import datetime
from google import genai
from google.genai import types
import edge_tts
import genanki
from pydub import AudioSegment

# ═══════════════════════════════════════════════
#   設定・環境変数
# ═══════════════════════════════════════════════
GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY", "")
NOTION_TOKEN       = os.environ.get("NOTION_TOKEN", "")
NOTION_DATABASE_ID = os.environ.get("NOTION_DATABASE_ID", "")
GEMINI_MODEL       = "gemini-3.1-flash-lite-preview"
TTS_RATE           = "+0%"
NOTION_VERSION     = "2022-06-28"

PROP_PHRASE    = "Phrase"
PROP_STATUS    = "Status"
STATUS_DONE    = "Done"
STATUS_ERROR   = "Error"

ANKI_MODEL_ID = 1607392333
ANKI_DECK_ID  = 2059400110
FORCE_REGEN = os.environ.get("FORCE_REGEN", "false").lower() == "true"

CONV_VOICES = {
    "A": "en-US-BrianNeural",
    "B": "en-US-AvaMultilingualNeural",
}

# CSS（カードのデザイン）
CARD_CSS = """
.card { font-family: sans-serif; background: #f4f6f9; text-align: left; }
.ep-front, .ep-back { max-width: 550px; margin: auto; padding: 20px; }
.sentence-row { display: flex; align-items: flex-start; gap: 10px; margin-bottom: 20px; }
.splay-wrap { display: flex; flex-direction: column; gap: 5px; flex-shrink: 0; }
.splay { width: 55px; height: 32px; border-radius: 16px; background: #fff; display: flex; align-items: center; justify-content: center; cursor: pointer; border: 1px solid #e2e8f0; font-size: 10px; font-weight: bold; color: #4a5568; }
.sentence-content { flex: 1; min-width: 0; }
.sentence { font-size: 17px; padding: 12px; background: #fff; border-radius: 8px; color: #2d3748; line-height: 1.5; border: 1px solid #e2e8f0; }
.sentence b { color: #000; font-weight: bold; border-bottom: 2px solid #cbd5e0; }
.meaning-box { display: flex; align-items: flex-end; justify-content: space-between; margin-top: 8px; background: #ebf4ff; padding: 10px; border-radius: 8px; border-left: 4px solid #4299e1; }
.mini-meaning { font-size: 14px; color: #2c5282; line-height: 1.4; flex: 1; }
.mplay { font-size: 11px; color: #3182ce; cursor: pointer; font-weight: bold; padding: 4px 8px; background: #fff; border-radius: 12px; margin-left: 10px; flex-shrink: 0; white-space: nowrap; border: 1px solid #bee3f8; }
.rl { font-size: 10px; font-weight: bold; padding: 2px 8px; border-radius: 10px; }
.rn { background: #e8f4fd; color: #1d6fa4; }
.vbtn { display: flex; align-items: center; background: #fff; padding: 10px 15px; border-radius: 10px; cursor: pointer; border: 1px solid #e2e8f0; width: fit-content; font-size: 14px; }
.vp { width: 24px; height: 24px; border-radius: 50%; background: #ebf4ff; display: flex; align-items: center; justify-content: center; margin-right: 10px; font-size: 10px; }
"""

# ═══════════════════════════════════════════════
#   ロジック強化：話者ラベルベースの解析
# ═══════════════════════════════════════════════
def get_speech_lines(phrase):
    """
    「A:」「B:」の出現を基準にテキストを分割する。
    戻り値: [{'speaker': 'A', 'text': '...'}, {'speaker': 'B', 'text': '...'}]
    """
    # 正規表現で A: または B: を区切り文字として分割
    raw_parts = re.split(r'(A:|B:)', phrase)
    
    lines = []
    current_speaker = "A" # デフォルト
    
    for part in raw_parts:
        p = part.strip()
        if not p: continue
        
        if p == "A:":
            current_speaker = "A"
        elif p == "B:":
            current_speaker = "B"
        else:
            # ラベルの直後に来るのがその話者のテキスト
            lines.append({"speaker": current_speaker, "text": p})
            
    return lines

def generate_content(client, speech_lines: list) -> dict:
    # ラベル（A: B:）の合計個数 ＝ 解説の必要数
    label_count = len(speech_lines)
    input_text = "\n".join([f"{l['speaker']}: {l['text']}" for l in speech_lines])
    
    # プロンプトで個数を厳格に指定
    prompt = f"""Explain the nuance of EACH phrase marked with A: or B: in the following dialogue. 
Explain how this phrase is used in everyday conversation by native speakers. Focus on the speaker’s feeling and intention. Keep it short, simple, and natural. Use plain, everyday English. Limit the explanation to one sentence. Avoid abstract wording or extra details. Capture the full meaning of the phrase naturally. Write as if explaining to an English learner in a casual conversation.
CRITICAL RULE: 
The input has exactly {label_count} labeled phrases. 
You MUST provide exactly {label_count} explanations in the "meanings" array. 
One explanation per phrase, in the same order.

Input:
{input_text}

Return ONLY valid JSON: 
{{
  "meanings": ["explanation for phrase 1", "explanation for phrase 2", ...]
}}"""

    for attempt in range(5):
        try:
            response = client.models.generate_content(
                model=GEMINI_MODEL, 
                contents=prompt, 
                config=types.GenerateContentConfig(response_mime_type="application/json", temperature=0.2)
            )
            data = json.loads(response.text.strip())
            
            # 念のための整合性チェック（足りなければ補完）
            while len(data["meanings"]) < label_count:
                data["meanings"].append("(Check original text for nuance)")
            # 多すぎればカット
            data["meanings"] = data["meanings"][:label_count]
            
            return data
        except Exception as e:
            if "429" in str(e):
                time.sleep(25)
                continue
            raise e

# ═══════════════════════════════════════════════
#   音声生成・ビルド（以下、前回の安定版を継承）
# ═══════════════════════════════════════════════
async def _tts_bytes(text, voice):
    communicate = edge_tts.Communicate(text, voice=voice, rate=TTS_RATE)
    data = b""
    async for chunk in communicate.stream():
        if chunk["type"] == "audio": data += chunk["data"]
    return data

async def process_audio(speech_lines: list, meanings: list, uid: str, tmpdir: str):
    s_files, m_files = [], []
    combined = AudioSegment.empty()
    
    for idx, line in enumerate(speech_lines):
        # 1. 発言音声
        clean_text = re.sub(r'\(|\)', '', line['text'])
        s_data = await _tts_bytes(clean_text, CONV_VOICES.get(line['speaker'], "en-US-BrianNeural"))
        s_fn = f"s_{uid}_{idx}.mp3"
        (Path(tmpdir) / s_fn).write_bytes(s_data)
        s_files.append(s_fn)
        combined += AudioSegment.from_file(io.BytesIO(s_data), format="mp3") + AudioSegment.silent(duration=500)
        
        # 2. 解説音声
        m_text = meanings[idx]
        m_data = await _tts_bytes(m_text, CONV_VOICES["B"])
        m_fn = f"m_{uid}_{idx}.mp3"
        (Path(tmpdir) / m_fn).write_bytes(m_data)
        m_files.append(m_fn)

    f_fn = f"full_{uid}.mp3"
    combined.export(str(Path(tmpdir) / f_fn), format="mp3")
    return f_fn, s_files, m_files

def format_script_text(text: str) -> str:
    t = text.replace("<", "&lt;").replace(">", "&gt;")
    return re.sub(r'\((.*?)\)', r'<b>\1</b>', t).replace("\n", "<br>")

def build_front(f_fn):
    return f'<div class="ep-front">[sound:{f_fn}]<div class="sec-label">Listen</div><div class="vbtn" onclick="document.getElementById(\'fa1\').play()"><div class="vp">▶</div><div>Play Full</div><audio id="fa1" src="{f_fn}"></audio></div></div>'

def build_back(speech_lines, s_files, m_files, meanings):
    combined_html = ""
    for idx, line in enumerate(speech_lines):
        disp = format_script_text(f"{line['speaker']}: {line['text']}")
        mt = meanings[idx]
        s_fn = s_files[idx]
        m_fn = m_files[idx]
        
        combined_html += f"""
        <div class="sentence-row">
            <div class="splay-wrap">
                <div class="splay" onclick="document.getElementById('s{idx}').play()">▶ Voice</div>
                <audio id="s{idx}" src="{s_fn}"></audio>
            </div>
            <div class="sentence-content">
                <div class="sentence">{disp}</div>
                <div class="meaning-box">
                    <div class="mini-meaning"><i>{mt}</i></div>
                    <div class="mplay" onclick="document.getElementById('m{idx}').play()">🔊 Explain</div>
                    <audio id="m{idx}" src="{m_fn}"></audio>
                </div>
            </div>
        </div>"""
    return f'<div class="ep-back"><div style="margin-bottom:10px"><span class="rl rn">● Script & Nuance</span></div>{combined_html}</div>'

# ═══════════════════════════════════════════════
#   Main
# ═══════════════════════════════════════════════
def main():
    current_dir = Path(__file__).parent.absolute()
    output_path = current_dir / "output"
    output_path.mkdir(parents=True, exist_ok=True)
    client = genai.Client(api_key=GEMINI_API_KEY)
    
    # Notionから未処理データ取得
    url = f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query"
    payload = {} if FORCE_REGEN else {"filter": {"or": [{"property": PROP_STATUS, "select": {"is_empty": True}}, {"property": PROP_STATUS, "select": {"equals": STATUS_ERROR}}]}}
    res = requests.post(url, headers={"Authorization": f"Bearer {NOTION_TOKEN}", "Notion-Version": NOTION_VERSION, "Content-Type": "application/json"}, json=payload, timeout=15)
    res.raise_for_status()
    
    pending = []
    for page in res.json().get("results", []):
        try:
            phrase = page["properties"][PROP_PHRASE]["title"][0]["text"]["content"].strip()
            if phrase: pending.append({"phrase": phrase, "page_id": page["id"]})
        except: continue

    if not pending: return

    model = genanki.Model(ANKI_MODEL_ID, "EP_Model_v14", fields=[{"name": "Front"}, {"name": "Back"}], templates=[{"name": "Card 1", "qfmt": "{{Front}}", "afmt": "{{Back}}"}], css=CARD_CSS)
    deck = genanki.Deck(ANKI_DECK_ID, "English Phrases (Auto)")
    all_media = []

    with tempfile.TemporaryDirectory() as tmpdir:
        for i, item in enumerate(pending, 1):
            phrase = item["phrase"]
            print(f"[{i}/{len(pending)}] {phrase[:40]}...")
            uid = hashlib.md5(phrase.encode()).hexdigest()[:10]
            try:
                if i > 1: time.sleep(12)
                
                # 1. 話者ラベルの個数ベースで分割
                speech_lines = get_speech_lines(phrase)
                # 2. コンテンツ生成（個数を厳守させる）
                content = generate_content(client, speech_lines)
                meanings = content["meanings"]
                
                # 3. 音声生成（各文＋各解説）
                f_fn, s_files, m_files = asyncio.run(process_audio(speech_lines, meanings, uid, tmpdir))
                
                all_media.extend([str(Path(tmpdir) / f) for f in [f_fn] + s_files + m_files])
                deck.add_note(genanki.Note(model=model, fields=[build_front(f_fn), build_back(speech_lines, s_files, m_files, meanings)], guid=uid))
                
                # Notion更新
                requests.patch(f"https://api.notion.com/v1/pages/{item['page_id']}", headers={"Authorization": f"Bearer {NOTION_TOKEN}", "Notion-Version": NOTION_VERSION, "Content-Type": "application/json"}, json={"properties": {PROP_STATUS: {"select": {"name": STATUS_DONE}}}}, timeout=10)
                print("  ✅ 成功")
            except Exception as e:
                print(f"  ❌ 失敗: {e}")
                requests.patch(f"https://api.notion.com/v1/pages/{item['page_id']}", headers={"Authorization": f"Bearer {NOTION_TOKEN}", "Notion-Version": NOTION_VERSION, "Content-Type": "application/json"}, json={"properties": {PROP_STATUS: {"select": {"name": STATUS_ERROR}}}}, timeout=10)

        if all_media:
            pkg = genanki.Package(deck)
            pkg.media_files = all_media
            final_name = output_path / f"deck_{datetime.now().strftime('%m%d%H%M')}.apkg"
            pkg.write_to_file(str(final_name))
            print(f"📦 生成完了: {final_name}")

if __name__ == "__main__":
    main()
