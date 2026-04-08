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
#   設定
# ═══════════════════════════════════════════════
GEMINI_API_KEY     = os.environ.get("GEMINI_API_KEY", "")
NOTION_TOKEN       = os.environ.get("NOTION_TOKEN", "")
NOTION_DATABASE_ID = os.environ.get("NOTION_DATABASE_ID", "")
GEMINI_MODEL       = "gemini-3.1-flash-lite-preview"

PROP_PHRASE, PROP_STATUS = "Phrase", "Status"
STATUS_DONE    = "Done"
STATUS_ERROR   = "Error"
STATUS_TIMEOUT = "Timeout"

ANKI_MODEL_ID = 1607392333
ANKI_DECK_ID  = 2059400110
FORCE_REGEN = os.environ.get("FORCE_REGEN", "false").lower() == "true"

VOICE_CONFIG = {
    "p1": {"name": "Brian & Ava",  "A": "en-US-BrianNeural", "B": "en-US-AvaMultilingualNeural", "M": "en-US-BrianNeural"},
    "p2": {"name": "Emma & Andrew", "A": "en-US-EmmaNeural",  "B": "en-US-AndrewNeural",       "M": "en-US-EmmaNeural"},
    "p3": {"name": "Jason & Eric",  "A": "en-US-JasonNeural", "B": "en-US-EricNeural",         "M": "en-US-JasonNeural"},
}

# --- ロジック ---

def extract_notion_text(property_data):
    """
    🆕 Notionのrich_text/title配列から全テキストを安全に抽出・結合する
    """
    try:
        parts = property_data.get("title", []) or property_data.get("rich_text", [])
        # 複数のリッチテキスト要素（太字やリンク等）をすべて結合
        text = "".join([part.get("plain_text", "") for part in parts])
        return text.strip()
    except:
        return ""

def get_speech_lines(phrase):
    has_labels = bool(re.search(r'\b[AB]:', phrase))
    if not has_labels:
        return [{"speaker": "A", "text": phrase.strip()}]
    raw_parts = re.split(r'([AB]:)', phrase)
    lines = []
    curr = "A"
    for p in raw_parts:
        s = p.strip()
        if not s: continue
        if s == "A:": curr = "A"
        elif s == "B:": curr = "B"
        else: lines.append({"speaker": curr, "text": s})
    return lines

async def _tts_bytes(text, voice):
    # デバッグ用に読み上げ対象を表示
    clean = str(text).strip()
    if not clean:
        return AudioSegment.silent(duration=100).raw_data

    try:
        comm = edge_tts.Communicate(clean, voice=voice)
        data = b""
        async for chunk in comm.stream():
            if chunk["type"] == "audio": data += chunk["data"]
        
        if not data:
            return AudioSegment.silent(duration=100).raw_data
        return data
    except Exception as e:
        if "No audio was received" in str(e):
            return AudioSegment.silent(duration=100).raw_data
        raise e

async def process_audio_multi(lines, meanings, uid, tmpdir):
    full_files, split_files = {}, {}
    for p_id, voices in VOICE_CONFIG.items():
        combined = AudioSegment.empty()
        p_splits = []
        for idx, line in enumerate(lines):
            v = voices.get(line['speaker'], voices["A"])
            # セリフ生成
            s_data = await _tts_bytes(line['text'], v)
            s_fn = f"s_{uid}_{p_id}_{idx}.mp3"
            (Path(tmpdir) / s_fn).write_bytes(s_data)
            
            try:
                combined += AudioSegment.from_file(io.BytesIO(s_data), format="mp3") + AudioSegment.silent(duration=600)
            except:
                combined += AudioSegment.silent(duration=500)

            # 解説生成
            m_data = await _tts_bytes(meanings[idx], voices["M"])
            m_fn = f"m_{uid}_{p_id}_{idx}.mp3"
            (Path(tmpdir) / m_fn).write_bytes(m_data)
            p_splits.append({"s": s_fn, "m": m_fn})
            
        f_fn = f"full_{uid}_{p_id}.mp3"
        combined.export(str(Path(tmpdir) / f_fn), format="mp3")
        full_files[p_id] = f_fn
        split_files[p_id] = p_splits
    return full_files, split_files

# --- Anki UI定義は前回と同様 ---
def get_display_name(p_id, is_conv):
    if is_conv: return VOICE_CONFIG[p_id]["name"]
    return VOICE_CONFIG[p_id]["A"].split('-')[-1].replace('Neural', '')

def build_front(full_files, is_conv):
    btns, tags = "", ""
    for p_id, fn in full_files.items():
        label = get_display_name(p_id, is_conv)
        btns += f'<div class="vbtn-full" onclick="playA(\'fa_{p_id}\')"><div class="vp-icon">▶</div><div><strong>{label}</strong></div></div>'
        tags += f'<audio id="fa_{p_id}" src="{fn}"></audio>'
    return f'<div class="ep-front">[sound:{full_files["p1"]}]<span class="sec-label">Select Pattern</span><div class="voice-selector">{btns}</div>{tags}<script>function playA(id){{var a=document.getElementsByTagName("audio");for(var i=0;i<a.length;i++){{a[i].pause();a[i].currentTime=0;}}document.getElementById(id).play();}}</script></div>'

def build_back(lines, meanings, splits, is_conv):
    rows = ""
    for idx, line in enumerate(lines):
        txt = line['text'].replace("(", "<b>").replace(")", "</b>")
        prefix = f"{line['speaker']}: " if is_conv else ""
        ctrls, tags = "", ""
        for p_id in VOICE_CONFIG.keys():
            label = get_display_name(p_id, is_conv)
            ctrls += f'<div id="b_{p_id}_{idx}" class="p-btn" onclick="pc(\'{p_id}\',{idx})">{label}</div>'
            tags += f'<audio id="s_{p_id}_{idx}" src="{splits[p_id][idx]["s"]}" onended="document.getElementById(\'m_{p_id}_{idx}\').play()"></audio><audio id="m_{p_id}_{idx}" src="{splits[p_id][idx]["m"]}" onended="document.getElementById(\'b_{p_id}_{idx}\').classList.remove(\'playing\')"></audio>'
        rows += f'<div class="sentence-row"><div class="sentence-content"><div class="sentence">{prefix}{txt}</div><div class="meaning-box"><i>{meanings[idx]}</i></div><div class="play-controls">{ctrls}</div>{tags}</div></div>'
    return f'<div class="ep-back"><span class="sec-label">Script & Nuance</span>{rows}<script>function pc(p,i){{sa();document.getElementById("b_"+p+"_"+i).classList.add("playing");document.getElementById("s_"+p+"_"+i).play();}}function sa(){{var a=document.getElementsByTagName("audio");for(var i=0;i<a.length;i++){{a[i].pause();a[i].currentTime=0;}}var b=document.getElementsByClassName("p-btn");for(var i=0;i<b.length;i++){{b[i].classList.remove("playing");}}}}</script></div>'

def main():
    out = Path(__file__).parent / "output"; out.mkdir(exist_ok=True)
    client = genai.Client(api_key=GEMINI_API_KEY)
    headers = {"Authorization": f"Bearer {NOTION_TOKEN}", "Notion-Version": "2022-06-28", "Content-Type": "application/json"}
    
    filter_payload = {}
    if not FORCE_REGEN:
        filter_payload = {"filter": {"or": [{"property": PROP_STATUS, "select": {"is_empty": True}}, {"property": PROP_STATUS, "select": {"equals": STATUS_TIMEOUT}}]}}

    res = requests.post(f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query", headers=headers, json=filter_payload)
    res.raise_for_status()
    
    pending = []
    for p in res.json().get("results", []):
        # 🆕 改良されたテキスト抽出
        phrase_text = extract_notion_text(p["properties"].get(PROP_PHRASE, {}))
        if phrase_text:
            pending.append({"p": phrase_text, "id": p["id"]})
        else:
            print(f"⚠️ スキップ: ID {p['id']} のテキストが空です")

    if not pending: return

    # CARD_CSS等は既存のものを利用
    model = genanki.Model(ANKI_MODEL_ID, "EP_Model_v19", fields=[{"name": "F"},{"name": "B"}], templates=[{"name": "C1", "qfmt": "{{F}}", "afmt": "{{B}}"}], css=CARD_CSS)
    deck = genanki.Deck(ANKI_DECK_ID, "English Multi-Voice")
    media = []

    with tempfile.TemporaryDirectory() as tmp:
        for item in pending:
            uid = hashlib.md5(item["p"].encode()).hexdigest()[:8]
            print(f"🎙 処理中: [{item['p'][:20]}...]")
            try:
                lines = get_speech_lines(item["p"])
                is_conv = any(":" in item["p"] for _ in [0]) and (any(l["speaker"] == "B" for l in lines) or len(lines) > 1)
                
                # Gemini生成
                meanings = generate_content(client, lines)
                
                # 音声生成
                ff, sf = asyncio.run(process_audio_multi(lines, meanings, uid, tmp))
                
                for f in ff.values(): media.append(str(Path(tmp)/f))
                for p in sf.values(): 
                    for x in p: media.append(str(Path(tmp)/x["s"])); media.append(str(Path(tmp)/x["m"]))
                
                deck.add_note(genanki.Note(model=model, fields=[build_front(ff, is_conv), build_back(lines, meanings, sf, is_conv)], guid=uid))
                requests.patch(f"https://api.notion.com/v1/pages/{item['id']}", headers=headers, json={"properties": {PROP_STATUS: {"select": {"name": STATUS_DONE}}}})
                print(f"  ✅ 成功: {uid}")

            except Exception as e:
                err_msg = str(e).lower()
                if any(x in err_msg for x in ["timeout", "deadline", "429", "503", "no audio"]):
                    new_status = STATUS_TIMEOUT
                    print(f"  ⏳ タイムアウト/再試行対象: {e}")
                else:
                    new_status = STATUS_ERROR
                    print(f"  ❌ エラー: {e}")
                requests.patch(f"https://api.notion.com/v1/pages/{item['id']}", headers=headers, json={"properties": {PROP_STATUS: {"select": {"name": new_status}}}})

        if media:
            pkg = genanki.Package(deck)
            pkg.media_files = list(set(media))
            pkg.write_to_file(out / f"deck_{datetime.now().strftime('%m%d%H%M')}.apkg")

if __name__ == "__main__": main()
