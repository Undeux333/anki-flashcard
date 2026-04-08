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
STATUS_DONE, STATUS_ERROR = "Done", "Error"

ANKI_MODEL_ID = 1607392333
ANKI_DECK_ID  = 2059400110
FORCE_REGEN = os.environ.get("FORCE_REGEN", "false").lower() == "true"

# パターン定義
VOICE_CONFIG = {
    "p1": {"name": "Brian & Ava",  "A": "en-US-BrianNeural", "B": "en-US-AvaMultilingualNeural", "M": "en-US-BrianNeural"},
    "p2": {"name": "Emma & Andrew", "A": "en-US-EmmaNeural",  "B": "en-US-AndrewNeural",       "M": "en-US-EmmaNeural"},
    "p3": {"name": "Jason & Eric",  "A": "en-US-JasonNeural", "B": "en-US-EricNeural",         "M": "en-US-JasonNeural"},
}

CARD_CSS = """
.card { font-family: -apple-system, sans-serif; background: #fdfdfd; text-align: left; color: #333; }
.ep-front, .ep-back { max-width: 600px; margin: auto; padding: 25px 15px; }
.sec-label { font-size: 11px; color: #888; font-weight: bold; text-transform: uppercase; margin-bottom: 12px; display: block; }
.voice-selector { display: flex; gap: 12px; flex-wrap: wrap; justify-content: center; }
.vbtn-full { display: flex; align-items: center; gap: 12px; background: #fff; padding: 10px 18px; border-radius: 12px; cursor: pointer; border: 1px solid #e2e8f0; font-size: 13px; box-shadow: 0 2px 4px rgba(0,0,0,0.04); min-width: 160px; }
.vp-icon { width: 24px; height: 24px; border-radius: 50%; background: #ebf8ff; display: flex; align-items: center; justify-content: center; color: #3182ce; font-size: 10px; }
.sentence-row { margin-bottom: 25px; border-bottom: 1px solid #f0f0f0; padding-bottom: 20px; }
.sentence-content { background: #fff; padding: 18px; border-radius: 12px; border: 1px solid #eaeaea; line-height: 1.6; }
.sentence { font-size: 18px; color: #1a1a1a; margin-bottom: 12px; }
.sentence b { color: #000; border-bottom: 2px solid #3182ce; }
.meaning-box { background: #f4f9ff; padding: 14px; border-radius: 8px; border-left: 4px solid #3182ce; font-size: 15px; color: #2c5282; margin-bottom: 15px; }
.play-controls { display: flex; align-items: center; gap: 10px; border-top: 1px solid #f0f0f0; padding-top: 12px; }
.p-btn { padding: 6px 12px; border-radius: 8px; background: #f7fafc; cursor: pointer; font-size: 11px; font-weight: bold; border: 1px solid #edf2f7; color: #4a5568; }
.p-btn.playing { background: #3182ce; color: #fff; }
"""

# ═══════════════════════════════════════════════
#   ロジック
# ═══════════════════════════════════════════════
def get_speech_lines(phrase):
    # A: または B: が含まれているか判定
    has_labels = bool(re.search(r'\b[AB]:', phrase))
    
    if not has_labels:
        # ラベルがない場合は全文を一人(A)のセリフとする
        return [{"speaker": "A", "text": phrase.strip()}]
    
    # ラベルがある場合は分割
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

def generate_content(client, speech_lines):
    count = len(speech_lines)
    input_text = "\n".join([f"{l['speaker']}: {l['text']}" for l in speech_lines])
    prompt = f'Explain the nuance of EACH phrase. Keep it short, simple, natural English. Return JSON {{"meanings": []}} with {count} items.\nInput:\n{input_text}'
    for _ in range(3):
        try:
            res = client.models.generate_content(model=GEMINI_MODEL, contents=prompt, config=types.GenerateContentConfig(response_mime_type="application/json"))
            data = json.loads(res.text.strip())
            m = data.get("meanings", [])
            if not isinstance(m, list): m = [str(m)]
            while len(m) < count: m.append("...")
            return [str(i) for i in m[:count]]
        except: time.sleep(5)
    return ["No explanation available"] * count

async def _tts_bytes(text, voice):
    # クリーニングと安全な文字列変換
    clean = re.sub(r'[^a-zA-Z0-9\s,.\'?!]', '', str(text)).strip()
    if not clean: return AudioSegment.silent(duration=100).raw_data 
    try:
        comm = edge_tts.Communicate(clean, voice=voice)
        data = b""
        async for chunk in comm.stream():
            if chunk["type"] == "audio": data += chunk["data"]
        return data
    except: return AudioSegment.silent(duration=100).raw_data

async def process_audio_multi(lines, meanings, uid, tmpdir):
    full_files, split_files = {}, {}
    for p_id, voices in VOICE_CONFIG.items():
        combined = AudioSegment.empty()
        p_splits = []
        for idx, line in enumerate(lines):
            v = voices.get(line['speaker'], voices["A"])
            s_data = await _tts_bytes(line['text'], v)
            s_fn = f"s_{uid}_{p_id}_{idx}.mp3"
            (Path(tmpdir) / s_fn).write_bytes(s_data)
            
            try:
                combined += AudioSegment.from_file(io.BytesIO(s_data), format="mp3") + AudioSegment.silent(duration=600)
            except: pass

            m_data = await _tts_bytes(meanings[idx], voices["M"])
            m_fn = f"m_{uid}_{p_id}_{idx}.mp3"
            (Path(tmpdir) / m_fn).write_bytes(m_data)
            p_splits.append({"s": s_fn, "m": m_fn})
            
        f_fn = f"full_{uid}_{p_id}.mp3"
        combined.export(str(Path(tmpdir) / f_fn), format="mp3")
        full_files[p_id] = f_fn
        split_files[p_id] = p_splits
    return full_files, split_files

def get_display_name(p_id, is_conv):
    if is_conv:
        return VOICE_CONFIG[p_id]["name"]
    # 一人の場合: A役の声の末尾の名前を取得
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
    out = Path(__file__).parent / "output"
    out.mkdir(exist_ok=True)
    client = genai.Client(api_key=GEMINI_API_KEY)
    headers = {"Authorization": f"Bearer {NOTION_TOKEN}", "Notion-Version": "2022-06-28", "Content-Type": "application/json"}
    
    res = requests.post(f"https://api.notion.com/v1/databases/{NOTION_DATABASE_ID}/query", headers=headers, json={} if FORCE_REGEN else {"filter": {"property": PROP_STATUS, "select": {"is_empty": True}}})
    res.raise_for_status()
    
    pending = []
    for p in res.json().get("results", []):
        try:
            title_prop = p["properties"][PROP_PHRASE].get("title", [])
            if title_prop:
                text = str(title_prop[0]["text"]["content"]).strip()
                if text: pending.append({"p": text, "id": p["id"]})
        except: continue

    if not pending: return

    model = genanki.Model(ANKI_MODEL_ID, "EP_Model_v18", fields=[{"name": "F"},{"name": "B"}], templates=[{"name": "C1", "qfmt": "{{F}}", "afmt": "{{B}}"}], css=CARD_CSS)
    deck = genanki.Deck(ANKI_DECK_ID, "English Multi-Voice")
    media = []

    with tempfile.TemporaryDirectory() as tmp:
        for i, item in enumerate(pending):
            uid = hashlib.md5(item["p"].encode()).hexdigest()[:8]
            try:
                lines = get_speech_lines(item["p"])
                # 会話判定: A: B: ラベルがあるか、または2行以上の発言があるか
                is_conv = any(":" in item["p"] for _ in [0]) and (any(l["speaker"] == "B" for l in lines) or len(lines) > 1)
                
                meanings = generate_content(client, lines)
                ff, sf = asyncio.run(process_audio_multi(lines, meanings, uid, tmp))
                
                for f in ff.values(): media.append(str(Path(tmp)/f))
                for p in sf.values(): 
                    for x in p: media.append(str(Path(tmp)/x["s"])); media.append(str(Path(tmp)/x["m"]))
                
                deck.add_note(genanki.Note(model=model, fields=[build_front(ff, is_conv), build_back(lines, meanings, sf, is_conv)], guid=uid))
                requests.patch(f"https://api.notion.com/v1/pages/{item['id']}", headers=headers, json={"properties": {PROP_STATUS: {"select": {"name": STATUS_DONE}}}})
                print(f"✅ {uid}")
            except Exception as e: 
                print(f"❌ Error at {uid}: {e}")
                requests.patch(f"https://api.notion.com/v1/pages/{item['id']}", headers=headers, json={"properties": {PROP_STATUS: {"select": {"name": STATUS_ERROR}}}})

        if media:
            pkg = genanki.Package(deck)
            pkg.media_files = list(set(media))
            pkg.write_to_file(out / f"deck_{datetime.now().strftime('%m%d%H%M')}.apkg")

if __name__ == "__main__": main()
