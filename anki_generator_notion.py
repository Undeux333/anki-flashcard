# ═══════════════════════════════════════════════
#   音声生成 (発言＋解説の両方を生成)
# ═══════════════════════════════════════════════
async def process_audio(phrase: str, meanings: list, uid: str, tmpdir: str):
    """
    1. 各センテンスの音声
    2. 各解説(Meaning)の音声
    3. 表面用の全体結合音声
    を生成します。
    """
    parts = re.split(r'(A:|B:)', phrase)
    sentence_audio_files = []
    meaning_audio_files = []
    
    # 結合用
    combined_audio = AudioSegment.empty()
    
    if len(parts) > 1:
        # 会話形式
        current_speaker = "A"
        m_idx = 0
        for part in parts:
            clean = part.strip()
            if clean == "A:": current_speaker = "A"; continue
            if clean == "B:": current_speaker = "B"; continue
            if not clean: continue
            
            # --- 1. 発言の音声 ---
            speech_text = re.sub(r'\(|\)', '', clean)
            voice = CONV_VOICES[current_speaker]
            s_data = await _tts_save_bytes(speech_text, voice, TTS_RATE)
            s_fname = f"ep_{uid}_s{m_idx}.mp3"
            (Path(tmpdir) / s_fname).write_bytes(s_data)
            sentence_audio_files.append(s_fname)
            
            # 表面結合用に追加
            segment = AudioSegment.from_file(io.BytesIO(s_data), format="mp3")
            combined_audio += segment + AudioSegment.silent(duration=500)
            
            # --- 2. 解説(Meaning)の音声 ---
            if m_idx < len(meanings):
                m_text = meanings[m_idx]
                # 解説は落ち着いた女性の声(Ava)で固定
                m_data = await _tts_save_bytes(m_text, CONV_VOICES["B"], TTS_RATE)
                m_fname = f"ep_{uid}_m{m_idx}.mp3"
                (Path(tmpdir) / m_fname).write_bytes(m_data)
                meaning_audio_files.append(m_fname)
            
            m_idx += 1
    else:
        # 通常形式
        speech_text = re.sub(r'\(|\)', '', phrase)
        s_data = await _tts_save_bytes(speech_text, CONV_VOICES["A"], TTS_RATE)
        s_fname = f"ep_{uid}_s0.mp3"
        (Path(tmpdir) / s_fname).write_bytes(s_data)
        sentence_audio_files.append(s_fname)
        
        combined_audio += AudioSegment.from_file(io.BytesIO(s_data), format="mp3")
        
        if meanings:
            m_data = await _tts_save_bytes(meanings[0], CONV_VOICES["B"], TTS_RATE)
            m_fname = f"ep_{uid}_m0.mp3"
            (Path(tmpdir) / m_fname).write_bytes(m_data)
            meaning_audio_files.append(m_fname)

    # 表面用フル音声
    full_fname = f"ep_{uid}_full.mp3"
    combined_audio.export(str(Path(tmpdir) / full_fname), format="mp3")
    
    return full_fname, sentence_audio_files, meaning_audio_files

# ═══════════════════════════════════════════════
#   裏面のビルド (音声ボタンを2種類配置)
# ═══════════════════════════════════════════════
def build_back(full_fname, s_files, m_files, content):
    phrase_raw = content["phrase_display"]
    meanings = content.get("meanings", [])
    lines = re.split(r'(A:|B:)', phrase_raw)
    
    combined_html = ""
    idx = 0
    
    if len(lines) > 1:
        current_speaker = ""
        for part in lines:
            clean = part.strip()
            if clean in ["A:", "B:"]:
                current_speaker = clean
                continue
            if not clean: continue
            
            display_line = format_script_text(f"{current_speaker} {clean}")
            m_text = meanings[idx] if idx < len(meanings) else ""
            s_fn = s_files[idx] if idx < len(s_files) else ""
            m_fn = m_files[idx] if idx < len(m_files) else ""
            
            combined_html += f"""
            <div class="sentence-row">
                <div class="splay-wrap">
                    <div class="splay" onclick="document.getElementById('s{idx}').play()">▶ Voice</div>
                    <audio id="s{idx}" src="{s_fn}"></audio>
                </div>
                <div class="sentence-content">
                    <div class="sentence">{display_line}</div>
                    <div class="meaning-box">
                        <div class="mini-meaning"><i>{m_text}</i></div>
                        <div class="mplay" onclick="document.getElementById('m{idx}').play()">🔊 Explain</div>
                        <audio id="m{idx}" src="{m_fn}"></audio>
                    </div>
                </div>
            </div>"""
            idx += 1
    else:
        # (通常形式も同様に構築)
        # ...簡略化のため中略...
        pass

    return f"""<div class="ep-back">
<div style="margin-bottom:10px"><span class="rl rn">● Script & Nuance</span></div>
{combined_html}
</div>"""

# CSSに音声ボタンのスタイルを追加
CARD_CSS += """
.splay-wrap { display: flex; flex-direction: column; gap: 5px; }
.splay { width: 50px; height: 30px; border-radius: 15px; background: #fff; display: flex; align-items: center; justify-content: center; cursor: pointer; border: 1px solid #e2e8f0; font-size: 10px; font-weight: bold; color: #4a5568; }
.meaning-box { display: flex; align-items: flex-end; justify-content: space-between; margin-top: 5px; background: #f8fafc; padding: 5px 8px; border-radius: 5px; }
.mplay { font-size: 10px; color: #3182ce; cursor: pointer; font-weight: bold; padding: 2px 5px; }
"""
