"""
app.py  —  Multilingual Voice Translator (Web App)

Run locally:
    python app.py

Deploy to Hugging Face Spaces:
    Push repo, set SDK to Gradio, hardware CPU-Basic (free)

Deploy to Render / Railway:
    Start command: python app.py
    Port: 7860
"""

import os
import time
import tempfile
import threading
import numpy as np
import gradio as gr
import soundfile as sf

from models.model_manager import ModelManager
from pipeline.speech_to_translation import process_audio_bytes, LANGUAGE_MENU

# ── Boot ──────────────────────────────────────────────────────────────────
print("\nLoading models…")
manager = ModelManager()
manager.load_models()
print("All models ready.\n")

# ── Language options ───────────────────────────────────────────────────────
LANG_CHOICES   = ["Auto Detect"] + [name for _, name in LANGUAGE_MENU.values()]
TARGET_CHOICES = [name for _, name in LANGUAGE_MENU.values()]
NLLB_BY_NAME   = {name: code for code, name in LANGUAGE_MENU.values()}


def translate_audio(audio, source_lang_name, target_lang_name, preserve_voice, vc_strength):
    if audio is None:
        return "⚠️ No audio recorded.", "", None, "Please record audio first."

    sample_rate, audio_array = audio

    if audio_array.ndim > 1:
        audio_array = audio_array.mean(axis=1)
    audio_array = audio_array.astype(np.float32)

    peak = np.max(np.abs(audio_array))
    if peak > 0:
        audio_array = audio_array / peak

    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
        tmp_path = f.name
    sf.write(tmp_path, audio_array, sample_rate)

    try:
        source_lang_code = None if source_lang_name == "Auto Detect" else NLLB_BY_NAME.get(source_lang_name)
        target_lang_code = NLLB_BY_NAME.get(target_lang_name, "hin_Deva")

        t0 = time.time()

        source_text, translated_text, detected, output_audio_path = process_audio_bytes(
            manager=manager,
            audio_path=tmp_path,
            source_lang=source_lang_code,
            target_lang=target_lang_code,
            preserve_voice=preserve_voice,
            vc_strength=vc_strength,
        )

        elapsed = time.time() - t0
        status  = f"✅ Done in {elapsed:.1f}s  (detected: {detected})"
        return source_text, translated_text, output_audio_path, status

    except Exception as e:
        return "", "", None, f"❌ Error: {e}"
    finally:
        try:
            os.unlink(tmp_path)
        except Exception:
            pass


# ── Gradio UI (Gradio 6 compatible) ───────────────────────────────────────

css = """
#title    { text-align: center; margin-bottom: 4px; }
#subtitle { text-align: center; color: #888; margin-bottom: 20px; font-size: 14px; }
footer    { display: none !important; }
"""

theme = gr.themes.Soft(primary_hue="blue", neutral_hue="slate")

with gr.Blocks() as demo:

    gr.HTML("<h1 id='title'>🌐 Multilingual Voice Translator</h1>")
    gr.HTML("<p id='subtitle'>Whisper · NLLB-200 · MMS-TTS · Voice Preservation</p>")

    with gr.Row():

        # ── Left: inputs ──────────────────────────────────────────────
        with gr.Column(scale=1):

            audio_input = gr.Audio(
                sources=["microphone", "upload"],
                type="numpy",
                label="🎤 Record or Upload Audio",
            )

            with gr.Row():
                source_lang = gr.Dropdown(
                    choices=LANG_CHOICES,
                    value="Auto Detect",
                    label="Source Language",
                )
                target_lang = gr.Dropdown(
                    choices=TARGET_CHOICES,
                    value="Hindi",
                    label="Target Language",
                )

            with gr.Accordion("🎭 Voice Preservation", open=True):
                preserve_voice = gr.Checkbox(
                    value=True,
                    label="Preserve speaker voice (pitch + speed + tone)",
                )
                vc_strength = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.6, step=0.05,
                    label="Conversion Strength",
                    info="0.1 = subtle  ·  0.6 = balanced  ·  1.0 = aggressive",
                )

            translate_btn = gr.Button("▶  Translate", variant="primary", size="lg")

            status_box = gr.Textbox(
                label="Status", value="Ready", interactive=False,
            )

        # ── Right: outputs ────────────────────────────────────────────
        with gr.Column(scale=1):

            source_text_out = gr.Textbox(
                label="📝 Transcribed Text", lines=5, interactive=False,
            )
            translated_text_out = gr.Textbox(
                label="🌍 Translated Text", lines=5, interactive=False,
            )
            audio_output = gr.Audio(
                label="🔊 Translated Speech",
                type="filepath",
                autoplay=True,
            )

    gr.HTML("<hr style='margin:24px 0;border-color:#333'>")
    gr.HTML("""
    <div style='text-align:center;color:#888;font-size:13px;padding:0 20px'>
      <b>1.</b> Click the microphone and speak (or upload a .wav/.mp3 file)<br>
      <b>2.</b> Select source and target languages<br>
      <b>3.</b> Toggle Voice Preservation to keep your voice characteristics<br>
      <b>4.</b> Click <b>Translate</b> — audio plays automatically
    </div>
    """)

    # ── Wire ──────────────────────────────────────────────────────────
    _inputs  = [audio_input, source_lang, target_lang, preserve_voice, vc_strength]
    _outputs = [source_text_out, translated_text_out, audio_output, status_box]

    translate_btn.click(fn=translate_audio, inputs=_inputs, outputs=_outputs, show_progress="full")
    audio_input.stop_recording(fn=translate_audio, inputs=_inputs, outputs=_outputs, show_progress="full")


# ── Launch ─────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    share = os.getenv("SHARE", "false").lower() == "true"
    port  = int(os.getenv("PORT", 7860))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=True,
        show_error=True,
        theme=theme,        # moved to launch() in Gradio 6
        css=css,            # moved to launch() in Gradio 6
    )
