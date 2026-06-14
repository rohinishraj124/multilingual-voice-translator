   
import os
import time
import tempfile
import threading
import numpy as np
import gradio as gr
import soundfile as sf

from models.model_manager import ModelManager
from pipeline.speech_to_translation import process_audio_bytes, LANGUAGE_MENU

print("\nLoading models…")
manager = ModelManager()
manager.load_models()
print("All models ready.\n")

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

css = """
body, .gradio-container { background-color: #000000 !important; color: #FFFFFF !important; }
#title { 
    text-align: left; 
    margin-bottom: 2px; 
    font-family: 'Inter', sans-serif; 
    font-weight: 800;
    font-size: 28px;
    letter-spacing: -1px;
    background: linear-gradient(to right, #FFFFFF, #888888);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
}
#subtitle { text-align: left; color: #666; margin-bottom: 30px; font-size: 14px; font-weight: 500; }
.gr-button-primary { 
    background: linear-gradient(135deg, #2b5ea7 0%, #1a3d6e 100%) !important; 
    border: none !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
}
.gr-button-primary:hover { transform: translateY(-1px); box-shadow: 0 4px 12px rgba(43, 94, 167, 0.3) !important; }
.gr-input, .gr-box, .gr-form { 
    background-color: #0F0F0F !important; 
    border-color: #222 !important; 
    border-radius: 12px !important;
}
.gr-form { border-width: 1px !important; border-color: #1A1A1A !important; }
.gr-dropdown, .gr-slider { background-color: transparent !important; }
footer { display: none !important; }
#status-container { margin-top: 10px; }
.gr-accordion { border: 1px solid #1A1A1A !important; background-color: #080808 !important; border-radius: 12px !important; }
"""

black_theme = gr.themes.Default(
    primary_hue="blue",
    neutral_hue="slate",
).set(
    body_background_fill="#000000",
    block_background_fill="#080808",
    block_border_width="1px",
    block_border_color="#1A1A1A",
    block_title_text_color="#888888",
    input_background_fill="#0F0F0F",
    button_primary_background_fill="linear-gradient(135deg, #2b5ea7 0%, #1a3d6e 100%)",
)

with gr.Blocks(theme=black_theme, css=css) as demo:

    with gr.Row():
        with gr.Column(scale=1):
            gr.HTML("<h1 id='title'>VOICE TRANSLATOR</h1>")
            gr.HTML("<p id='subtitle'>Whisper · NLLB-200 · MMS-TTS · Voice Preservation</p>")

    with gr.Row(equal_height=True):

        with gr.Column(scale=4):
            with gr.Group():
                audio_input = gr.Audio(
                    sources=["microphone", "upload"],
                    type="numpy",
                    label="🎤 INPUT AUDIO",
                )

            with gr.Row():
                source_lang = gr.Dropdown(
                    choices=LANG_CHOICES,
                    value="Auto Detect",
                    label="FROM",
                )
                target_lang = gr.Dropdown(
                    choices=TARGET_CHOICES,
                    value="Hindi",
                    label="TO",
                )

            with gr.Accordion("🎭 VOICE SETTINGS", open=True):
                preserve_voice = gr.Checkbox(
                    value=True,
                    label="Preserve Voice Profile",
                )
                vc_strength = gr.Slider(
                    minimum=0.1, maximum=1.0, value=0.6, step=0.05,
                    label="Strength",
                )

            translate_btn = gr.Button("▶  TRANSLATE NOW", variant="primary", size="lg")

            with gr.Column(elem_id="status-container"):
                status_box = gr.Textbox(
                    label="SYSTEM STATUS", value="Ready", interactive=False,
                )

        with gr.Column(scale=5):
            with gr.Group():
                source_text_out = gr.Textbox(
                    label="TRANSCRIPTION", lines=7, interactive=False,
                    placeholder="Transcription will appear here..."
                )
                translated_text_out = gr.Textbox(
                    label="TRANSLATION", lines=7, interactive=False,
                    placeholder="Translation will appear here..."
                )
            
            audio_output = gr.Audio(
                label="🔊 SPEECH OUTPUT",
                type="filepath",
                autoplay=True,
            )

    gr.HTML("<hr style='margin:30px 0;border-color:#1A1A1A'>")
    gr.HTML("""
    <div style='display:flex; justify-content:center; gap:40px; color:#555; font-size:12px; font-weight:500;'>
      <span>1. RECORD OR UPLOAD</span>
      <span>2. SELECT LANGUAGES</span>
      <span>3. ADJUST VOICE</span>
      <span>4. GET TRANSLATION</span>
    </div>
    """)

    _inputs  = [audio_input, source_lang, target_lang, preserve_voice, vc_strength]
    _outputs = [source_text_out, translated_text_out, audio_output, status_box]

    translate_btn.click(fn=translate_audio, inputs=_inputs, outputs=_outputs, show_progress="full")
    audio_input.stop_recording(fn=translate_audio, inputs=_inputs, outputs=_outputs, show_progress="full")

if __name__ == "__main__":
    share = os.getenv("SHARE", "false").lower() == "true"
    port  = int(os.getenv("PORT", 7860))

    demo.launch(
        server_name="0.0.0.0",
        server_port=port,
        share=True,
        show_error=True,
        theme=black_theme,                                             
        css=css,            
    )
