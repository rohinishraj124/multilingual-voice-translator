import time
import threading
import tkinter as tk
import customtkinter as ctk

from translation.languages import LANGUAGES
from history.history_manager import HistoryManager
from translation.language_manager import LanguageManager
from pipeline.speech_to_translation import process_audio, reset_voice_profile
from streaming.stream_manager import StreamManager

_LANG_NAMES   = ["Auto Detect"] + list(LANGUAGES.keys())
_TARGET_NAMES = list(LANGUAGES.keys())

class MainWindow:

    def __init__(self, model_manager):
        self.model_manager     = model_manager
        self.history_manager   = HistoryManager()
        self.stream_manager    = None
        self._live_loop_active = False

        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")

        self.root = ctk.CTk()
        self.root.title("Multilingual Voice Translator")
        self.root.geometry("980x820")
        self.root.minsize(800, 680)

        self._build_layout()
        self._bind_shortcuts()

    def _build_layout(self):
        self.root.configure(fg_color="#000000")
        self.root.grid_columnconfigure(1, weight=1)
        self.root.grid_rowconfigure(0, weight=1)

        self.sidebar_frame = ctk.CTkFrame(self.root, width=220, corner_radius=0, fg_color="#101010", border_width=0)
        self.sidebar_frame.grid(row=0, column=0, sticky="nsew")
        self.sidebar_frame.grid_propagate(False)

        self.main_frame = ctk.CTkFrame(self.root, fg_color="#000000", corner_radius=0, border_width=0)
        self.main_frame.grid(row=0, column=1, sticky="nsew")
        self.main_frame.grid_columnconfigure(0, weight=1)
        self.main_frame.grid_rowconfigure(1, weight=1)

        self.header_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.header_frame.grid(row=0, column=0, sticky="ew", padx=20, pady=(20, 10))

        self.panels_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.panels_frame.grid(row=1, column=0, sticky="nsew", padx=20, pady=10)
        self.panels_frame.columnconfigure(0, weight=1)
        self.panels_frame.columnconfigure(1, weight=1)
        self.panels_frame.rowconfigure(0, weight=1)

        self.source_frame = ctk.CTkFrame(self.panels_frame, fg_color="#1A1A1A", corner_radius=12)
        self.source_frame.grid(row=0, column=0, sticky="nsew", padx=(0, 10))

        self.translation_frame = ctk.CTkFrame(self.panels_frame, fg_color="#1A1A1A", corner_radius=12)
        self.translation_frame.grid(row=0, column=1, sticky="nsew", padx=(10, 0))

        self.footer_frame = ctk.CTkFrame(self.main_frame, fg_color="transparent")
        self.footer_frame.grid(row=2, column=0, sticky="ew", padx=20, pady=(10, 20))

        self._build_widgets()

    def _build_widgets(self):

        ctk.CTkLabel(
            self.sidebar_frame,
            text="VOICE\nTRANSLATOR",
            font=ctk.CTkFont(family="Inter", size=22, weight="bold"),
            text_color="#FFFFFF",
            justify="left"
        ).pack(padx=20, pady=(40, 40), anchor="w")

        btn_padding = {"padx": 15, "pady": 5, "fill": "x"}
        
        self.record_button = ctk.CTkButton(
            self.sidebar_frame, text="🎤  Record Audio", 
            height=40, corner_radius=8, fg_color="#2b5ea7", hover_color="#3a7ebf",
            command=self.start_recording,
        )
        self.record_button.pack(**btn_padding)

        self.start_live_button = ctk.CTkButton(
            self.sidebar_frame, text="🟢  Start Live", 
            height=40, corner_radius=8, fg_color="#2e7d32", hover_color="#388e3c",
            command=self.start_live_translation,
        )
        self.start_live_button.pack(**btn_padding)

        self.stop_live_button = ctk.CTkButton(
            self.sidebar_frame, text="🔴  Stop Live", 
            height=40, corner_radius=8, fg_color="#c62828", hover_color="#d32f2f",
            command=self.stop_live_translation,
        )
        self.stop_live_button.pack(**btn_padding)

        ctk.CTkFrame(self.sidebar_frame, height=2, fg_color="#2A2A2A").pack(fill="x", padx=20, pady=15)

        self.history_button = ctk.CTkButton(
            self.sidebar_frame, text="📜  History", 
            height=40, corner_radius=8, fg_color="transparent", border_width=1, border_color="#3A3A3A",
            command=self.show_history,
        )
        self.history_button.pack(**btn_padding)

        self.export_button = ctk.CTkButton(
            self.sidebar_frame, text="💾  Export CSV", 
            height=40, corner_radius=8, fg_color="transparent", border_width=1, border_color="#3A3A3A",
            command=self.export_history,
        )
        self.export_button.pack(**btn_padding)

        self.clear_button = ctk.CTkButton(
            self.sidebar_frame, text="🗑  Clear All", 
            height=40, corner_radius=8, fg_color="transparent", border_width=1, border_color="#3A3A3A",
            command=self.clear_text,
        )
        self.clear_button.pack(**btn_padding)

        lang_row = ctk.CTkFrame(self.header_frame, fg_color="transparent")
        lang_row.pack(fill="x")

        ctk.CTkLabel(lang_row, text="Source:", font=ctk.CTkFont(size=13, weight="bold")).pack(side="left", padx=(0, 10))
        self.source_lang_menu = ctk.CTkOptionMenu(
            lang_row, values=_LANG_NAMES, width=150, 
            fg_color="#2A2A2A", button_color="#3A3A3A", button_hover_color="#4A4A4A"
        )
        self.source_lang_menu.set("Auto Detect")
        self.source_lang_menu.pack(side="left")

        ctk.CTkLabel(lang_row, text="→", font=ctk.CTkFont(size=20)).pack(side="left", padx=20)

        ctk.CTkLabel(lang_row, text="Target:", font=ctk.CTkFont(size=13, weight="bold")).pack(side="left", padx=(0, 10))
        self.target_lang_menu = ctk.CTkOptionMenu(
            lang_row, values=_TARGET_NAMES, width=150,
            fg_color="#2A2A2A", button_color="#3A3A3A", button_hover_color="#4A4A4A"
        )
        self.target_lang_menu.set("Hindi")
        self.target_lang_menu.pack(side="left")

        self.detected_pill = ctk.CTkLabel(
            lang_row, text="",
            font=ctk.CTkFont(size=11, weight="bold"),
            fg_color="#1a3d6e", corner_radius=12, padx=12, pady=4,
        )
        self.detected_pill.pack(side="right")

        self._build_text_panel(self.source_frame,      "SOURCE TEXT",  "source_textbox",      "source_count")
        self._build_text_panel(self.translation_frame, "TRANSLATION",  "translation_textbox", "translation_count", accent=True)

        settings_row = ctk.CTkFrame(self.footer_frame, fg_color="transparent")
        settings_row.pack(fill="x", pady=(0, 10))

        self.voice_preserve_var = tk.BooleanVar(value=True)
        self.vp_switch = ctk.CTkSwitch(
            settings_row, text="Preserve Voice", variable=self.voice_preserve_var,
            font=ctk.CTkFont(size=12), progress_color="#2b5ea7",
            command=self._on_vp_toggle,
        )
        self.vp_switch.pack(side="left", padx=(0, 20))

        ctk.CTkLabel(settings_row, text="Strength:", font=ctk.CTkFont(size=12)).pack(side="left", padx=(0, 10))
        self.strength_var = tk.DoubleVar(value=0.6)
        self.strength_slider = ctk.CTkSlider(
            settings_row, from_=0.1, to=1.0, variable=self.strength_var,
            width=120, progress_color="#2b5ea7", button_color="#3a7ebf",
            command=self._on_strength_change,
        )
        self.strength_slider.pack(side="left")
        self.strength_label = ctk.CTkLabel(settings_row, text="0.6", font=ctk.CTkFont(size=12, weight="bold"), width=35)
        self.strength_label.pack(side="left")

        self.reset_voice_btn = ctk.CTkButton(
            settings_row, text="↺ Reset Profile",
            width=110, height=28, corner_radius=6,
            fg_color="#333333", hover_color="#444444",
            font=ctk.CTkFont(size=11),
            command=self._reset_voice,
        )
        self.reset_voice_btn.pack(side="left", padx=20)

        self.voice_mode_label = ctk.CTkLabel(
            settings_row, text=self._voice_mode_text(),
            font=ctk.CTkFont(size=11), fg_color="#222222", corner_radius=6, padx=10, pady=3,
        )
        self.voice_mode_label.pack(side="right")

        self.progress_bar = ctk.CTkProgressBar(self.footer_frame, height=4, progress_color="#2b5ea7", fg_color="#1A1A1A")
        self.progress_bar.pack(fill="x", pady=(5, 5))
        self.progress_bar.set(0)

        self.status_label = ctk.CTkLabel(
            self.footer_frame, text="Status: Ready",
            font=ctk.CTkFont(size=12), text_color="gray70", anchor="w",
        )
        self.status_label.pack(fill="x")

        self.voice_preserve_var.trace_add(
            "write",
            lambda *_: self.voice_mode_label.configure(text=self._voice_mode_text()),
        )

    def _voice_mode_text(self):
        if self.voice_preserve_var.get():
            return f"🎭 Voice ON  (strength {self.strength_var.get():.1f})"
        return "🔈 Voice OFF  (MMS-TTS)"

    def _on_vp_toggle(self):
        on = self.voice_preserve_var.get()
        self.strength_slider.configure(state="normal" if on else "disabled")
        self.reset_voice_btn.configure(state="normal" if on else "disabled")

    def _on_strength_change(self, value):
        self.strength_label.configure(text=f"{value:.1f}")

    def _reset_voice(self):
        reset_voice_profile()
        self.update_status("Voice profile reset — speak again to re-sample")

    def _build_text_panel(self, parent, label, box_attr, count_attr, accent=False):
        top = ctk.CTkFrame(parent, fg_color="transparent")
        top.pack(fill="x", padx=12, pady=(12, 0))
        
        ctk.CTkLabel(
            top, text=label, 
            font=ctk.CTkFont(size=11, weight="bold"),
            text_color="gray60"
        ).pack(side="left")
        
        ctk.CTkButton(
            top, text="COPY", width=50, height=22,
            fg_color="#2A2A2A", hover_color="#3A3A3A",
            font=ctk.CTkFont(size=9, weight="bold"),
            command=lambda a=box_attr: self._copy(a),
        ).pack(side="right")
        
        tb = ctk.CTkTextbox(
            parent, corner_radius=8, 
            fg_color="#0F0F0F", border_width=1, border_color="#2A2A2A",
            font=ctk.CTkFont(family="Consolas", size=14),
            activate_scrollbars=True
        )
        if accent:
            tb.configure(border_color="#1a3d6e")
            
        tb.pack(fill="both", expand=True, padx=12, pady=(8, 0))
        setattr(self, box_attr, tb)
        
        lbl = ctk.CTkLabel(
            parent, text="0 words", font=ctk.CTkFont(size=10),
            text_color="gray40", anchor="e",
        )
        lbl.pack(fill="x", padx=15, pady=(4, 8))
        setattr(self, count_attr, lbl)
        
        tb.bind("<KeyRelease>", lambda e, a=box_attr, c=count_attr: self._update_count(a, c))

    def _bind_shortcuts(self):
        self.root.bind(
            "<space>",
            lambda e: self.start_recording() if str(self.record_button.cget("state")) == "normal" else None,
        )

    def _copy(self, box_attr):
        text = getattr(self, box_attr).get("1.0", "end").strip()
        self.root.clipboard_clear()
        self.root.clipboard_append(text)
        self.update_status("Copied to clipboard")

    def _update_count(self, box_attr, count_attr):
        text = getattr(self, box_attr).get("1.0", "end").strip()
        words = len(text.split()) if text else 0
        getattr(self, count_attr).configure(text=f"{words} words · {len(text)} chars")

    def _refresh_counts(self):
        self._update_count("source_textbox", "source_count")
        self._update_count("translation_textbox", "translation_count")

    def update_status(self, message):
        self.root.after(0, lambda: self.status_label.configure(text=f"Status: {message}"))

    def update_progress(self, value):
        self.root.after(0, lambda: self.progress_bar.set(value))

    def update_textboxes(self, source_text, translated_text):
        def _do():
            self.source_textbox.delete("1.0", "end")
            self.translation_textbox.delete("1.0", "end")
            self.source_textbox.insert("1.0", source_text)
            self.translation_textbox.insert("1.0", translated_text)
            self._refresh_counts()
        self.root.after(0, _do)

    def clear_text(self):
        self.source_textbox.delete("1.0", "end")
        self.translation_textbox.delete("1.0", "end")
        self.progress_bar.set(0)
        self.detected_pill.configure(text="")
        self._refresh_counts()
        self.update_status("Ready")

    def start_recording(self):
        self.record_button.configure(state="disabled")
        self.progress_bar.set(0)
        threading.Thread(
            target=self._run_pipeline_worker,
            args=(self.source_lang_menu.get(), self.target_lang_menu.get()),
            daemon=True,
        ).start()

    def _run_pipeline_worker(self, source_language, target_language):
        t0 = time.time()
        try:
            self.update_textboxes("", "")
            self.update_progress(0.1)

            source_lang_code = None if source_language == "Auto Detect" else LanguageManager.get_code(source_language)
            target_lang_code = LanguageManager.get_code(target_language)
            preserve_voice   = self.voice_preserve_var.get()
            vc_strength      = round(self.strength_var.get(), 1)

            source_text, translated_text, detected = process_audio(
                self.model_manager,
                source_lang=source_lang_code,
                target_lang=target_lang_code,
                status_callback=self.update_status,
                preserve_voice=preserve_voice,
                vc_strength=vc_strength,
            )

            self.history_manager.add_record(
                source_language=source_language,
                target_language=target_language,
                source_text=source_text,
                translated_text=translated_text,
            )

            elapsed = time.time() - t0
            self.update_progress(0.9)
            self.update_textboxes(source_text, translated_text)
            self.update_progress(1.0)
            self.update_status(f"Done in {elapsed:.1f}s  (detected: {detected})")
            self.root.after(0, lambda: self.detected_pill.configure(text=f"  {detected}  "))
            self.root.after(0, lambda: self.voice_mode_label.configure(text=self._voice_mode_text()))

        except Exception as e:
            self.update_status(f"Error: {e}")
        finally:
            self.update_progress(0)
            self.root.after(0, lambda: self.record_button.configure(state="normal"))

    def start_live_translation(self):
        if self.stream_manager and self.stream_manager.running:
            return
        self.start_live_button.configure(state="disabled")
        self.record_button.configure(state="disabled")
        source_language = self.source_lang_menu.get()
        source_code = None if source_language == "Auto Detect" else LanguageManager.get_code(source_language)
        target_code = LanguageManager.get_code(self.target_lang_menu.get())
        self.stream_manager = StreamManager(self.model_manager, source_code, target_code)
        self.stream_manager.start_streaming()
        self.update_status("🔴 Live Streaming…")
        if not self._live_loop_active:
            self._live_loop_active = True
            self._update_live_display()

    def stop_live_translation(self):
        if not self.stream_manager:
            return
        self.stream_manager.stop_streaming()
        self.start_live_button.configure(state="normal")
        self.record_button.configure(state="normal")
        self.update_status("Live Stopped")

    def _update_live_display(self):
        if not self._live_loop_active:
            return
        if self.stream_manager:
            try:
                while not self.stream_manager.transcript_queue.empty():
                    text = self.stream_manager.transcript_queue.get()
                    self.source_textbox.insert("end", text + "\n")
                    self.source_textbox.see("end")
                while not self.stream_manager.translation_display_queue.empty():
                    text = self.stream_manager.translation_display_queue.get()
                    self.translation_textbox.insert("end", text + "\n")
                    self.translation_textbox.see("end")
                self._refresh_counts()
            except Exception as e:
                print(f"[UI live update error] {e}")
        self.root.after(100, self._update_live_display)

    def show_history(self):
        history = self.history_manager.get_history()
        popup = ctk.CTkToplevel(self.root)
        popup.title("Translation History")
        popup.geometry("960x640")
        popup.configure(fg_color="#000000")
        popup.grab_set()

        search_var = tk.StringVar()
        ctk.CTkEntry(
            popup, placeholder_text="Search history...",
            textvariable=search_var, height=40,
            fg_color="#1A1A1A", border_color="#333333",
            corner_radius=8
        ).pack(fill="x", padx=20, pady=(20, 10))

        textbox = ctk.CTkTextbox(
            popup, activate_scrollbars=True,
            fg_color="#0F0F0F", border_width=1, border_color="#2A2A2A",
            font=ctk.CTkFont(family="Consolas", size=13),
            corner_radius=12
        )
        textbox.pack(fill="both", expand=True, padx=20, pady=10)

        def _render(filter_text=""):
            textbox.configure(state="normal")
            textbox.delete("1.0", "end")
            items = (
                [i for i in reversed(history)
                 if filter_text.lower() in (i["source_text"] + i["translated_text"]).lower()]
                if filter_text else list(reversed(history))
            )
            if not items:
                textbox.insert("end", "No results." if filter_text else "No history yet.")
            else:
                for item in items:
                    textbox.insert("end",
                        f"{'─'*80}\n"
                        f"🕐  {item['timestamp']}     "
                        f"{item['source_language']}  →  {item['target_language']}\n\n"
                        f"Source:\n{item['source_text']}\n\n"
                        f"Translation:\n{item['translated_text']}\n\n"
                    )
            textbox.configure(state="disabled")

        search_var.trace_add("write", lambda *_: _render(search_var.get()))
        _render()

        btns = ctk.CTkFrame(popup, fg_color="transparent")
        btns.pack(fill="x", padx=20, pady=(10, 20))

        ctk.CTkButton(
            btns, text="💾  Export CSV", width=140, height=36,
            fg_color="#2b5ea7", hover_color="#3a7ebf",
            command=lambda: (self.export_history(), popup.focus()),
        ).pack(side="left", padx=4)

        def _clear():
            self.history_manager.clear_history()
            history.clear()
            _render()

        ctk.CTkButton(
            btns, text="🗑  Clear History", width=140, height=36,
            fg_color="#c62828", hover_color="#d32f2f",
            command=_clear,
        ).pack(side="left", padx=4)

        ctk.CTkButton(
            btns, text="Close", width=100, height=36,
            fg_color="#333333", hover_color="#444444",
            command=popup.destroy,
        ).pack(side="right", padx=4)

    def export_history(self):
        try:
            filename = self.history_manager.export_csv()
            self.update_status(f"Exported → {filename}")
        except Exception as e:
            self.update_status(f"Export failed: {e}")

    def run(self):
        self.root.protocol("WM_DELETE_WINDOW", self._on_close)
        self.root.mainloop()

    def _on_close(self):
        self._live_loop_active = False
        if self.stream_manager and self.stream_manager.running:
            self.stream_manager.stop_streaming()
        self.root.destroy()
