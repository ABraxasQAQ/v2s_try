from __future__ import annotations

import queue
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from transcribe_video import DEFAULT_MODEL, TranscriptionResult, transcribe_video_url


LANGUAGE_OPTIONS = [
    ("Auto detect", ""),
    ("Chinese (zh)", "zh"),
    ("English (en)", "en"),
    ("Japanese (ja)", "ja"),
    ("Korean (ko)", "ko"),
    ("French (fr)", "fr"),
    ("German (de)", "de"),
    ("Spanish (es)", "es"),
]

MODEL_OPTIONS = ["tiny", "base", "small", "medium", "large-v3"]
BROWSER_COOKIE_OPTIONS = ["", "chrome", "edge", "firefox", "brave", "chromium", "opera", "vivaldi"]


class VideoTranscriberApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("Video URL to SRT")
        self.geometry("760x520")
        self.minsize(680, 460)

        self.events: queue.Queue[tuple[str, object]] = queue.Queue()
        self.worker: threading.Thread | None = None

        self.url_var = tk.StringVar()
        self.folder_var = tk.StringVar()
        self.language_var = tk.StringVar(value=LANGUAGE_OPTIONS[0][0])
        self.model_var = tk.StringVar(value=DEFAULT_MODEL)
        self.device_var = tk.StringVar(value="cpu")
        self.cookies_var = tk.StringVar()
        self.browser_cookie_var = tk.StringVar(value=BROWSER_COOKIE_OPTIONS[0])

        self._build_ui()
        self.after(150, self._poll_events)

    def _build_ui(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(5, weight=1)

        container = ttk.Frame(self, padding=16)
        container.grid(row=0, column=0, sticky="nsew")
        container.columnconfigure(1, weight=1)
        container.rowconfigure(7, weight=1)

        ttk.Label(container, text="Video URL").grid(row=0, column=0, sticky="w", pady=(0, 6))
        url_entry = ttk.Entry(container, textvariable=self.url_var)
        url_entry.grid(row=0, column=1, columnspan=3, sticky="ew", pady=(0, 6))

        ttk.Label(container, text="Save folder").grid(row=1, column=0, sticky="w", pady=6)
        ttk.Entry(container, textvariable=self.folder_var).grid(row=1, column=1, sticky="ew", pady=6)
        ttk.Button(container, text="Browse...", command=self._choose_folder).grid(
            row=1, column=2, sticky="ew", padx=(8, 0), pady=6
        )

        ttk.Label(container, text="Language").grid(row=2, column=0, sticky="w", pady=6)
        ttk.Combobox(
            container,
            textvariable=self.language_var,
            values=[label for label, _ in LANGUAGE_OPTIONS],
            state="readonly",
            width=18,
        ).grid(row=2, column=1, sticky="w", pady=6)

        ttk.Label(container, text="Model").grid(row=2, column=2, sticky="e", padx=(12, 6), pady=6)
        ttk.Combobox(
            container,
            textvariable=self.model_var,
            values=MODEL_OPTIONS,
            state="readonly",
            width=12,
        ).grid(row=2, column=3, sticky="w", pady=6)

        ttk.Label(container, text="Device").grid(row=3, column=0, sticky="w", pady=6)
        ttk.Combobox(
            container,
            textvariable=self.device_var,
            values=["cpu", "cuda"],
            state="readonly",
            width=18,
        ).grid(row=3, column=1, sticky="w", pady=6)

        self.start_button = ttk.Button(container, text="Start", command=self._start)
        self.start_button.grid(row=3, column=3, sticky="e", pady=6)

        ttk.Label(container, text="Cookies file").grid(row=4, column=0, sticky="w", pady=6)
        ttk.Entry(container, textvariable=self.cookies_var).grid(row=4, column=1, sticky="ew", pady=6)
        ttk.Button(container, text="Browse...", command=self._choose_cookies).grid(
            row=4, column=2, sticky="ew", padx=(8, 0), pady=6
        )

        ttk.Label(container, text="Browser cookies").grid(row=5, column=0, sticky="w", pady=6)
        ttk.Combobox(
            container,
            textvariable=self.browser_cookie_var,
            values=BROWSER_COOKIE_OPTIONS,
            state="readonly",
            width=10,
        ).grid(row=5, column=1, sticky="w", pady=6)

        ttk.Label(container, text="Progress").grid(row=6, column=0, sticky="w", pady=(14, 6))
        log_frame = ttk.Frame(container)
        log_frame.grid(row=7, column=0, columnspan=4, sticky="nsew")
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)

        self.log_text = tk.Text(log_frame, height=12, wrap="word", state="disabled")
        self.log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(log_frame, command=self.log_text.yview)
        scrollbar.grid(row=0, column=1, sticky="ns")
        self.log_text.configure(yscrollcommand=scrollbar.set)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(container, textvariable=self.status_var).grid(
            row=8, column=0, columnspan=4, sticky="w", pady=(10, 0)
        )

        url_entry.focus_set()

    def _choose_folder(self) -> None:
        folder = filedialog.askdirectory(title="Choose save folder")
        if folder:
            self.folder_var.set(folder)

    def _choose_cookies(self) -> None:
        cookies = filedialog.askopenfilename(
            title="Choose cookies.txt",
            filetypes=[("Cookies text file", "*.txt"), ("All files", "*.*")],
        )
        if cookies:
            self.cookies_var.set(cookies)

    def _selected_language(self) -> str | None:
        selected = self.language_var.get()
        for label, code in LANGUAGE_OPTIONS:
            if label == selected:
                return code or None
        return None

    def _append_log(self, message: str) -> None:
        self.log_text.configure(state="normal")
        self.log_text.insert("end", f"{message}\n")
        self.log_text.see("end")
        self.log_text.configure(state="disabled")

    def _set_running(self, running: bool) -> None:
        self.start_button.configure(state="disabled" if running else "normal")
        self.status_var.set("Running..." if running else "Ready")

    def _start(self) -> None:
        url = self.url_var.get().strip()
        folder = self.folder_var.get().strip()
        if not url:
            messagebox.showwarning("Missing URL", "Please enter a video URL.")
            return
        if not folder:
            messagebox.showwarning("Missing folder", "Please choose a save folder.")
            return

        save_dir = Path(folder)
        save_dir.mkdir(parents=True, exist_ok=True)
        model = self.model_var.get()
        language = self._selected_language()
        device = self.device_var.get()
        compute_type = "float16" if device == "cuda" else "int8"
        cookies = self.cookies_var.get().strip() or None
        browser_cookies = self.browser_cookie_var.get().strip() or None

        self._set_running(True)
        self._append_log("Starting...")
        self.worker = threading.Thread(
            target=self._run_transcription,
            args=(url, save_dir, model, language, device, compute_type, cookies, browser_cookies),
            daemon=True,
        )
        self.worker.start()

    def _run_transcription(
        self,
        url: str,
        save_dir: Path,
        model: str,
        language: str | None,
        device: str,
        compute_type: str,
        cookies: str | None,
        browser_cookies: str | None,
    ) -> None:
        try:
            result = transcribe_video_url(
                url,
                save_dir,
                model_name=model,
                language=language,
                device=device,
                compute_type=compute_type,
                output_formats=("srt",),
                cookies=cookies,
                cookies_from_browser=browser_cookies,
                progress=lambda message: self.events.put(("log", message)),
            )
            self.events.put(("done", result))
        except Exception as exc:
            self.events.put(("error", exc))

    def _poll_events(self) -> None:
        try:
            while True:
                event, payload = self.events.get_nowait()
                if event == "log":
                    self._append_log(str(payload))
                elif event == "done":
                    self._handle_done(payload)
                elif event == "error":
                    self._handle_error(payload)
        except queue.Empty:
            pass
        self.after(150, self._poll_events)

    def _handle_done(self, result: object) -> None:
        self._set_running(False)
        if not isinstance(result, TranscriptionResult):
            messagebox.showinfo("Done", "Finished.")
            return

        srt_path = result.output_paths.get("srt")
        self._append_log(f"Audio: {result.audio_path}")
        if srt_path:
            self._append_log(f"SRT: {srt_path}")
        messagebox.showinfo(
            "Done",
            "Finished.\n\n"
            f"Audio:\n{result.audio_path}\n\n"
            f"SRT:\n{srt_path or 'Not created'}",
        )

    def _handle_error(self, error: object) -> None:
        self._set_running(False)
        self._append_log(f"Error: {error}")
        messagebox.showerror("Error", str(error))


def main() -> None:
    app = VideoTranscriberApp()
    app.mainloop()


if __name__ == "__main__":
    main()
