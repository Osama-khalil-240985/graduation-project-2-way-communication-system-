"""Record microphone audio and transcribe it with a pretrained Whisper model."""

import argparse
import os
import queue
import re
import threading
import tkinter as tk
from tkinter import messagebox, ttk

import numpy as np
import sounddevice as sd
from faster_whisper import WhisperModel


SAMPLE_RATE = 16_000
DEFAULT_MODEL = "base.en"
DEFAULT_SECONDS = 3.0
DEFAULT_BEAM_SIZE = 3


def to_gloss_sequence(text):
    """Format transcription text as a basic ASL-style lexical gloss sequence."""
    glosses = re.findall(r"[A-Za-z0-9]+(?:'[A-Za-z0-9]+)?", text.upper())
    wh_words = {"WHERE", "WHAT", "WHEN", "WHO", "WHY", "HOW"}
    if glosses and glosses[0] in wh_words:
        question_word = glosses.pop(0)
        if glosses and glosses[0] in {"IS", "ARE", "AM"}:
            glosses.pop(0)
        glosses.append(question_word)
    return " ".join(glosses)


def transcribe(model, audio):
    """Transcribe one mono float32 recording into a gloss sequence."""
    segments, _ = model.transcribe(
        audio,
        language="en",
        vad_filter=True,
        beam_size=DEFAULT_BEAM_SIZE,
    )
    text = " ".join(segment.text.strip() for segment in segments).strip()
    return to_gloss_sequence(text)


def run_recognizer(model_name, seconds, device, use_gpu, threads, messages, stop_event):
    """Load Whisper and send recognition events to the GUI."""
    try:
        messages.put(("status", f"Loading Whisper model: {model_name}"))
        model = WhisperModel(
            model_name,
            device="cuda" if use_gpu else "cpu",
            compute_type="float16" if use_gpu else "int8",
            cpu_threads=threads,
        )
        accelerator = "GPU (CUDA)" if use_gpu else "CPU"
        messages.put(("ready", f"Ready. Running on {accelerator}."))
        while not stop_event.is_set():
            messages.put(("status", "Listening..."))
            recording = sd.rec(
                int(seconds * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32",
                device=device,
            )
            sd.wait()
            if stop_event.is_set():
                break
            text = transcribe(model, np.asarray(recording[:, 0]))
            messages.put(("text", text or "[no speech detected]"))
    except (OSError, RuntimeError, ValueError, sd.PortAudioError) as error:
        messages.put(("error", str(error)))


def create_window(model_name, seconds, device, use_gpu, threads):
    """Create and run the speech-to-text output window."""
    window = tk.Tk()
    window.title("Speech to Text")
    window.geometry("720x460")
    window.minsize(520, 320)

    status = tk.StringVar(value="Starting...")
    ttk.Label(window, textvariable=status, font=("Segoe UI", 11)).pack(
        anchor="w", padx=16, pady=(16, 8)
    )
    output = tk.Text(window, wrap="word", font=("Segoe UI", 14), state="disabled")
    output.pack(fill="both", expand=True, padx=16, pady=(0, 16))

    messages = queue.Queue()
    stop_event = threading.Event()
    worker = threading.Thread(
        target=run_recognizer,
        args=(model_name, seconds, device, use_gpu, threads, messages, stop_event),
        daemon=True,
    )
    worker.start()

    def append_text(text):
        output.configure(state="normal")
        output.insert("end", text + "\n")
        output.see("end")
        output.configure(state="disabled")

    def poll_messages():
        try:
            while True:
                kind, value = messages.get_nowait()
                if kind == "status":
                    status.set(value)
                elif kind == "ready":
                    status.set(value)
                elif kind == "text":
                    append_text(value)
                    status.set("Ready. Speak into the microphone.")
                elif kind == "error":
                    status.set("Error")
                    messagebox.showerror("Speech-to-text error", value, parent=window)
        except queue.Empty:
            pass
        if window.winfo_exists():
            window.after(100, poll_messages)

    def close():
        stop_event.set()
        window.destroy()

    window.protocol("WM_DELETE_WINDOW", close)
    poll_messages()
    window.mainloop()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default=DEFAULT_MODEL,
        help="Whisper model name or local model directory.",
    )
    parser.add_argument(
        "--seconds",
        type=float,
        default=DEFAULT_SECONDS,
        help="Maximum recording length for each window.",
    )
    parser.add_argument(
        "--device",
        help="Microphone name or numeric device ID.",
    )
    accelerator = parser.add_mutually_exclusive_group()
    accelerator.add_argument(
        "--gpu",
        dest="use_gpu",
        action="store_true",
        help="Use NVIDIA CUDA GPU with float16 inference.",
    )
    accelerator.add_argument(
        "--cpu",
        dest="use_gpu",
        action="store_false",
        help="Use CPU int8 inference (default).",
    )
    parser.set_defaults(use_gpu=False)
    parser.add_argument(
        "--threads",
        type=int,
        default=max(1, os.cpu_count() or 1),
        help="CPU threads for Whisper (default: all available cores).",
    )
    args = parser.parse_args()

    if args.seconds <= 0:
        parser.error("--seconds must be greater than zero")
    if args.threads < 1:
        parser.error("--threads must be positive")

    device = int(args.device) if args.device and args.device.isdigit() else args.device
    create_window(args.model, args.seconds, device, args.use_gpu, args.threads)


if __name__ == "__main__":
    main()
