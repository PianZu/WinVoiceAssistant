"""
speech.py

Speaks a short confirmation when a command runs, using the Windows SAPI
voice (no extra dependency: pywin32 is already required). The language
follows STT_LANGUAGE: German phrases for "de", English otherwise.

The mic is live in wake-word mode, so the caller must ignore transcripts
while the voice is speaking and drop later transcripts that are just the
assistant hearing itself.
"""

from __future__ import annotations

import os
import queue
import re
import threading
from typing import Callable, Optional

# Spoken once, from the badge label. Later result strings such as
# "Opening Chrome" are not phrases — those would talk over the badge.
_DE_APPS = {
    "CAMERA": "Die Kamera",
    "KAMERA": "Die Kamera",
    "CALCULATOR": "Der Rechner",
    "CALC": "Der Rechner",
    "RECHNER": "Der Rechner",
    "TASCHENRECHNER": "Der Rechner",
    "TERMINAL": "Das Terminal",
    "CONSOLE": "Das Terminal",
    "CMD": "Das Terminal",
    "KONSOLE": "Das Terminal",
    "FILE EXPLORER": "Der Explorer",
    "EXPLORER": "Der Explorer",
    "FILES": "Der Explorer",
    "DATEIEN": "Der Explorer",
    "CODE": "Visual Studio Code",
    "VS CODE": "Visual Studio Code",
    "VISUAL STUDIO CODE": "Visual Studio Code",
    "NOTEPAD": "Notepad",
    "NOTES": "Notepad",
    "NOTIZBLOCK": "Notepad",
}


def phrase_for_label(label: str, language: str = "de") -> Optional[str]:
    """German or English sentence for a confirmation badge, or None when
    this label should stay silent (result text, unknown badge)."""
    if label.startswith("SAY:"):
        sentence = label[4:].strip()
        return sentence or None
    if (language or "").lower().startswith("de"):
        return _phrase_de(label.strip())
    return _phrase_en(label.strip())


def looks_like_echo(heard: str, spoken: str) -> bool:
    """True when a transcript is mostly the sentence we just spoke."""
    heard_words = re.findall(r"[a-zäöüß0-9]+", heard.lower())
    spoken_words = set(re.findall(r"[a-zäöüß0-9]+", spoken.lower()))
    if len(heard_words) < 2 or not spoken_words:
        return False
    shared = sum(1 for word in heard_words if word in spoken_words)
    return shared >= 2 and shared / len(heard_words) >= 0.6


def _spoken_name(raw: str, language: str) -> str:
    key = re.sub(r"\s+", " ", raw.strip().upper())
    if language == "de" and key in _DE_APPS:
        return _DE_APPS[key]
    if re.fullmatch(r"[A-Z0-9.-]+\.[A-Z]{2,}(?:/\S*)?", key):
        return key.lower()
    return key.title()


def _phrase_de(label: str) -> Optional[str]:
    if label.startswith("Couldn't") or label.startswith("Error:"):
        return "Das hat nicht geklappt."
    matchers: list[tuple[re.Pattern[str], Callable[[re.Match[str]], str]]] = [
        (re.compile(r"^\[LAUNCHED (.+)\]$"), lambda m: f"{_spoken_name(m.group(1), 'de')} wird geöffnet."),
        (re.compile(r"^\[CLOSING (.+)\]$"), lambda m: f"{_spoken_name(m.group(1), 'de')} wird geschlossen."),
        (re.compile(r"^\[NAVIGATING TO (.+)\]$"), lambda m: f"Ich öffne {_spoken_name(m.group(1), 'de')}."),
        (re.compile(r"^\[SEARCHING FOR (.+)\]$"), lambda m: f"Ich suche nach {_spoken_name(m.group(1), 'de')}."),
        (re.compile(r"^\[CAPTURING PHOTO\]$"), lambda m: "Foto wird aufgenommen."),
        (re.compile(r"^\[WRITING TEXT\]$"), lambda m: "Ich schreibe."),
        (re.compile(r"^\[PLAYING (.+)\]$"), lambda m: f"Ich spiele {_spoken_name(m.group(1), 'de')}."),
        (re.compile(r"^\[PAUSED\]$"), lambda m: "Pause."),
        (re.compile(r"^\[RESUMING\]$"), lambda m: "Weiter."),
        (re.compile(r"^\[NEXT TRACK\]$"), lambda m: "Nächster Titel."),
        (re.compile(r"^\[PREVIOUS TRACK\]$"), lambda m: "Vorheriger Titel."),
        (re.compile(r"^\[VOLUME UP\]$"), lambda m: "Lauter."),
        (re.compile(r"^\[VOLUME DOWN\]$"), lambda m: "Leiser."),
        (re.compile(r"^\[VOLUME (.+)\]$"), lambda m: f"Lautstärke {m.group(1).replace('%', '').strip()}."),
        (re.compile(r"^\[SYSTEM VOLUME UP\]$"), lambda m: "Computer lauter."),
        (re.compile(r"^\[SYSTEM VOLUME DOWN\]$"), lambda m: "Computer leiser."),
        (
            re.compile(r"^\[SYSTEM VOLUME (.+)\]$"),
            lambda m: f"Systemlautstärke {m.group(1).replace('%', '').strip()}.",
        ),
        (re.compile(r"^\[MUTED\]$"), lambda m: "Stumm."),
        (re.compile(r"^\[UNMUTED\]$"), lambda m: "Ton an."),
        (re.compile(r"^\[SCREENSHOT\]$"), lambda m: "Screenshot."),
        (re.compile(r"^\[MINIMIZED\]$"), lambda m: "Fenster minimiert."),
        (re.compile(r"^\[MAXIMIZED\]$"), lambda m: "Fenster maximiert."),
        (re.compile(r"^\[RESTORED\]$"), lambda m: "Fenster wiederhergestellt."),
        (re.compile(r"^\[WINDOW LEFT\]$"), lambda m: "Fenster nach links."),
        (re.compile(r"^\[WINDOW RIGHT\]$"), lambda m: "Fenster nach rechts."),
        (re.compile(r"^\[TIMER DONE\]$"), lambda m: "Der Timer ist abgelaufen."),
        (re.compile(r"^\[FOLDER (.+)\]$"), lambda m: f"{_spoken_name(m.group(1), 'de')} wird geöffnet."),
        (re.compile(r"^\[FILE (.+)\]$"), lambda m: f"Ich öffne {_spoken_name(m.group(1), 'de')}."),
        (re.compile(r"^\[SHUFFLE ON\]$"), lambda m: "Zufall an."),
        (re.compile(r"^\[SHUFFLE OFF\]$"), lambda m: "Zufall aus."),
    ]
    for pattern, build in matchers:
        match = pattern.match(label)
        if match:
            return build(match)
    return None


def _phrase_en(label: str) -> Optional[str]:
    if label.startswith("Couldn't") or label.startswith("Error:"):
        return "That didn't work."
    matchers: list[tuple[re.Pattern[str], Callable[[re.Match[str]], str]]] = [
        (re.compile(r"^\[LAUNCHED (.+)\]$"), lambda m: f"Opening {_spoken_name(m.group(1), 'en')}."),
        (re.compile(r"^\[CLOSING (.+)\]$"), lambda m: f"Closing {_spoken_name(m.group(1), 'en')}."),
        (re.compile(r"^\[NAVIGATING TO (.+)\]$"), lambda m: f"Opening {_spoken_name(m.group(1), 'en')}."),
        (re.compile(r"^\[SEARCHING FOR (.+)\]$"), lambda m: f"Searching for {_spoken_name(m.group(1), 'en')}."),
        (re.compile(r"^\[CAPTURING PHOTO\]$"), lambda m: "Taking a photo."),
        (re.compile(r"^\[WRITING TEXT\]$"), lambda m: "Writing."),
        (re.compile(r"^\[PLAYING (.+)\]$"), lambda m: f"Playing {_spoken_name(m.group(1), 'en')}."),
        (re.compile(r"^\[PAUSED\]$"), lambda m: "Paused."),
        (re.compile(r"^\[RESUMING\]$"), lambda m: "Resuming."),
        (re.compile(r"^\[NEXT TRACK\]$"), lambda m: "Next track."),
        (re.compile(r"^\[PREVIOUS TRACK\]$"), lambda m: "Previous track."),
        (re.compile(r"^\[VOLUME UP\]$"), lambda m: "Louder."),
        (re.compile(r"^\[VOLUME DOWN\]$"), lambda m: "Quieter."),
        (re.compile(r"^\[VOLUME (.+)\]$"), lambda m: f"Volume {m.group(1).replace('%', '').strip()}."),
        (re.compile(r"^\[SYSTEM VOLUME UP\]$"), lambda m: "Computer louder."),
        (re.compile(r"^\[SYSTEM VOLUME DOWN\]$"), lambda m: "Computer quieter."),
        (
            re.compile(r"^\[SYSTEM VOLUME (.+)\]$"),
            lambda m: f"System volume {m.group(1).replace('%', '').strip()}.",
        ),
        (re.compile(r"^\[MUTED\]$"), lambda m: "Muted."),
        (re.compile(r"^\[UNMUTED\]$"), lambda m: "Unmuted."),
        (re.compile(r"^\[SCREENSHOT\]$"), lambda m: "Screenshot."),
        (re.compile(r"^\[MINIMIZED\]$"), lambda m: "Window minimized."),
        (re.compile(r"^\[MAXIMIZED\]$"), lambda m: "Window maximized."),
        (re.compile(r"^\[RESTORED\]$"), lambda m: "Window restored."),
        (re.compile(r"^\[WINDOW LEFT\]$"), lambda m: "Window left."),
        (re.compile(r"^\[WINDOW RIGHT\]$"), lambda m: "Window right."),
        (re.compile(r"^\[TIMER DONE\]$"), lambda m: "Timer's up."),
        (re.compile(r"^\[FOLDER (.+)\]$"), lambda m: f"Opening {_spoken_name(m.group(1), 'en')}."),
        (re.compile(r"^\[FILE (.+)\]$"), lambda m: f"Opening {_spoken_name(m.group(1), 'en')}."),
        (re.compile(r"^\[SHUFFLE ON\]$"), lambda m: "Shuffle on."),
        (re.compile(r"^\[SHUFFLE OFF\]$"), lambda m: "Shuffle off."),
    ]
    for pattern, build in matchers:
        match = pattern.match(label)
        if match:
            return build(match)
    return None


class Speaker:
    """Background SAPI voice. speak() returns immediately."""

    def __init__(self) -> None:
        self._queue: queue.Queue[Optional[tuple[int, str]]] = queue.Queue()
        self._gen = 0
        self._lock = threading.Lock()
        self._on_idle: Optional[Callable[[], None]] = None
        self._enabled = os.environ.get("TTS_ENABLED", "1").strip() != "0"
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        if not self._enabled:
            print("[tts] disabled")
            return
        self._thread = threading.Thread(target=self._run, name="tts", daemon=True)
        self._thread.start()

    def speak(self, text: str, on_idle: Optional[Callable[[], None]] = None) -> bool:
        if not self._enabled or not text.strip():
            return False
        with self._lock:
            self._gen += 1
            gen = self._gen
            if on_idle is not None:
                self._on_idle = on_idle
        self._queue.put((gen, text.strip()))
        return True

    def stop(self) -> None:
        if self._thread is None:
            return
        self._queue.put(None)
        self._thread.join(timeout=3)

    def _run(self) -> None:
        import pythoncom

        pythoncom.CoInitialize()
        try:
            voice = self._open_voice()
            if voice is None:
                self._enabled = False
                return
            while True:
                item = self._queue.get()
                if item is None:
                    return
                gen, text = item
                try:
                    voice.Speak(text)
                except Exception as exc:  # noqa: BLE001 - a voice glitch must not kill the app
                    print(f"[tts] speak failed: {exc}")
                with self._lock:
                    idle = self._on_idle if gen == self._gen and self._queue.empty() else None
                if idle is not None:
                    idle()
        finally:
            pythoncom.CoUninitialize()

    def _open_voice(self):
        import win32com.client

        try:
            voice = win32com.client.Dispatch("SAPI.SpVoice")
        except Exception as exc:  # noqa: BLE001
            print(f"[tts] SAPI unavailable: {exc}")
            return None
        token = _select_voice(voice)
        if token is not None:
            voice.Voice = token
            print(f"[tts] voice: {token.GetDescription()}")
        else:
            print("[tts] no German voice found, using the Windows default")
        return voice


def _select_voice(voice):
    """Prefer TTS_VOICE, then any de-DE SAPI token (language id 407)."""
    preferred = os.environ.get("TTS_VOICE", "").strip().lower()
    try:
        tokens = voice.GetVoices()
        count = tokens.Count
    except Exception as exc:  # noqa: BLE001
        print(f"[tts] could not list voices: {exc}")
        return None
    german = None
    for index in range(count):
        token = tokens.Item(index)
        description = token.GetDescription()
        if preferred and preferred in description.lower():
            return token
        language = ""
        try:
            language = (token.GetAttribute("Language") or "").lower()
        except Exception:  # noqa: BLE001
            language = ""
        if german is None and language.startswith("407"):
            german = token
    if preferred:
        print(f"[tts] TTS_VOICE={preferred!r} not installed")
    return german
