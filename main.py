"""
main.py

App entry point. Wires together:
  - PillOverlay (Qt UI, main thread)
  - a background thread running the asyncio STT streamer (voice_stream.py)
  - IntentEngine (regex + optional LLM fallback, dispatches to windows_actions)
  - listening_control.py: gates *when* the pipeline is allowed to listen
    (push-to-talk hotkeys, or a wake-word text filter), so the mic isn't
    streamed/acted on unconditionally.

All cross-thread communication into the UI goes through pyqtSignal, since
Qt widgets may only be touched from the thread that owns them. This applies
to three different background threads here: the asyncio voice thread, the
IntentEngine's ThreadPoolExecutor, and pynput's global-hotkey listener
thread (LISTENING_MODE=ptt only).
"""

from __future__ import annotations

import asyncio
import os
import sys
import threading
import time

from dotenv import load_dotenv
from PyQt6.QtCore import QObject, pyqtSignal
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QApplication, QMenu, QStyle, QSystemTrayIcon

from intent_engine import IntentEngine
from listening_control import PushToTalkController, WakeWordGate
from overlay_widget import PillOverlay
from speech import Speaker, looks_like_echo, phrase_for_label
import voice_stream
from voice_stream import make_streamer


class VoicePipelineBridge(QObject):
    """Marshals events from background threads onto the Qt thread."""

    transcript_received = pyqtSignal(str, bool, bool, int)
    action_started = pyqtSignal(str)
    listening_changed = pyqtSignal(bool)
    ptt_engaged = pyqtSignal()
    ptt_released = pyqtSignal()


class VoiceThread(threading.Thread):
    def __init__(self, bridge: VoicePipelineBridge, wake_word: bool = False) -> None:
        super().__init__(daemon=True)
        self._bridge = bridge
        self._wake_word = wake_word
        self._loop: asyncio.AbstractEventLoop | None = None
        self._streamer = None

    def run(self) -> None:
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)

        deepgram_key = os.environ.get("DEEPGRAM_API_KEY") or None
        backend_override = os.environ.get("STT_BACKEND") or None
        self._streamer = make_streamer(deepgram_key, backend_override, wake_word=self._wake_word)

        self._bridge.listening_changed.emit(True)

        def on_transcript(text: str, is_final: bool, speech_final: bool, utterance_id: int) -> None:
            self._bridge.transcript_received.emit(text, is_final, speech_final, utterance_id)

        try:
            self._loop.run_until_complete(self._streamer.run(on_transcript))
        except Exception as exc:  # noqa: BLE001
            print(f"[main] voice thread stopped: {exc}")
        finally:
            self._bridge.listening_changed.emit(False)

    def stop(self) -> None:
        if self._streamer is not None:
            self._streamer.stop()


def build_tray_icon(app: QApplication, on_quit) -> QSystemTrayIcon:
    icon = app.style().standardIcon(QStyle.StandardPixmap.SP_ComputerIcon)
    tray = QSystemTrayIcon(icon)
    tray.setToolTip("Voice Assistant")
    menu = QMenu()
    quit_action = QAction("Quit")
    quit_action.triggered.connect(on_quit)
    menu.addAction(quit_action)
    tray.setContextMenu(menu)
    tray.show()
    return tray


def main() -> int:
    load_dotenv()

    app = QApplication(sys.argv)
    app.setQuitOnLastWindowClosed(False)

    overlay = PillOverlay()  # hidden by default - reveals itself on PTT/wake word

    bridge = VoicePipelineBridge()

    # After an action (especially play) the mic hears the song / the app
    # launching and Whisper invents "Hey Jarvis, open Chrome". The spoken
    # confirmation is the same problem: ignore transcripts while the voice
    # is talking, and drop a later transcript that is just that sentence.
    ignore_stt_until = 0.0
    ACTION_ECHO_S = 0.8
    # Short on purpose: a longer deaf window swallowed the next command
    # the moment the confirmation finished. Echo matching still drops the
    # sentence Jarvis itself just spoke.
    SPEECH_TAIL_S = 0.8
    recent_speech: list[tuple[float, str]] = []
    stt_language = (os.environ.get("STT_LANGUAGE") or "en").strip().lower()
    speaker = Speaker()
    speaker.start()

    def speech_finished() -> None:
        nonlocal ignore_stt_until
        ignore_stt_until = time.monotonic() + SPEECH_TAIL_S

    def dispatch_action_label(label: str) -> None:
        nonlocal ignore_stt_until
        phrase = phrase_for_label(label, stt_language)
        shown = label[4:].strip() if label.startswith("SAY:") else label
        if phrase and speaker.speak(phrase, on_idle=speech_finished):
            recent_speech.append((time.monotonic(), phrase))
            del recent_speech[:-4]
            # Ceiling while SAPI is talking. speech_finished() shortens it.
            ignore_stt_until = time.monotonic() + 12
        else:
            ignore_stt_until = max(ignore_stt_until, time.monotonic() + ACTION_ECHO_S)
        bridge.action_started.emit(shown)

    intent_engine = IntentEngine(on_action_started=dispatch_action_label)

    # Note: bridge.listening_changed (fired the instant the streamer object
    # is constructed, before any audio is even heard) is intentionally NOT
    # wired to overlay.set_listening - both PTT and wake-word mode drive the
    # overlay's state explicitly (flash_active() / set_listening(False)) at
    # the moments that actually matter (hotkey press, wake word heard,
    # utterance done). Connecting it here as well would immediately
    # overwrite the "JARVIS LISTENING" pop with the plain "Listening..."
    # state within milliseconds, since the streamer starts well before any
    # words are transcribed.
    bridge.action_started.connect(overlay.show_action)

    listening_mode = os.environ.get("LISTENING_MODE", "ptt").lower()
    wake_gate = WakeWordGate()
    awake_utterances: set[int] = set()

    def handle_transcript(text: str, is_final: bool, speech_final: bool, utterance_id: int) -> None:
        # Already on the Qt main thread here (queued-connection delivery of
        # transcript_received), so touching the overlay directly is safe.
        if time.monotonic() < ignore_stt_until:
            return
        now = time.monotonic()
        if any(now - spoken_at < 8 and looks_like_echo(text, phrase) for spoken_at, phrase in recent_speech):
            print(f"[stt] #{utterance_id} ignored own voice: {text!r}")
            return
        kind = "FINAL  " if speech_final else "partial"
        if listening_mode == "wakeword" and voice_stream.wake_model_is_ready():
            stripped = wake_gate.strip(text)
            if not stripped:
                return
            print(f"[stt] #{utterance_id} {kind} heard={text!r}  -> {stripped!r}")
            overlay.flash_active()
            overlay.update_transcript(stripped)
            intent_engine.on_transcript(stripped, is_final, speech_final, utterance_id)
            if speech_final:
                overlay.set_listening(False)
        elif listening_mode == "wakeword":
            stripped = wake_gate.process(text, utterance_id, speech_final)
            gate = "ignored (no wake word)" if stripped is None else f"-> {stripped!r}"
            print(f"[stt] #{utterance_id} {kind} heard={text!r}  {gate}")
            if stripped is None:
                return  # not woken up yet - don't show or act on overheard speech
            if utterance_id not in awake_utterances:
                awake_utterances.add(utterance_id)
                overlay.flash_active()
            overlay.update_transcript(stripped)
            intent_engine.on_transcript(stripped, is_final, speech_final, utterance_id)
            if speech_final:
                awake_utterances.discard(utterance_id)
                overlay.set_listening(False)  # back to idle/hint until the next wake word
        else:
            print(f"[stt] #{utterance_id} {kind} heard={text!r}")
            overlay.update_transcript(text)
            intent_engine.on_transcript(text, is_final, speech_final, utterance_id)

    bridge.transcript_received.connect(handle_transcript)

    voice_thread_box: list[VoiceThread | None] = [None]
    ptt_controller: PushToTalkController | None = None

    if listening_mode == "wakeword":
        overlay.set_idle_hint('Say "Hey Jarvis"')
        voice_thread_box[0] = VoiceThread(bridge, wake_word=True)
        voice_thread_box[0].start()
    else:
        overlay.set_idle_hint("Hold Right Alt to talk  •  Ctrl+Alt+A to toggle")

        def start_listening() -> None:
            # Runs on pynput's listener thread - only touch Qt via signals.
            if voice_thread_box[0] is None:
                vt = VoiceThread(bridge)
                voice_thread_box[0] = vt
                vt.start()
            bridge.ptt_engaged.emit()

        def stop_listening() -> None:
            vt = voice_thread_box[0]
            voice_thread_box[0] = None
            if vt is not None:
                vt.stop()
            bridge.ptt_released.emit()

        bridge.ptt_engaged.connect(lambda: overlay.flash_active())
        bridge.ptt_released.connect(lambda: overlay.set_listening(False))

        ptt_controller = PushToTalkController(
            on_start=start_listening,
            on_stop=stop_listening,
            hold_key=os.environ.get("PTT_HOLD_KEY", "alt_r"),
            toggle_letter=os.environ.get("PTT_TOGGLE_LETTER", "a"),
        )
        ptt_controller.start()

    def on_about_to_quit() -> None:
        if ptt_controller is not None:
            ptt_controller.stop()
        speaker.stop()
        vt = voice_thread_box[0]
        if vt is not None:
            vt.stop()
        intent_engine.shutdown(wait=True)  # must finish before QApplication tears down

    app.aboutToQuit.connect(on_about_to_quit)

    def on_quit() -> None:
        app.quit()

    build_tray_icon(app, on_quit)

    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
