"""
simulate_demo.py

Replays a scripted transcript (mimicking the interim/final events a real STT
backend would emit) through the same PillOverlay + IntentEngine pipeline used
in main.py, so the UI transitions and Windows automations can be verified
end-to-end without a microphone or any API credits.

Usage:
    python simulate_demo.py            # dry run: actions are logged, not executed
    python simulate_demo.py --live     # actually opens apps / browsers / camera
"""

from __future__ import annotations

import os
import sys

if "--live" not in sys.argv:
    os.environ.setdefault("DRY_RUN", "1")

from PyQt6.QtCore import QObject, QTimer, pyqtSignal
from PyQt6.QtWidgets import QApplication

from intent_engine import IntentEngine
from overlay_widget import PillOverlay


class _ActionBridge(QObject):
    # IntentEngine may invoke the action-started callback from its
    # ThreadPoolExecutor worker thread (e.g. when an action's result label
    # differs from its initial "Opening..." label). QWidget methods are not
    # thread-safe, so the callback must cross back to the GUI thread via a
    # signal rather than touching the overlay directly.
    action_started = pyqtSignal(str)

# Each step: (delay_ms_after_previous, transcript_so_far, is_final, speech_final, utterance_id)
# Mirrors how a real STT stream grows the transcript token-by-token within one
# utterance, then marks it final on silence.
SCRIPT: list[tuple[int, str, bool, bool, int]] = [
    (400, "hey", False, False, 0),
    (250, "hey can", False, False, 0),
    (250, "hey can you", False, False, 0),
    (250, "hey can you open", False, False, 0),
    (300, "hey can you open arc", False, False, 0),  # <- mid-sentence trigger fires here
    (400, "hey can you open arc for", False, False, 0),
    (300, "hey can you open arc for me", True, True, 0),  # finalized: must NOT refire

    # "search" only fires on speech_final, not mid-sentence: the query is
    # open-ended growing text, so firing early ("search x" -> "search x
    # for" -> ...) would trigger a separate, incomplete search each time.
    (900, "then", False, False, 1),
    (250, "then search", False, False, 1),
    (250, "then search x", False, False, 1),
    (300, "then search x for", False, False, 1),
    (350, "then search x for latest ai news", False, False, 1),
    (400, "then search x for latest ai news", True, True, 1),  # <- fires here (once)

    (900, "now", False, False, 2),
    (250, "now take", False, False, 2),
    (250, "now take a", False, False, 2),
    (300, "now take a photo", False, False, 2),  # <- fires here
    (300, "now take a photo", True, True, 2),

    (900, "write", False, False, 3),
    (300, "write hello", False, False, 3),
    (300, "write hello from", False, False, 3),
    (400, "write hello from the voice assistant demo", False, False, 3),  # <- fires here
    (300, "write hello from the voice assistant demo", True, True, 3),

    (900, "hey", False, False, 4),
    (250, "hey can you", False, False, 4),
    (250, "hey can you please", False, False, 4),
    (250, "hey can you please open up", False, False, 4),
    (300, "hey can you please open up notepad", False, False, 4),  # <- fires here (regression: "notepad" previously unmatched)
    (400, "hey can you please open up notepad for me", True, True, 4),

    (900, "launch", False, False, 5),
    (300, "launch terminal", False, False, 5),  # <- fires here ("launch" as a trigger verb)
    (300, "launch terminal", True, True, 5),

    (900, "please search", False, False, 6),
    (300, "please search for the latest ai news", False, False, 6),
    (300, "please search for the latest ai news", True, True, 6),  # <- fires here (no named target)

    # Punctuation/casing Deepgram's smart-formatting can add, plus new app
    # aliases (calculator, file explorer) - all handled by _normalize_transcript.
    (900, "Hey,", False, False, 7),
    (300, "Hey, can you open", False, False, 7),
    (300, "Hey, can you open the calculator app", False, False, 7),  # <- fires here
    (300, "Hey, can you open the calculator app, please!", True, True, 7),

    (900, "open", False, False, 8),
    (300, "open file explorer", False, False, 8),  # <- fires here
    (300, "open file explorer", True, True, 8),

    # Compound command: opens Chrome mid-sentence, then the trailing
    # "search for ..." fires separately once finalized (regression test for
    # _match_regex's dedup-aware scanning - open_app's "open chrome" match
    # would otherwise keep "winning" on every later pass and hide the
    # search entirely, since it's already fired and gets silently deduped).
    (900, "open", False, False, 9),
    (250, "open chrome", False, False, 9),  # <- fires here (open_app)
    (250, "open chrome and", False, False, 9),
    (250, "open chrome and search", False, False, 9),
    (300, "open chrome and search for", False, False, 9),
    (300, "open chrome and search for python tutorials", False, False, 9),
    (300, "open chrome and search for python tutorials", True, True, 9),  # <- fires here (browser_search)

    # Spoken URL: "dot" said as a word, not an actual period - despeaks to
    # "animusinfra.com" and navigates directly instead of Googling the
    # literal words "animus infra dot com".
    (900, "go", False, False, 10),
    (300, "go to", False, False, 10),
    (300, "go to animus infra dot com", True, True, 10),  # <- fires here (navigate_url)

    # A "search" whose entire query is just a domain should also navigate
    # directly rather than Google-searching the domain name as text.
    (900, "search", False, False, 11),
    (300, "search for github dot com", True, True, 11),  # <- fires here (navigate_url, not browser_search)
]


def run() -> int:
    app = QApplication(sys.argv)
    overlay = PillOverlay()
    overlay.show()
    overlay.flash_active()  # exercise the instant cyan "JARVIS LISTENING" pop
    overlay.set_listening(True)

    fired_log: list[str] = []
    bridge = _ActionBridge()

    def on_action_started(label: str) -> None:
        fired_log.append(label)
        print(f"[simulate_demo] action: {label}")
        bridge.action_started.emit(label)  # safe from any thread

    bridge.action_started.connect(overlay.show_action)
    engine = IntentEngine(on_action_started=on_action_started)
    app.aboutToQuit.connect(lambda: engine.shutdown(wait=True))

    def step(index: int) -> None:
        if index >= len(SCRIPT):
            print("\n[simulate_demo] script complete. Actions fired:")
            for label in fired_log:
                print(f"  - {label}")
            QTimer.singleShot(3000, app.quit)
            return

        delay, text, is_final, speech_final, utterance_id = SCRIPT[index]

        def fire():
            overlay.update_transcript(text)
            engine.on_transcript(text, is_final, speech_final, utterance_id)
            step(index + 1)

        QTimer.singleShot(delay, fire)

    step(0)
    return app.exec()


if __name__ == "__main__":
    sys.exit(run())
