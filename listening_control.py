"""
listening_control.py

Decides *when* the pipeline is allowed to listen, so the mic isn't streamed
and acted on unconditionally and continuously:

  - Push-to-talk (PushToTalkController): mic capture only runs while a
    hotkey is held (default: Right Alt), or between two presses of a
    toggle combo (default: Ctrl+Left Alt+A). No hotkey engaged -> no
    streamer is even started, so nothing is captured at all.

  - Wake word (WakeWordGate): this one is a pure text-level gate, not a
    mic controller - some form of continuous local listening is
    unavoidable for any wake-word system (the mic has to be monitoring in
    order to ever hear the trigger phrase - this is true of Alexa/Google
    Home too, they just don't act on anything until the wake word fires
    locally). What WakeWordGate guarantees is that no transcript reaches
    the UI or the intent engine - i.e. nothing is *shown* or *acted on* -
    until an utterance actually starts with "jarvis" / "hey jarvis", and
    only the text after the wake phrase is forwarded.
"""

from __future__ import annotations

import re
import threading
from typing import Callable, Optional

# Whisper (English models + German accent) constantly mangles "Jarvis" into
# nearby English words. Treat the ones we have actually seen in live logs
# as the same wake word so a good command isn't dropped.
WAKE_PATTERN = re.compile(
    r"\b(?:(?:hey|hallo)\s+)?(?:"
    r"j[ae]rv(?:is|es|ets|ish|as|a)|jerva|jaravis|jairus|"
    r"d[aeo]rv(?:is|es|us|ish)|darva'?s?|derbys|"
    r"doris|drivers|"
    r"java'?s|javas|jaws|"
    r"job\s+is|"
    r"travis"
    r")\b[,:]?\s*",
    re.IGNORECASE,
)


class WakeWordGate:
    """Strips the wake phrase and decides whether a transcript may pass."""

    def __init__(self) -> None:
        self._awake_utterance_id: Optional[int] = None

    def process(self, text: str, utterance_id: int, speech_final: bool) -> Optional[str]:
        """Returns the text to act on (wake phrase stripped), or None if the
        utterance hasn't triggered the wake word (yet)."""
        match = WAKE_PATTERN.search(text)
        if match:
            self._awake_utterance_id = utterance_id
            remainder = text[match.end():]
        elif self._awake_utterance_id == utterance_id:
            # Same utterance already woke us up on an earlier (shorter)
            # partial; keep forwarding even if this exact revision's search
            # somehow doesn't re-match (STT can revise earlier words).
            remainder = text
        else:
            remainder = None

        if speech_final and self._awake_utterance_id == utterance_id:
            self._awake_utterance_id = None  # back to sleep for the next utterance

        return remainder


class PushToTalkController:
    """Global hold-to-talk (default: Right Alt) + toggle (default:
    Ctrl+Alt+A), via a single pynput keyboard hook.

    Toggle detection is done by hand (tracking Ctrl/left-Alt modifier state
    plus the letter key's virtual-key code) rather than via pynput's
    GlobalHotKeys: on Windows, holding Ctrl+Alt changes how the OS resolves
    a letter key, so it often arrives as a bare vk-only KeyCode with no
    `char` set - GlobalHotKeys matches letter hotkeys by `char`, so it
    silently never fires for a Ctrl+Alt+<letter> combo. Comparing `vk`
    directly sidesteps that - but note `KeyCode.from_char(letter).vk` is
    *also* unreliable (pynput's Windows backend leaves it None), so the vk
    is derived straight from the documented Win32 constant instead: letter
    virtual-key codes equal their uppercase ASCII value (VK_A..VK_Z =
    0x41..0x5A), regardless of modifiers (verified against a real synthetic
    Ctrl+Alt+A press).

    The toggle's Alt is intentionally left-Alt only, distinct from the
    hold key (right-Alt/AltGr), so holding PTT and pressing the toggle combo
    can never be ambiguous about which physical key was meant.

    `on_start`/`on_stop` fire on transitions of a single "active" boolean
    that is true while the hold key is down OR toggle mode is switched on -
    so holding the key while toggled on, then releasing it, correctly keeps
    listening active until the toggle is pressed again.

    These callbacks run on pynput's internal listener thread, not the Qt
    main thread - callers must marshal them onto the GUI thread (e.g. via a
    pyqtSignal) before touching any widgets, the same rule that applies to
    IntentEngine's executor callbacks.
    """

    def __init__(
        self,
        on_start: Callable[[], None],
        on_stop: Callable[[], None],
        hold_key: str = "alt_r",
        toggle_letter: str = "a",
    ) -> None:
        from pynput import keyboard as pynput_keyboard

        self._keys = pynput_keyboard.Key
        self._on_start = on_start
        self._on_stop = on_stop

        # pynput on Windows reports the physical Right Alt key as Key.alt_gr,
        # not Key.alt_r (confirmed by capturing a real synthetic press) -
        # accept both so "alt_r" works regardless of platform/driver quirks.
        self._hold_keys = {getattr(self._keys, hold_key)}
        if hold_key == "alt_r":
            self._hold_keys.add(self._keys.alt_gr)

        self._toggle_vk = ord(toggle_letter.upper())
        self._ctrl_down = False
        self._toggle_alt_down = False
        self._toggle_key_down = False

        self._lock = threading.Lock()
        self._held = False
        self._toggled_on = False
        self._active = False

        self._listener = pynput_keyboard.Listener(on_press=self._on_press, on_release=self._on_release)

    def start(self) -> None:
        self._listener.start()
        self._listener.wait()

    def stop(self) -> None:
        self._listener.stop()

    def _on_press(self, key) -> None:
        if key in self._hold_keys:
            if not self._held:
                self._held = True
                self._recompute()
            return
        if key in (self._keys.ctrl_l, self._keys.ctrl_r):
            self._ctrl_down = True
            return
        if key == self._keys.alt_l:
            self._toggle_alt_down = True
            return
        if getattr(key, "vk", None) == self._toggle_vk:
            if self._ctrl_down and self._toggle_alt_down and not self._toggle_key_down:
                self._toggle_key_down = True
                self._toggled_on = not self._toggled_on
                self._recompute()

    def _on_release(self, key) -> None:
        if key in self._hold_keys:
            if self._held:
                self._held = False
                self._recompute()
            return
        if key in (self._keys.ctrl_l, self._keys.ctrl_r):
            self._ctrl_down = False
            return
        if key == self._keys.alt_l:
            self._toggle_alt_down = False
            return
        if getattr(key, "vk", None) == self._toggle_vk:
            self._toggle_key_down = False

    def _recompute(self) -> None:
        with self._lock:
            active = self._held or self._toggled_on
            if active and not self._active:
                self._active = True
                self._on_start()
            elif not active and self._active:
                self._active = False
                self._on_stop()
