"""
answers.py

Spoken replies and timers. Command badges are built before the work runs,
so a reply whose text is only known afterwards is returned as "SAY:<sentence>".
The UI speaks that sentence and shows it without the prefix.
"""

from __future__ import annotations

import ctypes
import os
import threading
from datetime import datetime
from typing import Callable, Optional

_lock = threading.Lock()
_timer: Optional[threading.Timer] = None
_on_done: Optional[Callable[[], None]] = None


def spoken(sentence: str) -> str:
    return f"SAY:{sentence.strip()}"


def set_done_callback(callback: Callable[[], None]) -> None:
    global _on_done
    _on_done = callback


def language() -> str:
    return "de" if (os.environ.get("STT_LANGUAGE") or "en").lower().startswith("de") else "en"


def current_time_sentence(lang: str, now: Optional[datetime] = None) -> str:
    moment = now or datetime.now()
    if lang == "de":
        if moment.minute == 0:
            return f"Es ist {moment.hour} Uhr."
        return f"Es ist {moment.hour} Uhr {moment.minute}."
    clock = moment.strftime("%I:%M %p").lstrip("0")
    return f"It's {clock}."


def battery_sentence(lang: str, percent: Optional[int], charging: Optional[bool]) -> str:
    if percent is None:
        return "Ich sehe keinen Akku." if lang == "de" else "I can't see a battery."
    if lang == "de":
        extra = " und lädt" if charging else ""
        return f"Der Akku ist bei {percent} Prozent{extra}."
    state = " and charging" if charging else ""
    return f"Battery is at {percent} percent{state}."


def now_playing_sentence(lang: str, track: Optional[str]) -> str:
    if not track:
        return "Es läuft gerade nichts." if lang == "de" else "Nothing is playing."
    if lang == "de":
        return f"Gerade läuft {track}."
    return f"Now playing {track}."


def timer_done_label() -> str:
    return "[TIMER DONE]"


def timer_set_sentence(seconds: int, lang: str) -> str:
    amount, unit = _timer_amount(seconds, lang)
    if lang == "de":
        return f"Timer auf {amount} {unit}."
    return f"Timer set for {amount} {unit}."


def timer_cancel_sentence(lang: str, cancelled: bool) -> str:
    if lang == "de":
        return "Timer abgebrochen." if cancelled else "Es läuft kein Timer."
    return "Timer cancelled." if cancelled else "No timer is running."


def read_battery() -> tuple[Optional[int], Optional[bool]]:
    """(percent 0–100 or None, charging or None when unknown)."""

    class SYSTEM_POWER_STATUS(ctypes.Structure):
        _fields_ = [
            ("ACLineStatus", ctypes.c_byte),
            ("BatteryFlag", ctypes.c_byte),
            ("BatteryLifePercent", ctypes.c_byte),
            ("SystemStatusFlag", ctypes.c_byte),
            ("BatteryLifeTime", ctypes.c_ulong),
            ("BatteryFullLifeTime", ctypes.c_ulong),
        ]

    status = SYSTEM_POWER_STATUS()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return None, None
    percent = status.BatteryLifePercent
    if percent < 0 or percent > 100:
        return None, None
    if status.ACLineStatus == 1:
        charging: Optional[bool] = True
    elif status.ACLineStatus == 0:
        charging = False
    else:
        charging = None
    return int(percent), charging


def tell_time() -> str:
    return spoken(current_time_sentence(language()))


def tell_battery() -> str:
    percent, charging = read_battery()
    return spoken(battery_sentence(language(), percent, charging))


def tell_now_playing() -> str:
    import spotify_control as sp

    return spoken(now_playing_sentence(language(), sp.current_track_label()))


def start_timer(seconds: int) -> str:
    global _timer
    seconds = max(1, min(int(seconds), 24 * 3600))
    with _lock:
        if _timer is not None:
            _timer.cancel()
        timer = threading.Timer(seconds, _fire_timer)
        timer.daemon = True
        _timer = timer
        timer.start()
    print(f"[timer] {seconds}s")
    return spoken(timer_set_sentence(seconds, language()))


def cancel_timer() -> str:
    global _timer
    with _lock:
        running = _timer is not None
        if _timer is not None:
            _timer.cancel()
            _timer = None
    return spoken(timer_cancel_sentence(language(), running))


def _fire_timer() -> None:
    global _timer
    with _lock:
        _timer = None
        callback = _on_done
    print("[timer] done")
    if callback is not None:
        callback()


def _timer_amount(seconds: int, lang: str) -> tuple[int, str]:
    if seconds % 3600 == 0:
        count = seconds // 3600
        if lang == "de":
            return count, "Stunde" if count == 1 else "Stunden"
        return count, "hour" if count == 1 else "hours"
    if seconds % 60 == 0:
        count = seconds // 60
        if lang == "de":
            return count, "Minute" if count == 1 else "Minuten"
        return count, "minute" if count == 1 else "minutes"
    if lang == "de":
        return seconds, "Sekunde" if seconds == 1 else "Sekunden"
    return seconds, "second" if seconds == 1 else "seconds"
