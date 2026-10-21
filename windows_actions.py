"""
windows_actions.py

Native Windows automation primitives used by the voice assistant. Every
function is synchronous and safe to call from a worker thread (never call
these directly from the Qt UI thread — dispatch through a ThreadPoolExecutor,
see main.py::ActionExecutor).

DRY_RUN=1 (env var) makes every action log what it *would* do instead of
touching the OS. simulate_demo.py sets this automatically unless --live is
passed.
"""

from __future__ import annotations

import ctypes
from ctypes import wintypes
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
import webbrowser
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

DRY_RUN = os.environ.get("DRY_RUN", "0") == "1"

# ---------------------------------------------------------------------------
# App resolution
# ---------------------------------------------------------------------------

# Known install locations checked in order; first hit wins. Kept short and
# explicit rather than a fuzzy PATH search, since these are the apps named in
# the product brief.
_ARC_CANDIDATES = [
    Path(os.environ.get("LOCALAPPDATA", "")) / "Arc" / "Arc.exe",
    Path(os.environ.get("PROGRAMFILES", "")) / "Arc" / "Arc.exe",
]

_CHROME_CANDIDATES = [
    Path(os.environ.get("PROGRAMFILES", "")) / "Google" / "Chrome" / "Application" / "chrome.exe",
    Path(os.environ.get("PROGRAMFILES(X86)", "")) / "Google" / "Chrome" / "Application" / "chrome.exe",
    Path(os.environ.get("LOCALAPPDATA", "")) / "Google" / "Chrome" / "Application" / "chrome.exe",
]

# Every key resolves to one of the handler branches in open_app() below.
# Multi-word phrases ("notes app", "file explorer") are matched here for
# direct resolve_app() lookups (e.g. from the LLM fallback); the regex path
# in intent_engine.py already matches them via its own word-boundary
# patterns without needing the exact multi-word key.
_APP_ALIASES = {
    "notes": "notepad",
    "note": "notepad",
    "notes app": "notepad",
    "notepad": "notepad",
    "terminal": "terminal",
    "cmd": "terminal",
    "console": "terminal",
    "arc": "arc",
    "browser": "browser",
    "chrome": "chrome",
    "google chrome": "chrome",
    "x": "x",
    "twitter": "x",
    "camera": "camera",
    "photo booth": "camera",
    "photobooth": "camera",
    "calculator": "calc",
    "calc": "calc",
    "code": "code",
    "vs code": "code",
    "vscode": "code",
    "visual studio code": "code",
    "file explorer": "explorer",
    "explorer": "explorer",
    "files": "explorer",
    "dateien": "explorer",
    "rechner": "calc",
    "taschenrechner": "calc",
    "kamera": "camera",
    "konsole": "terminal",
    "notizblock": "notepad",
}


class ActionError(RuntimeError):
    """Raised when a requested automation cannot be completed."""


def _run(cmd: list[str] | str, **kwargs) -> None:
    if DRY_RUN:
        print(f"[DRY_RUN] would run: {cmd}")
        return
    if isinstance(cmd, str):
        os.startfile(cmd)  # noqa: S606 - intentional, user-initiated
    else:
        subprocess.Popen(cmd, **kwargs)  # noqa: S603 - trusted local commands


def _open_browser(url: str) -> None:
    if DRY_RUN:
        print(f"[DRY_RUN] would open in default browser: {url}")
        return
    webbrowser.open_new_tab(url)


def resolve_app(name: str) -> str:
    """Map a spoken app name to a handler key. Names with no dedicated
    handler in open_app() are returned as-is (normalised) and end up in the
    generic launcher, which looks them up on PATH / in the Start Menu /
    among installed Store apps."""
    # Multi-word aliases ("visual studio code", "file explorer") are matched
    # by regex with \s+ between words, so whitespace can vary - collapse it
    # before the exact-string dict lookup.
    key = re.sub(r"\s+", " ", name.strip().lower())
    return _APP_ALIASES.get(key, key)


_START_MENU_DIRS = [
    Path(os.environ.get("APPDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs",
    Path(os.environ.get("PROGRAMDATA", "")) / "Microsoft" / "Windows" / "Start Menu" / "Programs",
]


def _find_start_menu_shortcut(name: str) -> Optional[Path]:
    """Best-matching .lnk in the user's + system Start Menu, or None.

    Exact (case-insensitive) basename match wins; otherwise a shortcut whose
    basename starts with the spoken name ("firefox" -> "Firefox.lnk" rather
    than "Firefox Privater Modus.lnk"); otherwise one that merely contains
    it. Uninstaller shortcuts are skipped - "open spotify" must never hit
    "Uninstall Spotify.lnk".
    """
    needle = name.lower()
    exact: list[Path] = []
    prefix: list[Path] = []
    contains: list[Path] = []
    for root in _START_MENU_DIRS:
        if not root.is_dir():
            continue
        for lnk in root.rglob("*.lnk"):
            base = lnk.stem.lower()
            if "uninstall" in base or "deinstall" in base:
                continue
            if base == needle:
                exact.append(lnk)
            elif base.startswith(needle):
                prefix.append(lnk)
            elif needle in base:
                contains.append(lnk)
    for bucket in (exact, prefix, contains):
        if bucket:
            # Shortest name = closest match within the bucket.
            return min(bucket, key=lambda p: len(p.stem))
    return None


def _find_store_app_id(name: str) -> Optional[str]:
    """AppUserModelID of an installed (Store/packaged) app whose Start Menu
    name contains `name`, via PowerShell's Get-StartApps. ~1s, so only used
    as the last resort."""
    if DRY_RUN:
        return None
    script = (
        "Get-StartApps | Where-Object { $_.Name -like '*" + name.replace("'", "''") + "*' } "
        "| Sort-Object { $_.Name.Length } | Select-Object -First 1 -ExpandProperty AppID"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
            capture_output=True, text=True, timeout=8, creationflags=subprocess.CREATE_NO_WINDOW,
        )
    except (OSError, subprocess.TimeoutExpired):
        return None
    app_id = out.stdout.strip()
    return app_id or None


def _launch_generic(name: str) -> str:
    """Open an app that has no dedicated handler. Tries, in order: an
    executable on PATH, a Start Menu shortcut, an installed Store app."""
    pretty = name.title()

    exe = shutil.which(name) or shutil.which(name.replace(" ", ""))
    if exe:
        _run([exe])
        return f"Opening {pretty}"

    lnk = _find_start_menu_shortcut(name)
    if lnk is not None:
        _run(str(lnk))  # os.startfile resolves the shortcut like a double-click
        return f"Opening {lnk.stem}"

    app_id = _find_store_app_id(name)
    if app_id:
        _run(["explorer.exe", f"shell:AppsFolder\\{app_id}"])
        return f"Opening {pretty}"

    raise ActionError(f"Couldn't find an app called {name!r}")


def open_app(app_name: str) -> str:
    """Open (or focus) a friendly-named application. Returns a short status string."""
    resolved = resolve_app(app_name)

    if resolved == "notepad":
        _run(["notepad.exe"])
        return "Opening Notes"

    if resolved == "terminal":
        wt = shutil.which("wt.exe") or shutil.which("wt")
        if wt:
            _run([wt])
        else:
            _run(["cmd.exe"])
        return "Opening Terminal"

    if resolved == "arc":
        for candidate in _ARC_CANDIDATES:
            if candidate.exists():
                _run([str(candidate)])
                return "Opening Arc"
        # Arc not installed on this machine — fall back to default browser.
        _open_browser("https://arc.net")
        return "Arc not found, opening arc.net instead"

    if resolved == "chrome":
        for candidate in _CHROME_CANDIDATES:
            if candidate.exists():
                _run([str(candidate)])
                return "Opening Chrome"
        exe = shutil.which("chrome")
        if exe:
            _run([exe])
            return "Opening Chrome"
        _open_browser("about:blank")
        return "Chrome not found, opening default browser instead"

    if resolved == "browser":
        _open_browser("about:blank")
        return "Opening browser"

    if resolved == "x":
        navigate_url("https://x.com")
        return "Opening X"

    if resolved == "camera":
        _run("microsoft.windows.camera:")
        return "Opening Camera"

    if resolved == "calc":
        _run(["calc.exe"])
        return "Opening Calculator"

    if resolved == "code":
        exe = shutil.which("code.cmd") or shutil.which("code")
        if exe:
            _run([exe])
            return "Opening VS Code"
        raise ActionError("VS Code ('code') not found on PATH")

    if resolved == "explorer":
        _run(["explorer.exe"])
        return "Opening File Explorer"

    # Anything else (spotify, firefox, discord, ...): no hand-written
    # handler, look it up on the system instead.
    return _launch_generic(resolved)


# ---------------------------------------------------------------------------
# Closing apps
# ---------------------------------------------------------------------------

# Process image names (without .exe) per resolved app key, for apps whose
# process isn't simply called like the app. Anything not listed here is
# matched generically: any running process whose name contains the spoken
# word (spaces removed) - "steam" -> steam.exe + steamwebhelper.exe.
_CLOSE_PROCESS_NAMES = {
    "terminal": ["WindowsTerminal"],
    "calc": ["CalculatorApp", "Calculator"],
    "camera": ["WindowsCamera"],
    "code": ["Code"],
    "edge": ["msedge"],
    "x": [],  # a tab, not a process - nothing sensible to close
    "browser": [],
}

# Never touched by the generic matcher, no matter what was said: OS core
# processes, plus our own interpreter.
_PROTECTED_PROCESSES = {
    "system", "registry", "smss", "csrss", "wininit", "winlogon", "services",
    "lsass", "svchost", "dwm", "fontdrvhost", "sihost", "taskhostw", "ctfmon",
    "runtimebroker", "startmenuexperiencehost", "shellexperiencehost",
    "searchhost", "textinputhost", "audiodg", "spoolsv", "dllhost", "conhost",
    "wmiprvse", "msmpeng", "securityhealthservice", "securityhealthsystray",
    "explorer", "python", "pythonw",
}

_CLOSE_GRACE_S = 2.5


def _running_image_names() -> set[str]:
    """Lower-cased image names (without .exe) of all running processes."""
    out = subprocess.run(
        ["tasklist", "/FO", "CSV", "/NH"],
        capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW,
    )
    names: set[str] = set()
    for line in out.stdout.splitlines():
        if not line.startswith('"'):
            continue
        image = line.split('","', 1)[0].strip('"')
        names.add(re.sub(r"\.exe$", "", image, flags=re.IGNORECASE).lower())
    return names


def _taskkill(images: list[str], force: bool) -> None:
    cmd = ["taskkill"] + (["/F"] if force else [])
    for image in images:
        cmd += ["/IM", f"{image}.exe"]
    subprocess.run(cmd, capture_output=True, creationflags=subprocess.CREATE_NO_WINDOW)


def _close_explorer_windows() -> str:
    # explorer.exe is also the desktop/taskbar - killing it is never what
    # "close file explorer" means. Close the folder windows via the shell's
    # COM automation instead.
    script = (
        "$w = (New-Object -ComObject Shell.Application).Windows(); "
        "$n = 0; foreach ($x in @($w)) { try { $x.Quit(); $n++ } catch {} }; $n"
    )
    out = subprocess.run(
        ["powershell", "-NoProfile", "-NonInteractive", "-Command", script],
        capture_output=True, text=True, timeout=8, creationflags=subprocess.CREATE_NO_WINDOW,
    )
    count = out.stdout.strip() or "0"
    return f"Closed {count} Explorer window(s)" if count != "0" else "No Explorer windows open"


def close_app(app_name: str) -> str:
    """Close every running instance of a friendly-named application.

    Asks politely first (plain taskkill sends WM_CLOSE, so apps get to save
    state), then force-kills whatever is still alive after _CLOSE_GRACE_S -
    needed for apps like Steam or Spotify that otherwise just minimise to
    the tray instead of exiting.
    """
    resolved = resolve_app(app_name)

    if resolved == "explorer":
        if DRY_RUN:
            print("[DRY_RUN] would close File Explorer windows")
            return "Closing File Explorer"
        return _close_explorer_windows()

    running = _running_image_names()
    if resolved in _CLOSE_PROCESS_NAMES:
        wanted = [p.lower() for p in _CLOSE_PROCESS_NAMES[resolved] if p.lower() in running]
        if not _CLOSE_PROCESS_NAMES[resolved]:
            raise ActionError(f"Can't close {app_name!r} - it isn't an application")
    else:
        needle = resolved.replace(" ", "")
        wanted = sorted(
            name for name in running
            if needle in name and name not in _PROTECTED_PROCESSES
        )

    if not wanted:
        return f"{app_name.title()} isn't running"

    if DRY_RUN:
        print(f"[DRY_RUN] would close processes: {wanted}")
        return f"Closing {app_name.title()}"

    _taskkill(wanted, force=False)
    deadline = time.time() + _CLOSE_GRACE_S
    while time.time() < deadline:
        time.sleep(0.25)
        if not (set(wanted) & _running_image_names()):
            return f"Closed {app_name.title()}"
    survivors = sorted(set(wanted) & _running_image_names())
    _taskkill(survivors, force=True)
    return f"Closed {app_name.title()}"


# ---------------------------------------------------------------------------
# Text injection
# ---------------------------------------------------------------------------

# Modifier virtual-key codes: generic + left/right variants for Alt, Ctrl, Shift.
_MODIFIER_VKS = (0x12, 0xA4, 0xA5, 0x11, 0xA2, 0xA3, 0x10, 0xA0, 0xA1)
_KEYEVENTF_EXTENDEDKEY = 0x0001
_KEYEVENTF_KEYUP = 0x0002


def _release_modifier_keys() -> None:
    """Force-clears the OS's key-state for Alt/Ctrl/Shift.

    Confirmed via live reproduction: if the PTT hold key (Right Alt) is
    still logically down when keystrokes get injected right after, every
    injected letter is read as an Alt+<letter> accelerator instead of typed
    text - in a tabbed app like modern Notepad this drove File-menu/Open
    behavior and pulled in unrelated recently-used files, not just opened a
    menu.

    Right-side modifiers (Right Alt, Right Ctrl) are "extended" keys - a
    KEYUP without KEYEVENTF_EXTENDEDKEY set doesn't reliably clear the same
    internal state that a real (or pynput-synthesized) press of the
    extended key set, confirmed by a follow-up repro where a single
    unflagged KEYUP left the paste still going nowhere. Sending both
    variants for every vk covers whichever one is actually the active bit,
    since releasing an already-up key is a harmless no-op regardless of
    flags.
    """
    for vk in _MODIFIER_VKS:
        ctypes.windll.user32.keybd_event(vk, 0, _KEYEVENTF_KEYUP, 0)
        ctypes.windll.user32.keybd_event(vk, 0, _KEYEVENTF_EXTENDEDKEY | _KEYEVENTF_KEYUP, 0)


def _get_clipboard_text() -> Optional[str]:
    import win32clipboard

    try:
        win32clipboard.OpenClipboard()
        try:
            if win32clipboard.IsClipboardFormatAvailable(win32clipboard.CF_UNICODETEXT):
                return win32clipboard.GetClipboardData(win32clipboard.CF_UNICODETEXT)
            return None
        finally:
            win32clipboard.CloseClipboard()
    except Exception:
        return None


def _set_clipboard_text(text: str) -> None:
    import win32clipboard

    win32clipboard.OpenClipboard()
    try:
        win32clipboard.EmptyClipboard()
        win32clipboard.SetClipboardText(text, win32clipboard.CF_UNICODETEXT)
    finally:
        win32clipboard.CloseClipboard()


def write_in_app(text: str, title: Optional[str] = None) -> str:
    """
    Type `text` into the target app via clipboard-paste (Ctrl+V), not
    character-by-character keystroke injection.

    Clipboard-paste replaces the old SendKeys-style approach for two
    reasons confirmed via live reproduction against a real Notepad window:
    (1) if a modifier is still logically held when keys are injected, every
    injected letter is read as an Alt+<letter> accelerator - a single
    Ctrl+V is far less exposed to that than dozens of individual
    keystrokes; (2) it types with 100% fidelity, with no risk of a
    character in the dictated text (`%^+~{}()`) being misread as SendKeys
    syntax.

    If `title` is given, focuses the first top-level window whose title
    contains it and pastes into whatever tab is currently active there.
    Otherwise defaults to Notepad, launching it fresh if none is open, and
    always opens a brand-new tab (Ctrl+T) before pasting - an ambient,
    already-running Notepad window can have the user's own real files open
    in other tabs, and we must never type into one of those by accident.
    """
    if DRY_RUN:
        print(f"[DRY_RUN] would type into {title or 'Notepad'}: {text!r}")
        return "Typed (dry run)"

    from pywinauto import Application, findwindows  # local import: optional dep

    _release_modifier_keys()
    time.sleep(0.15)  # let the OS settle the key-up before injecting anything

    target_title = title or "Notepad"
    matches = findwindows.find_windows(title_re=f".*{target_title}.*", top_level_only=True)

    if matches:
        app = Application(backend="uia").connect(handle=matches[0])
        window = app.top_window()
    else:
        app = Application(backend="uia").start("notepad.exe")
        time.sleep(0.6)  # let the process create its main window
        window = app.top_window()

    try:
        import win32gui

        win32gui.SetForegroundWindow(window.handle)
    except Exception:
        # Best-effort: Windows' foreground-lock can refuse this from a
        # non-activating background process. set_focus() below is the
        # primary mechanism and has its own internal fallback for that.
        pass
    window.set_focus()
    time.sleep(0.05)
    # A held Alt can leave the window in menu/accelerator-navigation mode
    # even after the OS's own modifier key-state is cleared - confirmed via
    # a follow-up repro where the paste still landed nowhere despite
    # _release_modifier_keys(). Escape is the standard way to force a
    # window out of that mode regardless of which input subsystem (classic
    # Win32 vs. a modern app's own accelerator handling) is tracking it.
    window.type_keys("{ESC}", pause=0)

    if title is None:
        window.type_keys("^t", pause=0)  # new blank tab - never paste into an ambient real file
        time.sleep(0.15)

    # Re-release right before injecting the paste itself: set_focus()/
    # SetForegroundWindow can (on some Windows versions) synthesize their
    # own Alt keypress internally to defeat the foreground-lock, so the
    # earlier release alone isn't guaranteed to still hold by this point.
    _release_modifier_keys()
    window.type_keys("{ESC}", pause=0)

    previous_clipboard = _get_clipboard_text()
    _set_clipboard_text(text)
    try:
        window.type_keys("^v", pause=0)
        time.sleep(0.1)
    finally:
        if previous_clipboard is not None:
            _set_clipboard_text(previous_clipboard)

    return "Typed text"


# ---------------------------------------------------------------------------
# Browser control
# ---------------------------------------------------------------------------

_BROWSER_COMMANDS = {
    "edge": "msedge",
    "firefox": "firefox",
}

# Browsers that aren't reliably on PATH get an explicit install-path search,
# same as open_app()'s _ARC_CANDIDATES/_CHROME_CANDIDATES.
_BROWSER_CANDIDATES = {
    "arc": _ARC_CANDIDATES,
    "chrome": _CHROME_CANDIDATES,
}


def _open_url_in_browser(url: str, browser: str = "default") -> None:
    browser = browser.lower()

    if browser in ("default", ""):
        _open_browser(url)
        return

    candidates = _BROWSER_CANDIDATES.get(browser)
    if candidates is not None:
        for candidate in candidates:
            if candidate.exists():
                _run([str(candidate), url])
                return
        _open_browser(url)
        return

    exe = _BROWSER_COMMANDS.get(browser)
    if exe and shutil.which(exe):
        _run([exe, url])
        return

    # Unknown/uninstalled browser name — don't fail the whole action.
    _open_browser(url)


def browser_search(query: str, browser: str = "default") -> str:
    """Launch/focus `browser` and search for `query` using the default search engine."""
    from urllib.parse import quote_plus

    search_url = f"https://www.google.com/search?q={quote_plus(query)}"
    _open_url_in_browser(search_url, browser)
    return f"Searching for '{query}'"


def navigate_url(url: str) -> str:
    """Navigate straight to `url`, adding a scheme if the caller omitted one."""
    if "://" not in url:
        url = f"https://{url}"
    _open_url_in_browser(url)
    return f"Navigating to {url}"


# ---------------------------------------------------------------------------
# Camera
# ---------------------------------------------------------------------------

@dataclass
class PhotoResult:
    path: Optional[Path]
    message: str


def take_photo_countdown(seconds: int = 3, on_tick=None, save_dir: Optional[Path] = None) -> PhotoResult:
    """
    Count down from `seconds`, calling on_tick(remaining) each second (the UI
    layer uses this to drive the pill's visual countdown), then capture a
    single frame from the default webcam via OpenCV and save it as a JPEG.
    """
    for remaining in range(seconds, 0, -1):
        if on_tick:
            on_tick(remaining)
        if not DRY_RUN:
            time.sleep(1)

    if DRY_RUN:
        print("[DRY_RUN] would capture photo from webcam")
        return PhotoResult(path=None, message="Captured (dry run)")

    import cv2

    capture = cv2.VideoCapture(0, cv2.CAP_DSHOW)
    try:
        if not capture.isOpened():
            raise ActionError("No webcam available")
        ok, frame = capture.read()
        if not ok:
            raise ActionError("Failed to read a frame from the webcam")
    finally:
        capture.release()

    out_dir = save_dir or (Path.home() / "Pictures" / "VoiceAssistant")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"photo_{int(time.time())}.jpg"
    cv2.imwrite(str(out_path), frame)
    return PhotoResult(path=out_path, message=f"Saved photo to {out_path}")


# ---------------------------------------------------------------------------
# System volume (default playback device — not Spotify)
# ---------------------------------------------------------------------------

# Each "computer louder" / "computer quieter" step. Spotify's own nudge is
# separate and stays on the music commands.
SYSTEM_VOLUME_STEP = 10

_CLSCTX_ALL = 23
_E_RENDER = 0
_E_CONSOLE = 0
_COINIT_APARTMENTTHREADED = 0x2
_RPC_E_CHANGED_MODE = 0x80010106

_com_ready = threading.local()
_ole32 = None
_user32_dll = None


class GUID(ctypes.Structure):
    _fields_ = [
        ("Data1", ctypes.c_ulong),
        ("Data2", ctypes.c_ushort),
        ("Data3", ctypes.c_ushort),
        ("Data4", ctypes.c_ubyte * 8),
    ]


def _guid(text: str) -> GUID:
    return GUID.from_buffer_copy(uuid.UUID(text).bytes_le)


def _ole32_dll():
    global _ole32
    if _ole32 is None:
        dll = ctypes.WinDLL("ole32", use_last_error=True)
        dll.CoInitializeEx.argtypes = [ctypes.c_void_p, ctypes.c_ulong]
        dll.CoInitializeEx.restype = ctypes.c_long
        dll.CoCreateInstance.argtypes = [
            ctypes.POINTER(GUID),
            ctypes.c_void_p,
            ctypes.c_ulong,
            ctypes.POINTER(GUID),
            ctypes.POINTER(ctypes.c_void_p),
        ]
        dll.CoCreateInstance.restype = ctypes.c_long
        _ole32 = dll
    return _ole32


def _user32():
    global _user32_dll
    if _user32_dll is None:
        dll = ctypes.WinDLL("user32", use_last_error=True)
        dll.GetForegroundWindow.restype = ctypes.c_void_p
        dll.IsWindowVisible.argtypes = [ctypes.c_void_p]
        dll.IsWindowVisible.restype = ctypes.c_int
        dll.GetWindowTextLengthW.argtypes = [ctypes.c_void_p]
        dll.GetWindowTextLengthW.restype = ctypes.c_int
        dll.GetWindowTextW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int]
        dll.GetWindowTextW.restype = ctypes.c_int
        dll.GetClassNameW.argtypes = [ctypes.c_void_p, ctypes.c_wchar_p, ctypes.c_int]
        dll.GetClassNameW.restype = ctypes.c_int
        dll.GetWindowThreadProcessId.argtypes = [ctypes.c_void_p, ctypes.POINTER(wintypes.DWORD)]
        dll.GetWindowThreadProcessId.restype = wintypes.DWORD
        dll.EnumWindows.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        dll.EnumWindows.restype = ctypes.c_int
        dll.GetWindowRect.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
        dll.GetWindowRect.restype = ctypes.c_int
        dll.ShowWindow.argtypes = [ctypes.c_void_p, ctypes.c_int]
        dll.ShowWindow.restype = ctypes.c_int
        dll.SetForegroundWindow.argtypes = [ctypes.c_void_p]
        dll.SetForegroundWindow.restype = ctypes.c_int
        dll.BringWindowToTop.argtypes = [ctypes.c_void_p]
        dll.BringWindowToTop.restype = ctypes.c_int
        dll.AttachThreadInput.argtypes = [wintypes.DWORD, wintypes.DWORD, ctypes.c_int]
        dll.AttachThreadInput.restype = ctypes.c_int
        dll.keybd_event.argtypes = [ctypes.c_ubyte, ctypes.c_ubyte, ctypes.c_ulong, ctypes.c_void_p]
        _user32_dll = dll
    return _user32_dll


def _ensure_com() -> None:
    if getattr(_com_ready, "ok", False):
        return
    hr = _ole32_dll().CoInitializeEx(None, _COINIT_APARTMENTTHREADED)
    if hr < 0 and (hr & 0xFFFFFFFF) != _RPC_E_CHANGED_MODE:
        raise ActionError(f"COM init failed ({hr & 0xFFFFFFFF:08X})")
    _com_ready.ok = True


def _com_method(ptr: int, index: int, restype, argtypes):
    vtbl = ctypes.cast(ctypes.c_void_p(ptr), ctypes.POINTER(ctypes.c_void_p))[0]
    fn = ctypes.cast(vtbl, ctypes.POINTER(ctypes.c_void_p))[index]
    return ctypes.WINFUNCTYPE(restype, ctypes.c_void_p, *argtypes)(fn)


def _release(ptr: int) -> None:
    if not ptr:
        return
    _com_method(ptr, 2, ctypes.c_ulong, [])(ptr)


def _check_hr(hr: int, what: str) -> None:
    if hr < 0:
        raise ActionError(f"{what} failed ({hr & 0xFFFFFFFF:08X})")


class _MasterVolume:
    """IAudioEndpointVolume on the default render device."""

    def __init__(self) -> None:
        self._ptrs: list[int] = []
        try:
            self._endpoint = self._open()
        except Exception:
            self.close()
            raise

    def _open(self) -> int:
        _ensure_com()
        ole32 = _ole32_dll()
        clsid = _guid("BCDE0395-E52F-467C-8E3D-C4579291692E")
        iid_enum = _guid("A95664D2-9614-4F35-A746-DE8DB63617E6")
        enumerator = ctypes.c_void_p()
        hr = ole32.CoCreateInstance(
            ctypes.byref(clsid), None, _CLSCTX_ALL, ctypes.byref(iid_enum), ctypes.byref(enumerator)
        )
        _check_hr(hr, "Audio device enumerator")
        if not enumerator.value:
            raise ActionError("No audio device enumerator")
        self._ptrs.append(enumerator.value)

        device = ctypes.c_void_p()
        get_default = _com_method(
            enumerator.value, 4, ctypes.c_long, [ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_void_p)]
        )
        hr = get_default(enumerator.value, _E_RENDER, _E_CONSOLE, ctypes.byref(device))
        _check_hr(hr, "Default playback device")
        if not device.value:
            raise ActionError("No default playback device")
        self._ptrs.append(device.value)

        iid_volume = _guid("5CDF2C82-841E-4546-9722-0CF74078229A")
        endpoint = ctypes.c_void_p()
        activate = _com_method(
            device.value,
            3,
            ctypes.c_long,
            [ctypes.POINTER(GUID), ctypes.c_ulong, ctypes.c_void_p, ctypes.POINTER(ctypes.c_void_p)],
        )
        hr = activate(device.value, ctypes.byref(iid_volume), _CLSCTX_ALL, None, ctypes.byref(endpoint))
        _check_hr(hr, "Volume control")
        if not endpoint.value:
            raise ActionError("Volume control unavailable")
        self._ptrs.append(endpoint.value)
        return endpoint.value

    def close(self) -> None:
        for ptr in reversed(self._ptrs):
            _release(ptr)
        self._ptrs.clear()

    def __enter__(self) -> "_MasterVolume":
        return self

    def __exit__(self, *args) -> None:
        self.close()

    def get_scalar(self) -> float:
        level = ctypes.c_float()
        get_level = _com_method(self._endpoint, 9, ctypes.c_long, [ctypes.POINTER(ctypes.c_float)])
        _check_hr(get_level(self._endpoint, ctypes.byref(level)), "Read volume")
        return float(level.value)

    def set_scalar(self, level: float) -> None:
        set_level = _com_method(self._endpoint, 7, ctypes.c_long, [ctypes.c_float, ctypes.c_void_p])
        _check_hr(set_level(self._endpoint, ctypes.c_float(level), None), "Set volume")

    def set_mute(self, muted: bool) -> None:
        set_mute = _com_method(self._endpoint, 14, ctypes.c_long, [ctypes.c_int, ctypes.c_void_p])
        _check_hr(set_mute(self._endpoint, 1 if muted else 0, None), "Set mute")

    def get_mute(self) -> bool:
        muted = ctypes.c_int()
        get_mute = _com_method(self._endpoint, 15, ctypes.c_long, [ctypes.POINTER(ctypes.c_int)])
        _check_hr(get_mute(self._endpoint, ctypes.byref(muted)), "Read mute")
        return bool(muted.value)


def _clamp_percent(percent: int) -> int:
    return max(0, min(100, int(percent)))


def get_system_volume() -> int:
    """Current master volume, 0–100. Returns 0 in dry-run."""
    if DRY_RUN:
        return 0
    with _MasterVolume() as volume:
        return _clamp_percent(round(volume.get_scalar() * 100))


def set_system_volume(percent: int) -> str:
    """Set the PC master volume. A level above 0 also unmutes."""
    percent = _clamp_percent(percent)
    if DRY_RUN:
        print(f"[DRY_RUN] would set system volume to {percent}")
        return f"System volume {percent}%"
    with _MasterVolume() as volume:
        volume.set_scalar(percent / 100.0)
        if percent > 0:
            volume.set_mute(False)
    return f"System volume {percent}%"


def change_system_volume(delta: int) -> str:
    """Nudge the PC master volume by `delta` percent."""
    if DRY_RUN:
        direction = "up" if delta > 0 else "down"
        print(f"[DRY_RUN] would turn system volume {direction}")
        return f"System volume {direction}"
    return set_system_volume(get_system_volume() + delta)


def set_system_mute(muted: bool) -> str:
    if DRY_RUN:
        print(f"[DRY_RUN] would set system mute={muted}")
        return "Muted" if muted else "Unmuted"
    with _MasterVolume() as volume:
        volume.set_mute(muted)
    return "Muted" if muted else "Unmuted"


# ---------------------------------------------------------------------------
# Screenshot
# ---------------------------------------------------------------------------


def take_screenshot(save_dir: Optional[Path] = None) -> str:
    """Grab every monitor and save a PNG under Pictures\\Screenshots."""
    out_dir = save_dir or (Path.home() / "Pictures" / "Screenshots")
    stamp = time.strftime("%Y%m%d_%H%M%S")
    out_path = out_dir / f"screenshot_{stamp}.png"
    suffix = 2
    while out_path.exists():
        out_path = out_dir / f"screenshot_{stamp}_{suffix}.png"
        suffix += 1
    if DRY_RUN:
        print(f"[DRY_RUN] would save screenshot to {out_path}")
        return f"Saved screenshot to {out_path}"

    try:
        from PIL import ImageGrab
    except ImportError as exc:
        raise ActionError("Screenshot needs Pillow (installed with pyautogui)") from exc

    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        image = ImageGrab.grab(all_screens=True)
    except TypeError:
        image = ImageGrab.grab()
    image.save(out_path)
    return f"Saved screenshot to {out_path}"


# ---------------------------------------------------------------------------
# Foreground window
# ---------------------------------------------------------------------------

_SW_MAXIMIZE = 3
_SW_MINIMIZE = 6
_SW_RESTORE = 9
_VK_LEFT = 0x25
_VK_RIGHT = 0x27
_VK_LWIN = 0x5B
_SKIP_WINDOW_CLASSES = {"Progman", "WorkerW", "Shell_TrayWnd", "Shell_SecondaryTrayWnd"}


def _window_title(hwnd: int) -> str:
    user32 = _user32()
    length = user32.GetWindowTextLengthW(hwnd)
    buf = ctypes.create_unicode_buffer(length + 1)
    user32.GetWindowTextW(hwnd, buf, length + 1)
    return buf.value.strip()


def _window_class(hwnd: int) -> str:
    buf = ctypes.create_unicode_buffer(256)
    _user32().GetClassNameW(hwnd, buf, 256)
    return buf.value


def _is_controllable(hwnd: int) -> bool:
    if not hwnd:
        return False
    user32 = _user32()
    if not user32.IsWindowVisible(hwnd):
        return False
    if _window_class(hwnd) in _SKIP_WINDOW_CLASSES:
        return False
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if pid.value == os.getpid():
        return False
    return bool(_window_title(hwnd))


def _foreground_hwnd() -> int:
    """The window the user is working in, skipping the shell and this process."""
    user32 = _user32()
    hwnd = user32.GetForegroundWindow()
    if _is_controllable(hwnd):
        return int(hwnd)
    chosen: list[int] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def visit(handle, _lparam):
        if _is_controllable(handle):
            chosen.append(int(handle))
            return False
        return True

    user32.EnumWindows(visit, 0)
    if not chosen:
        raise ActionError("No window to control")
    return chosen[0]


def _focus_window(hwnd: int) -> None:
    user32 = _user32()
    foreground = user32.GetForegroundWindow()
    if foreground == hwnd:
        return
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    current = kernel32.GetCurrentThreadId()
    fg_pid = wintypes.DWORD()
    fg_thread = user32.GetWindowThreadProcessId(foreground, ctypes.byref(fg_pid)) if foreground else 0
    if fg_thread:
        user32.AttachThreadInput(current, fg_thread, True)
    user32.BringWindowToTop(hwnd)
    user32.SetForegroundWindow(hwnd)
    if fg_thread:
        user32.AttachThreadInput(current, fg_thread, False)


def _press_win_arrow(vk_arrow: int) -> None:
    # PTT holds Right Alt. A Win+arrow injected while Alt is still down is a
    # different shortcut, so clear modifiers the same way typing does.
    _release_modifier_keys()
    user32 = _user32()
    user32.keybd_event(_VK_LWIN, 0, 0, 0)
    user32.keybd_event(vk_arrow, 0, _KEYEVENTF_EXTENDEDKEY, 0)
    user32.keybd_event(vk_arrow, 0, _KEYEVENTF_EXTENDEDKEY | _KEYEVENTF_KEYUP, 0)
    user32.keybd_event(_VK_LWIN, 0, _KEYEVENTF_KEYUP, 0)


def _show_foreground(command: int, dry_message: str, result: str) -> str:
    if DRY_RUN:
        print(f"[DRY_RUN] would {dry_message}")
        return result
    _user32().ShowWindow(_foreground_hwnd(), command)
    return result


def minimize_window() -> str:
    return _show_foreground(_SW_MINIMIZE, "minimize the foreground window", "Minimized")


def maximize_window() -> str:
    return _show_foreground(_SW_MAXIMIZE, "maximize the foreground window", "Maximized")


def restore_window() -> str:
    return _show_foreground(_SW_RESTORE, "restore the foreground window", "Restored")


def snap_window(side: str) -> str:
    """Snap the foreground window to the left or right half of the screen."""
    side = side.lower()
    if side not in ("left", "right"):
        raise ActionError(f"Unknown snap side: {side}")
    if DRY_RUN:
        print(f"[DRY_RUN] would snap the foreground window {side}")
        return f"Snapped {side}"
    hwnd = _foreground_hwnd()
    _focus_window(hwnd)
    _press_win_arrow(_VK_LEFT if side == "left" else _VK_RIGHT)
    return f"Snapped {side}"


def is_foreground_fullscreen() -> bool:
    """Best-effort check so the overlay can dim/hide itself over fullscreen apps."""
    user32 = ctypes.windll.user32
    hwnd = user32.GetForegroundWindow()
    if not hwnd:
        return False

    class RECT(ctypes.Structure):
        _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                    ("right", ctypes.c_long), ("bottom", ctypes.c_long)]
    win_rect = RECT()
    user32.GetWindowRect(hwnd, ctypes.byref(win_rect))
    screen_w = user32.GetSystemMetrics(0)
    screen_h = user32.GetSystemMetrics(1)
    return (win_rect.right - win_rect.left) >= screen_w and (win_rect.bottom - win_rect.top) >= screen_h


# ---------------------------------------------------------------------------
# Focus a window, open a known folder, find a file
# ---------------------------------------------------------------------------

_FOLDERS = {
    "downloads": "Downloads",
    "download": "Downloads",
    "documents": "Documents",
    "dokumente": "Documents",
    "desktop": "Desktop",
    "schreibtisch": "Desktop",
    "pictures": "Pictures",
    "bilder": "Pictures",
    "music": "Music",
    "musik": "Music",
    "videos": "Videos",
}

_FILE_SKIP_DIRS = {"node_modules", ".git", "__pycache__", "appdata", "windows"}


def open_folder(name: str) -> str:
    """Open Downloads, Documents, Desktop, Pictures, Music, or Videos."""
    key = name.strip().lower()
    folder = _FOLDERS.get(key)
    if folder is None:
        raise ActionError(f"Unknown folder: {name}")
    path = Path.home() / folder
    if DRY_RUN:
        print(f"[DRY_RUN] would open folder {path}")
        return f"Opened {folder}"
    if not path.exists():
        raise ActionError(f"{folder} is not there")
    _run(str(path))
    return f"Opened {folder}"


def focus_window(name: str) -> str:
    """Bring a window to the front. Launches the app when none is open."""
    import answers

    cleaned = name.strip()
    spoken_name = cleaned.title()
    lang_de = answers.language() == "de"
    if DRY_RUN:
        print(f"[DRY_RUN] would focus {cleaned}")
        sentence = f"{spoken_name} ist vorn." if lang_de else f"{spoken_name} is in front."
        return answers.spoken(sentence)
    hwnd = _find_window(cleaned)
    if hwnd is None:
        open_app(cleaned)
        sentence = f"{spoken_name} wird geöffnet." if lang_de else f"Opening {spoken_name}."
        return answers.spoken(sentence)
    _focus_window(hwnd)
    sentence = f"{spoken_name} ist vorn." if lang_de else f"{spoken_name} is in front."
    return answers.spoken(sentence)


def find_file(query: str, roots: Optional[list[Path]] = None) -> str:
    """Open the best name match under Desktop, Documents, and Downloads."""
    needle = query.strip().strip("\"'").lower()
    if len(needle) < 2:
        raise ActionError("Say more of the file name")
    search_roots = roots or [Path.home() / folder for folder in ("Desktop", "Documents", "Downloads")]
    best: Optional[Path] = None
    best_score = -1
    seen = 0
    for root in search_roots:
        if not root.exists():
            continue
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d.lower() not in _FILE_SKIP_DIRS and not d.startswith(".")]
            for filename in filenames:
                seen += 1
                if seen > 4000:
                    break
                score = _file_score(filename, needle)
                if score > best_score:
                    best_score = score
                    best = Path(dirpath) / filename
            if seen > 4000:
                break
        if seen > 4000:
            break
    if best is None or best_score < 0:
        raise ActionError(f"No file matching {query}")
    if DRY_RUN:
        print(f"[DRY_RUN] would open file {best}")
        return f"Opened {best.name}"
    _run(str(best))
    return f"Opened {best.name}"


def _file_score(filename: str, needle: str) -> int:
    lowered = filename.lower()
    stem = Path(filename).stem.lower()
    if stem == needle or lowered == needle:
        return 100
    if stem.startswith(needle):
        return 80
    if needle in lowered:
        return 60
    return -1


def _window_exe(hwnd: int) -> str:
    user32 = _user32()
    pid = wintypes.DWORD()
    user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    if not pid.value:
        return ""
    kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
    kernel32.OpenProcess.argtypes = [ctypes.c_ulong, ctypes.c_int, ctypes.c_ulong]
    kernel32.OpenProcess.restype = ctypes.c_void_p
    kernel32.CloseHandle.argtypes = [ctypes.c_void_p]
    kernel32.QueryFullProcessImageNameW.argtypes = [
        ctypes.c_void_p,
        ctypes.c_ulong,
        ctypes.c_wchar_p,
        ctypes.POINTER(wintypes.DWORD),
    ]
    kernel32.QueryFullProcessImageNameW.restype = ctypes.c_int
    handle = kernel32.OpenProcess(0x1000, False, pid.value)
    if not handle:
        return ""
    try:
        size = wintypes.DWORD(512)
        buf = ctypes.create_unicode_buffer(512)
        if not kernel32.QueryFullProcessImageNameW(handle, 0, buf, ctypes.byref(size)):
            return ""
        return Path(buf.value).stem.lower()
    finally:
        kernel32.CloseHandle(handle)


def _window_area(hwnd: int) -> int:
    class RECT(ctypes.Structure):
        _fields_ = [
            ("left", ctypes.c_long),
            ("top", ctypes.c_long),
            ("right", ctypes.c_long),
            ("bottom", ctypes.c_long),
        ]

    rect = RECT()
    if not _user32().GetWindowRect(hwnd, ctypes.byref(rect)):
        return 0
    return max(0, rect.right - rect.left) * max(0, rect.bottom - rect.top)


def _find_window(name: str) -> Optional[int]:
    needle = name.strip().lower()
    if not needle:
        return None
    user32 = _user32()
    found: list[tuple[int, int]] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def visit(handle, _lparam):
        if not _is_controllable(handle):
            return True
        title = _window_title(handle).lower()
        exe = _window_exe(handle)
        words = set(re.split(r"[^a-z0-9äöüß+]+", title))
        score = 0
        if exe == needle:
            score = 100
        elif needle in words:
            score = 80
        elif needle in title:
            score = 50
        if score:
            found.append((score * 1_000_000 + _window_area(handle), int(handle)))
        return True

    user32.EnumWindows(visit, 0)
    if not found:
        return None
    found.sort(reverse=True)
    return found[0][1]
