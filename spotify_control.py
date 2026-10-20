"""
spotify_control.py

Spotify playback control for the voice assistant, via the Spotify Web API
(Authorization Code + PKCE, so no client secret is needed on this machine).

Setup (one-off):
  1. Create a free app at https://developer.spotify.com/dashboard, add
     http://127.0.0.1:8888/callback as a Redirect URI, copy the Client ID.
  2. Put it in .env as SPOTIFY_CLIENT_ID=...
  3. On the first "play ..." command a browser tab opens asking you to
     approve the app. The resulting refresh token is cached in
     %LOCALAPPDATA%\\WinVoiceAssistant\\spotify_token.json and reused from
     then on - no further interaction needed.

Playback control (play/pause/next/volume/...) requires Spotify Premium; the
API answers 403 for free accounts. Without a configured Client ID the
transport commands (pause/resume/next/previous) fall back to the Windows
media keys, which work with whatever player is in the foreground - but
"play <song>" opens Spotify search and plays the first result via keystrokes
(no UI Automation - that hung the whole app on Spotify's Electron window).

All functions are synchronous and meant to run on the IntentEngine's worker
thread, never on the Qt thread (the auth flow blocks for up to 2 minutes
waiting for the browser).
"""

from __future__ import annotations

import base64
import ctypes
import hashlib
import json
import os
import re
import secrets
import socket
import subprocess
import threading
import time
import urllib.parse
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from typing import Optional

import windows_actions as wa

SCOPES = "user-modify-playback-state user-read-playback-state"
DEFAULT_REDIRECT_URI = "http://127.0.0.1:8888/callback"
TOKEN_PATH = Path(os.environ.get("LOCALAPPDATA", ".")) / "WinVoiceAssistant" / "spotify_token.json"
AUTH_TIMEOUT_S = 120
DEVICE_WAIT_S = 20  # how long to wait for a freshly launched Spotify to show up as a device

_API = "https://api.spotify.com/v1"
_lock = threading.Lock()


class SpotifyError(wa.ActionError):
    """Raised for anything the user should see on the pill."""


# ---------------------------------------------------------------------------
# Media-key fallback (no API needed)
# ---------------------------------------------------------------------------

_VK_MEDIA_NEXT_TRACK = 0xB0
_VK_MEDIA_PREV_TRACK = 0xB1
_VK_MEDIA_PLAY_PAUSE = 0xB3
_KEYEVENTF_EXTENDEDKEY = 0x0001
_KEYEVENTF_KEYUP = 0x0002


def _press_media_key(vk: int) -> None:
    if wa.DRY_RUN:
        print(f"[DRY_RUN] would press media key 0x{vk:X}")
        return
    user32 = ctypes.windll.user32
    user32.keybd_event(vk, 0, 0, 0)
    user32.keybd_event(vk, 0, _KEYEVENTF_KEYUP, 0)


# ---------------------------------------------------------------------------
# Auth (PKCE)
# ---------------------------------------------------------------------------

def _client_id() -> str:
    return os.environ.get("SPOTIFY_CLIENT_ID", "").strip()


def _redirect_uri() -> str:
    return os.environ.get("SPOTIFY_REDIRECT_URI", "").strip() or DEFAULT_REDIRECT_URI


def is_configured() -> bool:
    return bool(_client_id())


def _load_token() -> Optional[dict]:
    try:
        return json.loads(TOKEN_PATH.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _save_token(tok: dict) -> None:
    TOKEN_PATH.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_PATH.write_text(json.dumps(tok), encoding="utf-8")


class _CallbackHandler(BaseHTTPRequestHandler):
    code: Optional[str] = None
    error: Optional[str] = None

    def do_GET(self) -> None:  # noqa: N802 - http.server API
        query = urllib.parse.parse_qs(urllib.parse.urlparse(self.path).query)
        _CallbackHandler.code = (query.get("code") or [None])[0]
        _CallbackHandler.error = (query.get("error") or [None])[0]
        body = (
            "<html><body style='font-family:sans-serif;text-align:center;margin-top:3em'>"
            + ("<h2>Jarvis is connected to Spotify.</h2>You can close this tab."
               if _CallbackHandler.code else f"<h2>Spotify login failed</h2>{_CallbackHandler.error}")
            + "</body></html>"
        ).encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args) -> None:  # silence the default stderr access log
        pass


def _authorize_interactively() -> dict:
    """Full PKCE dance: open the browser, catch the redirect, swap the code
    for tokens. Blocks until the user approves or AUTH_TIMEOUT_S passes."""
    import requests

    verifier = base64.urlsafe_b64encode(secrets.token_bytes(64)).rstrip(b"=").decode()
    challenge = base64.urlsafe_b64encode(hashlib.sha256(verifier.encode()).digest()).rstrip(b"=").decode()
    state = secrets.token_urlsafe(16)
    redirect = _redirect_uri()
    parsed = urllib.parse.urlparse(redirect)

    params = {
        "client_id": _client_id(),
        "response_type": "code",
        "redirect_uri": redirect,
        "scope": SCOPES,
        "code_challenge_method": "S256",
        "code_challenge": challenge,
        "state": state,
    }
    url = "https://accounts.spotify.com/authorize?" + urllib.parse.urlencode(params)

    _CallbackHandler.code = None
    _CallbackHandler.error = None
    server = HTTPServer((parsed.hostname or "127.0.0.1", parsed.port or 80), _CallbackHandler)
    server.timeout = 1.0
    print(f"[spotify] opening browser for Spotify login ({AUTH_TIMEOUT_S}s timeout)")
    webbrowser.open(url)
    deadline = time.time() + AUTH_TIMEOUT_S
    try:
        while time.time() < deadline and _CallbackHandler.code is None and _CallbackHandler.error is None:
            server.handle_request()
    finally:
        server.server_close()

    if _CallbackHandler.error:
        raise SpotifyError(f"Spotify login refused: {_CallbackHandler.error}")
    if not _CallbackHandler.code:
        raise SpotifyError("Spotify login timed out - say the command again and approve in the browser")

    resp = requests.post(
        "https://accounts.spotify.com/api/token",
        data={
            "grant_type": "authorization_code",
            "code": _CallbackHandler.code,
            "redirect_uri": redirect,
            "client_id": _client_id(),
            "code_verifier": verifier,
        },
        timeout=10,
    )
    if resp.status_code != 200:
        raise SpotifyError(f"Spotify token exchange failed: {resp.status_code} {resp.text[:120]}")
    tok = resp.json()
    tok["expires_at"] = time.time() + tok.get("expires_in", 3600) - 30
    _save_token(tok)
    print("[spotify] connected, token cached at", TOKEN_PATH)
    return tok


def _refresh(tok: dict) -> dict:
    import requests

    resp = requests.post(
        "https://accounts.spotify.com/api/token",
        data={
            "grant_type": "refresh_token",
            "refresh_token": tok["refresh_token"],
            "client_id": _client_id(),
        },
        timeout=10,
    )
    if resp.status_code != 200:
        # Refresh token revoked/expired -> force a fresh interactive login.
        TOKEN_PATH.unlink(missing_ok=True)
        raise SpotifyError("Spotify session expired - say the command again to log in")
    new = resp.json()
    new.setdefault("refresh_token", tok["refresh_token"])
    new["expires_at"] = time.time() + new.get("expires_in", 3600) - 30
    _save_token(new)
    return new


def _access_token() -> str:
    if not is_configured():
        raise SpotifyError("Set SPOTIFY_CLIENT_ID in .env to control Spotify")
    with _lock:
        tok = _load_token()
        if tok is None:
            tok = _authorize_interactively()
        elif time.time() >= tok.get("expires_at", 0):
            tok = _refresh(tok)
        return tok["access_token"]


# ---------------------------------------------------------------------------
# API helpers
# ---------------------------------------------------------------------------

def _request(method: str, path: str, *, params: Optional[dict] = None, body: Optional[dict] = None,
             allow_404: bool = False):
    import requests

    headers = {"Authorization": f"Bearer {_access_token()}"}
    resp = requests.request(method, _API + path, headers=headers, params=params, json=body, timeout=10)
    if resp.status_code == 401:
        # Token rejected despite not being expired - drop it and retry once.
        TOKEN_PATH.unlink(missing_ok=True)
        headers = {"Authorization": f"Bearer {_access_token()}"}
        resp = requests.request(method, _API + path, headers=headers, params=params, json=body, timeout=10)
    if resp.status_code == 403:
        detail = ""
        try:
            detail = resp.json().get("error", {}).get("reason") or resp.json().get("error", {}).get("message", "")
        except ValueError:
            pass
        if "PREMIUM" in detail.upper():
            raise SpotifyError("Spotify Premium is required for playback control")
        raise SpotifyError(f"Spotify refused: {detail or resp.status_code}")
    if resp.status_code == 404 and allow_404:
        return None
    if resp.status_code >= 400:
        raise SpotifyError(f"Spotify error {resp.status_code}: {resp.text[:120]}")
    if resp.status_code == 204 or not resp.content:
        return None
    return resp.json()


def _devices() -> list[dict]:
    data = _request("GET", "/me/player/devices") or {}
    return data.get("devices", [])


def _pick_device(devices: list[dict]) -> Optional[dict]:
    """This PC's Spotify app if it's listed, else whatever is active, else
    the first computer, else anything."""
    host = socket.gethostname().lower()
    for d in devices:
        if d.get("type") == "Computer" and d.get("name", "").lower() == host:
            return d
    for d in devices:
        if d.get("is_active"):
            return d
    for d in devices:
        if d.get("type") == "Computer":
            return d
    return devices[0] if devices else None


def _computer_device(devices: list[dict]) -> Optional[dict]:
    """This PC if Spotify reports it, otherwise any computer. Phones and
    speakers are ignored so playback doesn't vanish onto another device."""
    host = socket.gethostname().lower()
    computers = [d for d in devices if d.get("type") == "Computer"]
    for d in computers:
        if d.get("name", "").lower() == host:
            return d
    return computers[0] if computers else None


def _ensure_device() -> dict:
    """Returns this PC's Spotify Connect device, launching the desktop app
    and waiting for it to show up if it isn't online yet."""
    device = _computer_device(_devices())
    if device is None:
        print("[spotify] no computer device online - launching Spotify")
        wa.open_app("spotify")
        deadline = time.time() + DEVICE_WAIT_S
        while device is None and time.time() < deadline:
            time.sleep(1.0)
            device = _computer_device(_devices())
    if device is None:
        raise SpotifyError(
            "Spotify on this PC isn't online. Open the Spotify app, make sure "
            "you're logged into the same account, and try again"
        )
    print(f"[spotify] device: {device.get('name')} ({device.get('type')})")
    return device


def _play_on(device: dict, body: Optional[dict] = None) -> None:
    """Make this PC the active Spotify Connect target, then start playback."""
    device_id = device["id"]
    _request("PUT", "/me/player", body={"device_ids": [device_id], "play": False})
    time.sleep(0.4)
    _request("PUT", "/me/player/play", params={"device_id": device_id}, body=body)
    if (device.get("volume_percent") or 100) < 15:
        try:
            _request(
                "PUT",
                "/me/player/volume",
                params={"volume_percent": 50, "device_id": device_id},
            )
        except SpotifyError as exc:
            print(f"[spotify] volume: {exc}")


# ---------------------------------------------------------------------------
# Public actions
# ---------------------------------------------------------------------------

_TYPE_PREFIX = re.compile(r"^(?:the\s+)?(album|playlist|artist|song|track)\s+(.+)$", re.IGNORECASE)
_ON_SPOTIFY = re.compile(r"\s+(?:on|in|with)\s+spotify$", re.IGNORECASE)
_SOME = re.compile(r"^(?:some|a\s+bit\s+of|me\s+some)\s+", re.IGNORECASE)


def _parse_query(raw: str) -> tuple[str, str]:
    """"the album abbey road" -> ("album", "abbey road"); "fire force
    opening" -> ("track", "fire force opening"); "some lo-fi" -> playlist
    search (that phrasing wants a mood/genre, not one specific song)."""
    q = _ON_SPOTIFY.sub("", raw.strip())
    m = _TYPE_PREFIX.match(q)
    if m:
        kind = m.group(1).lower()
        return ("track" if kind in ("song", "track") else kind), m.group(2).strip()
    if _SOME.match(q):
        return "playlist", _SOME.sub("", q).strip()
    return "track", q


_VK_TAB = 0x09
_VK_RETURN = 0x0D
_VK_ESCAPE = 0x1B
_VK_DOWN = 0x28
_VK_MENU = 0x12  # Alt - used to defeat Windows' foreground lock
_SW_RESTORE = 9


def _spotify_pids() -> list[int]:
    out = subprocess.run(
        ["tasklist", "/FI", "IMAGENAME eq Spotify.exe", "/FO", "CSV", "/NH"],
        capture_output=True, text=True, creationflags=subprocess.CREATE_NO_WINDOW,
    )
    pids: list[int] = []
    for line in out.stdout.splitlines():
        parts = line.split('","')
        if len(parts) >= 2:
            try:
                pids.append(int(parts[1].strip('"')))
            except ValueError:
                pass
    return pids


def _visible_hwnds() -> list[int]:
    user32 = ctypes.windll.user32
    hwnds: list[int] = []

    @ctypes.WINFUNCTYPE(ctypes.c_bool, ctypes.c_void_p, ctypes.c_void_p)
    def _cb(hwnd, _unused):
        if user32.IsWindowVisible(hwnd):
            hwnds.append(int(hwnd))
        return True

    user32.EnumWindows(_cb, 0)
    return hwnds


def _hwnd_pid(hwnd: int) -> int:
    pid = ctypes.c_ulong(0)
    ctypes.windll.user32.GetWindowThreadProcessId(hwnd, ctypes.byref(pid))
    return pid.value


def _hwnd_area(hwnd: int) -> int:
    class RECT(ctypes.Structure):
        _fields_ = [("left", ctypes.c_long), ("top", ctypes.c_long),
                    ("right", ctypes.c_long), ("bottom", ctypes.c_long)]

    rect = RECT()
    if not ctypes.windll.user32.GetWindowRect(hwnd, ctypes.byref(rect)):
        return 0
    return max(0, rect.right - rect.left) * max(0, rect.bottom - rect.top)


def _spotify_main_hwnd() -> Optional[int]:
    """Largest visible top-level window owned by Spotify.exe. Title matching
    is useless once a track is playing (title becomes 'Song - Artist')."""
    pids = set(_spotify_pids())
    if not pids:
        return None
    best: Optional[int] = None
    best_area = 0
    for hwnd in _visible_hwnds():
        if _hwnd_pid(hwnd) not in pids:
            continue
        area = _hwnd_area(hwnd)
        if area > best_area:
            best, best_area = hwnd, area
    return best


def _bring_to_front(hwnd: int) -> bool:
    """Best-effort activate without UI Automation. An Alt tap first is the
    usual way to get past Windows' foreground-lock from a background process."""
    user32 = ctypes.windll.user32
    user32.ShowWindow(hwnd, _SW_RESTORE)
    user32.keybd_event(_VK_MENU, 0, 0, 0)
    user32.keybd_event(_VK_MENU, 0, _KEYEVENTF_KEYUP, 0)
    return bool(user32.SetForegroundWindow(hwnd))


def _tap(*vks: int) -> None:
    user32 = ctypes.windll.user32
    for vk in vks:
        flags = _KEYEVENTF_EXTENDEDKEY if vk in (_VK_DOWN,) else 0
        user32.keybd_event(vk, 0, flags, 0)
        user32.keybd_event(vk, 0, flags | _KEYEVENTF_KEYUP, 0)
        time.sleep(0.06)


def _wait_spotify_hwnd(timeout_s: float) -> Optional[int]:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        hwnd = _spotify_main_hwnd()
        if hwnd and _hwnd_area(hwnd) > 200 * 200:
            return hwnd
        time.sleep(0.25)
    return _spotify_main_hwnd()


_SPOTIFY_TRACK_ID = re.compile(
    r"(?:spotify:track:|open\.spotify\.com/track/)([A-Za-z0-9]{22})"
)


_COVERISH = re.compile(
    r"\b(cover|karaoke|tribute|8[-\s]?bit|lo[-\s]?fi|piano\s+version|epic\s+finale)\b",
    re.IGNORECASE,
)
_GENERIC_THEME = re.compile(r"\b(opening|ending|theme|main theme)\b", re.IGNORECASE)


def _deezer_score(title: str, artist: str, query: str) -> int:
    """Prefer the real song name (Inferno, Guren no Yumiya) over YouTube-style
    covers titled 'Attack on Titan Opening' by random artists - those often
    have no MusicBrainz/Spotify mapping, which is why AOT failed live."""
    blob = f"{title} {artist}"
    score = 0
    if _COVERISH.search(blob):
        score -= 40
    if _GENERIC_THEME.search(query) and not _GENERIC_THEME.search(title):
        score += 25
    if _GENERIC_THEME.search(title):
        score -= 15
    return score


def _deezer_hit(q: str) -> Optional[tuple[str, str, Optional[str]]]:
    """Best Deezer hit as (title, artist, isrc). No API key."""
    import requests

    data = requests.get(
        "https://api.deezer.com/search", params={"q": q, "limit": 8}, timeout=8
    ).json()
    items = data.get("data") or []
    if not items:
        return None
    ranked = sorted(
        items,
        key=lambda i: _deezer_score(
            i.get("title") or "",
            ((i.get("artist") or {}).get("name") or ""),
            q,
        ),
        reverse=True,
    )
    pick = ranked[0]
    title = (pick.get("title") or "").strip()
    artist = ((pick.get("artist") or {}).get("name") or "").strip()
    if not title:
        return None
    isrc = None
    try:
        full = requests.get(f"https://api.deezer.com/track/{pick['id']}", timeout=8).json()
        isrc = (full.get("isrc") or "").strip() or None
    except Exception as exc:
        print(f"[spotify] deezer track lookup failed: {exc}")
    print(f"[spotify] deezer picked {title!r} - {artist!r} (isrc={isrc})")
    return title, artist, isrc


_MB_HEADERS = {
    "User-Agent": "WinVoiceAssistant/1.0 (local voice assistant)",
    "Accept": "application/json",
}


def _mb_spotify_from_recording(mbid: str) -> Optional[str]:
    import requests

    time.sleep(1.05)
    rel = requests.get(
        f"https://musicbrainz.org/ws/2/recording/{mbid}",
        params={"inc": "url-rels", "fmt": "json"},
        headers=_MB_HEADERS,
        timeout=8,
    ).json()
    for item in rel.get("relations") or []:
        url = ((item.get("url") or {}).get("resource") or "")
        match = _SPOTIFY_TRACK_ID.search(url)
        if match:
            return f"spotify:track:{match.group(1)}"
    return None


def _mb_search_recordings(query: str) -> list[dict]:
    import requests

    return (
        requests.get(
            "https://musicbrainz.org/ws/2/recording/",
            params={"query": query, "fmt": "json"},
            headers=_MB_HEADERS,
            timeout=8,
        ).json().get("recordings")
        or []
    )


def _mb_spotify_uri(isrc: str) -> Optional[str]:
    """MusicBrainz ISRC -> open.spotify.com/track/ID. 1 req/s courtesy limit."""
    recs = _mb_search_recordings(f"isrc:{isrc}")
    if not recs:
        return None
    return _mb_spotify_from_recording(recs[0]["id"])


def _mb_spotify_uri_by_names(title: str, artist: str) -> Optional[str]:
    query = f'recording:"{title}"'
    if artist:
        query += f' AND artist:"{artist}"'
    recs = _mb_search_recordings(query)
    if not recs:
        return None
    return _mb_spotify_from_recording(recs[0]["id"])


def _ddg_spotify_track_uri(q: str) -> Optional[str]:
    """First open.spotify.com/track link DuckDuckGo returns for `q`."""
    import requests

    html = requests.get(
        "https://lite.duckduckgo.com/lite/",
        params={"q": f"site:open.spotify.com/track {q}"},
        headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"},
        timeout=8,
    ).text
    ids = _SPOTIFY_TRACK_ID.findall(html)
    return f"spotify:track:{ids[0]}" if ids else None


_RESOLVE_CACHE: dict[str, tuple[str, str]] = {}


def _resolve_spotify_track(q: str) -> Optional[tuple[str, str]]:
    """Map a spoken query to (spotify:track:ID, 'Title - Artist')."""
    key = q.strip().lower()
    if key in _RESOLVE_CACHE:
        return _RESOLVE_CACHE[key]
    label = q
    search_q = q
    isrc = None
    try:
        hit = _deezer_hit(q)
    except Exception as exc:
        print(f"[spotify] deezer lookup failed: {exc}")
        hit = None
    if hit:
        title, artist, isrc = hit
        label = f"{title} - {artist}" if artist else title
        search_q = f"{title} {artist}".strip()
    uri = None
    if isrc:
        try:
            uri = _mb_spotify_uri(isrc)
        except Exception as exc:
            print(f"[spotify] musicbrainz lookup failed: {exc}")
            uri = None
    if uri is None and hit:
        try:
            uri = _mb_spotify_uri_by_names(title, artist)
        except Exception as exc:
            print(f"[spotify] musicbrainz name lookup failed: {exc}")
            uri = None
    if uri:
        _RESOLVE_CACHE[key] = (uri, label)
        return uri, label
    for candidate in (search_q, q):
        try:
            uri = _ddg_spotify_track_uri(candidate)
        except Exception as exc:
            print(f"[spotify] track lookup failed: {exc}")
            uri = None
        if uri:
            _RESOLVE_CACHE[key] = (uri, label)
            return uri, label
    return None


def _play_via_desktop_search(q: str) -> str:
    """No Client ID: open Spotify's search URI, then walk focus out of the
    search box onto the first result and press Enter.

    Must not use pywinauto/UIA here - connecting to Spotify's Electron
    accessibility tree hangs the whole interpreter (GIL held in COM), which
    froze Whisper and the overlay after the first 'play' command.
    """
    uri = f"spotify:search:{urllib.parse.quote(q)}"
    if wa.DRY_RUN:
        print(f"[DRY_RUN] would open {uri} and play the first result")
        return f"Playing {q}"
    wa._run(uri)
    hwnd = _wait_spotify_hwnd(6.0)
    if hwnd is None:
        return f"Opened Spotify search for '{q}'"
    _bring_to_front(hwnd)
    time.sleep(0.9)  # search page needs a beat to populate results
    if not _bring_to_front(hwnd):
        return f"Opened Spotify search for '{q}' - click the first result"
    wa._release_modifier_keys()
    # Search URI leaves the caret in the search box; Enter there only
    # re-runs the search (exactly what we saw live). Tab/Down lands on
    # the first track row, then Enter plays it.
    _tap(_VK_ESCAPE, _VK_TAB, _VK_TAB, _VK_DOWN, _VK_RETURN)
    return f"Playing {q}"


def _open_and_play_track(uri: str) -> None:
    """Show `uri` in the desktop app and actually start it.

    `os.startfile('spotify:track:…')` only navigates to the song page more
    often than not - that's why Inferno sometimes played and Dororo/others
    only 'found' the track. A :play suffix plus Enter on the track page
    (the green Play button is the default action) is what kicks playback
    without UI Automation.
    """
    if wa.DRY_RUN:
        print(f"[DRY_RUN] would open and play {uri}")
        return
    wa._run(uri)
    if uri.startswith("spotify:track:") and not uri.endswith(":play"):
        wa._run(uri + ":play")
    hwnd = _wait_spotify_hwnd(5.0)
    if hwnd is None:
        print("[spotify] Spotify window not found after opening track")
        return
    time.sleep(1.2)
    if not _bring_to_front(hwnd):
        print("[spotify] couldn't focus Spotify to press Play")
        return
    wa._release_modifier_keys()
    _tap(_VK_ESCAPE, _VK_RETURN)
    print("[spotify] sent Play on the track page")


def _play_first_search_hit(q: str) -> str:
    """Last resort: open search and try to play the first row."""
    search = f"spotify:search:{urllib.parse.quote(q)}"
    print(f"[spotify] no track URI for {q!r} - opening {search}")
    if wa.DRY_RUN:
        print(f"[DRY_RUN] would open {search} and play first hit")
        return f"Opened Spotify search for '{q}'"
    wa._run(search)
    hwnd = _wait_spotify_hwnd(5.0)
    if hwnd is None:
        return f"Opened Spotify search for '{q}'"
    time.sleep(1.6)
    if _bring_to_front(hwnd):
        wa._release_modifier_keys()
        _tap(_VK_ESCAPE, _VK_TAB, _VK_TAB, _VK_DOWN, _VK_RETURN)
    return f"Opened Spotify search for '{q}'"


def play(query: str) -> str:
    """"play <query>": search Spotify and start playback on this PC."""
    kind, q = _parse_query(query)
    if not q:
        return resume()

    if not is_configured():
        # Resolve a real track URI and hand it to the desktop app - that
        # actually starts playback. The old search-URI + keystrokes path
        # only re-ran the search because focus stayed in the search box.
        resolved = _resolve_spotify_track(q)
        if resolved:
            uri, label = resolved
            print(f"[spotify] {uri}  ({label})")
            _open_and_play_track(uri)
            return f"Playing {label}"
        return _play_first_search_hit(q)

    if wa.DRY_RUN:
        print(f"[DRY_RUN] would search spotify {kind}={q!r} and play it")
        return f"Playing {q}"

    data = _request("GET", "/search", params={"q": q, "type": kind, "limit": 1}) or {}
    items = (data.get(kind + "s") or {}).get("items") or []
    if not items:
        raise SpotifyError(f"Nothing found on Spotify for '{q}'")
    item = items[0]
    uri = item["uri"]
    if kind == "track":
        artists = ", ".join(a["name"] for a in item.get("artists", [])[:2])
        label = f"{item['name']} - {artists}" if artists else item["name"]
        body = {"uris": [uri]}
    else:
        owner = item.get("artists", [{}])[0].get("name") if kind == "album" else item.get("owner", {}).get("display_name")
        label = f"{item['name']} ({owner})" if owner else item["name"]
        body = {"context_uri": uri}

    # The Web API answers 204 and then does not start audio on this desktop
    # client (verified: player stays idle). Opening the track URI is what
    # the installed Spotify app actually plays.
    print(f"[spotify] {uri}  ({label})")
    if wa.DRY_RUN:
        print(f"[DRY_RUN] would open {uri}")
        return f"Playing {label}"
    wa._run(uri)
    return f"Playing {label}"


def pause() -> str:
    # Media key, not the Web API: on this desktop client the API returns 204
    # and leaves playback unchanged.
    _press_media_key(_VK_MEDIA_PLAY_PAUSE)
    return "Paused"


def resume() -> str:
    _press_media_key(_VK_MEDIA_PLAY_PAUSE)
    return "Resumed"


def next_track() -> str:
    _press_media_key(_VK_MEDIA_NEXT_TRACK)
    return "Next track"


def previous_track() -> str:
    _press_media_key(_VK_MEDIA_PREV_TRACK)
    return "Previous track"


def set_volume(percent: int) -> str:
    percent = max(0, min(100, int(percent)))
    if wa.DRY_RUN:
        print(f"[DRY_RUN] would set spotify volume to {percent}")
        return f"Volume {percent}%"
    if not is_configured():
        raise SpotifyError("Volume control needs SPOTIFY_CLIENT_ID in .env")
    _request("PUT", "/me/player/volume", params={"volume_percent": percent})
    return f"Volume {percent}%"


def change_volume(delta: int) -> str:
    """"louder" / "quieter": nudge relative to the current level."""
    if wa.DRY_RUN:
        print(f"[DRY_RUN] would change spotify volume by {delta:+d}")
        return "Volume changed"
    if not is_configured():
        raise SpotifyError("Volume control needs SPOTIFY_CLIENT_ID in .env")
    state = _request("GET", "/me/player", allow_404=True) or {}
    current = (state.get("device") or {}).get("volume_percent")
    if current is None:
        raise SpotifyError("Nothing is playing")
    return set_volume(current + delta)


def set_shuffle(on: bool) -> str:
    if wa.DRY_RUN:
        print(f"[DRY_RUN] would set spotify shuffle={on}")
        return f"Shuffle {'on' if on else 'off'}"
    if not is_configured():
        raise SpotifyError("Shuffle control needs SPOTIFY_CLIENT_ID in .env")
    _request("PUT", "/me/player/shuffle", params={"state": "true" if on else "false"})
    return f"Shuffle {'on' if on else 'off'}"
