"""
intent_engine.py

Consumes interim/final transcript events and fires Windows actions the
instant a clear imperative is recognized - it does not wait for the
utterance to end. A regex tokenizer handles the common commands with near
zero latency; an optional LLM classifier (Claude Haiku by default, Gemini
Flash as an alternative) is only consulted when the regex finds nothing,
since it costs a network round trip.

Anti-replay: each utterance has an id. Once an action's signature has fired
for a given utterance id, it will not fire again for that same utterance -
this is what stops the mid-sentence trigger and the later finalized-
transcript pass from double-executing the same command.
"""

from __future__ import annotations

import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import Callable, Optional

import answers
import macros
import spotify_control as sp
import windows_actions as wa

ActionCallback = Callable[[str], None]  # display label for the UI badge


@dataclass
class Action:
    name: str
    signature: str  # dedup key, e.g. "open_app:arc"
    run: Callable[[], str]  # returns a display label
    display_arg: str = ""  # human-readable arg for the confirmation badge, e.g. "arc"


# "notepad" must be its own alternative, not just "notes?" (note/notes):
# "notes?" can't match the "notepad" substring because \b after it fails
# ("notepad" has no word boundary between "note" and "pad") - this was why
# "open up notepad" previously matched nothing at all.
#
# Multi-word phrases ("visual studio code", "file explorer") must appear as
# their own alternative, not be assembled from shorter ones: alternation is
# tried at a fixed starting position (right after the trigger word), so
# "code" alone never matches when the first word there is actually "visual".
_APP_WORDS = (
    r"(notepad|notes?|arc|terminal|console|cmd|browser|chrome|firefox|edge|"
    r"spotify|discord|steam|x|twitter|"
    r"camera|photo\s?booth|calculator|calc|visual\s+studio\s+code|"
    r"vs\s+code|vscode|code|file\s+explorer|explorer|files|"
    r"rechner|taschenrechner|kamera|konsole|notizblock|dateien)"
)

# Matches "open <app>", "open up <app>", "open the <app>", "launch <app>",
# and "pull up <app>" (with an optional "the" in between). All patterns use
# .search(), so leading/trailing filler ("can you ...", "please ...",
# "... for me") is already ignored without needing to appear in the pattern.
_OPEN_TRIGGER = r"(?:open(?:\s+(?:the|up))?|launch(?:\s+the)?|pull\s+up(?:\s+the)?)"

_KNOWN_SEARCH_TARGETS = r"(chrome|edge|arc|firefox|google|x|twitter)"

# Verbs that mean "web search": "search (for) X", "look for X", "look up X",
# "google X", "find X". The optional "for"/"up" after the verb is swallowed
# so it never ends up in the query text.
# "find file …" is a local file search, so that phrase is not a web search.
_SEARCH_VERB = r"(?:search|look|google|find(?!\s+(?:the\s+)?file\b))\s+(?:for\s+|up\s+)?"

# Verbs that mean "close an app". "shut down" is deliberately excluded -
# too close to shutting down the PC; "stop" too, because "stop (the music /
# spotify)" means pause, not kill.
_CLOSE_TRIGGER = r"(?:close|closed|quit|exit|kill)(?:\s+the)?"

# Optional politeness around short, whole-utterance commands like "next"
# or "pause the music" - those are anchored (^...$) so a stray "next" inside
# a longer sentence can't trigger them.
_FILLER_PRE = r"(?:(?:please|can\s+you|could\s+you|would\s+you|jarvis)\s+)*"
_FILLER_POST = r"(?:\s+(?:please|now|jarvis))*"
_MUSIC_NOUN = (
    r"(?:\s+(?:the\s+|this\s+|that\s+)?(?:music|song|track|playback|playing|spotify|one|it))?"
)

# Only treats '.', '!', '?' as an end-of-utterance marker when followed by
# whitespace or end-of-string - NOT when embedded mid-word, as in a domain
# like "github.com". Without the lookahead, "search for python tutorials on
# github.com" would capture just "...github" and drop ".com", since the
# non-greedy capture stops at the first '.' it sees regardless of context.
_SENTENCE_END = r"(?:[.!?](?=\s|$)|$)"

# Allows an optional comma/colon right after "write" ("write, hello world"),
# since that's exactly the shape smart-formatted STT punctuation can produce
# between an imperative verb and its content.
# "schreib(e)" is the German imperative. Optional "mal"/"bitte" sits between
# the verb and the dictated text ("schreib mal Hallo") and must not be typed.
_WRITE_PATTERN = re.compile(
    rf"\b(?:write|schreib(?:e)?)\b[,:]?(?:\s+(?:mal|bitte))*\s*(.+?){_SENTENCE_END}",
    re.IGNORECASE,
)

# Spoken URLs ("animus infra dot com", "github dot com") say "dot" as a word
# instead of an actual period, so the literal-dot pattern below never
# matches them at all. TLD list is deliberately short and common rather than
# exhaustive - it only needs to cover realistic spoken commands.
_TLDS = r"com|de|io|net|org|dev|co|ai|app|info|biz|me|us|uk|in|edu|gov|xyz|tv|so"

_LITERAL_DOMAIN_FULL = re.compile(rf"[a-z0-9][\w-]*(?:\.[a-z0-9][\w-]*)*\.(?:{_TLDS})", re.IGNORECASE)
_SPOKEN_DOMAIN_FULL = re.compile(
    rf"(?:[a-z0-9-]+\s+){{0,3}}[a-z0-9-]+\s+dot\s+(?:{_TLDS})(?:\s+dot\s+(?:{_TLDS}))*", re.IGNORECASE
)


def _despeak_dots(raw: str) -> str:
    """"animus infra dot com" -> "animusinfra.com". A no-op on text that
    already has a literal dot and no spaces, so it's safe to apply
    unconditionally regardless of which alternative actually matched."""
    return re.sub(r"\s+dot\s+", ".", raw, flags=re.IGNORECASE).replace(" ", "")


def _looks_like_domain(text: str) -> Optional[str]:
    """If `text`, in full, is just a domain - spoken or literal - with
    nothing else around it, returns the normalized domain. Used to route a
    "search <query>" whose entire query is a bare domain to navigate_url
    instead of Googling the literal words. Deliberately requires a full
    match (not a search anywhere in the string): "search for python
    tutorials on github.com" should still be a real Google search, not a
    navigation, just because a domain is mentioned somewhere in it.
    """
    candidate = text.strip()
    if _SPOKEN_DOMAIN_FULL.fullmatch(candidate):
        return _despeak_dots(candidate)
    if _LITERAL_DOMAIN_FULL.fullmatch(candidate):
        return candidate.replace(" ", "")
    return None


def _clamp_spoken_percent(raw: str) -> int:
    return max(0, min(100, int(raw)))


def _system_volume_action(match: re.Match) -> Action:
    percent = _clamp_spoken_percent(match.group(1))
    return Action(
        name="system_volume",
        signature=f"system_volume:{percent}",
        run=lambda: wa.set_system_volume(percent),
        display_arg=f"{percent}%",
    )


def _system_volume_step(up: bool) -> Action:
    delta = wa.SYSTEM_VOLUME_STEP if up else -wa.SYSTEM_VOLUME_STEP
    name = "system_volume_up" if up else "system_volume_down"
    label = "UP" if up else "DOWN"
    return Action(name=name, signature=name, run=lambda: wa.change_system_volume(delta), display_arg=label)


def _system_mute_action(muted: bool) -> Action:
    name = "system_mute" if muted else "system_unmute"
    return Action(name=name, signature=name, run=lambda: wa.set_system_mute(muted))


def _timer_action(match: re.Match) -> Action:
    amount = int(match.group(1))
    unit = match.group(2).lower()
    seconds = amount * {"seconds": 1, "minutes": 60, "hours": 3600}[unit]
    label = f"{amount} {unit}"
    return Action(
        name="timer_set",
        signature=f"timer_set:{seconds}",
        run=lambda: answers.start_timer(seconds),
        display_arg=label,
    )


def _focus_action(match: re.Match) -> Action:
    name = match.group(1).strip()
    return Action(
        name="focus_window",
        signature=f"focus_window:{name.lower()}",
        run=lambda: wa.focus_window(name),
        display_arg=name,
    )


def _file_action(match: re.Match) -> Action:
    query = match.group(1).strip()
    return Action(
        name="find_file",
        signature=f"find_file:{query.lower()}",
        run=lambda: wa.find_file(query),
        display_arg=query,
    )


def _folder_action(match: re.Match) -> Action:
    name = match.group(1).strip()
    return Action(
        name="open_folder",
        signature=f"open_folder:{name.lower()}",
        run=lambda: wa.open_folder(name),
        display_arg=name,
    )


def _snap_action(match: re.Match) -> Action:
    side = match.group(1).lower()
    name = "window_left" if side == "left" else "window_right"
    return Action(name=name, signature=name, run=lambda: wa.snap_window(side))


_FOLDER_NAMES = (
    r"downloads|download|documents|dokumente|desktop|schreibtisch|pictures|bilder|music|musik|videos"
)

_PATTERNS: list[tuple[re.Pattern, Callable[[re.Match], Action]]] = [
    (
        re.compile(rf"^{_FILLER_PRE}what\s+time\s+is\s+it{_FILLER_POST}$", re.IGNORECASE),
        lambda m: Action(name="answer_time", signature="answer_time", run=answers.tell_time),
    ),
    (
        re.compile(rf"^{_FILLER_PRE}battery{_FILLER_POST}$", re.IGNORECASE),
        lambda m: Action(name="answer_battery", signature="answer_battery", run=answers.tell_battery),
    ),
    (
        re.compile(rf"^{_FILLER_PRE}whats\s+playing{_FILLER_POST}$", re.IGNORECASE),
        lambda m: Action(name="answer_now_playing", signature="answer_now_playing", run=answers.tell_now_playing),
    ),
    (
        re.compile(rf"^{_FILLER_PRE}cancel\s+(?:the\s+)?timer{_FILLER_POST}$", re.IGNORECASE),
        lambda m: Action(name="timer_cancel", signature="timer_cancel", run=answers.cancel_timer),
    ),
    (
        re.compile(
            rf"^{_FILLER_PRE}{_OPEN_TRIGGER}\s+(?:the\s+|my\s+)?({_FOLDER_NAMES})(?:\s+(?:folder|ordner))?{_FILLER_POST}$",
            re.IGNORECASE,
        ),
        _folder_action,
    ),
    (
        re.compile(
            rf"^{_FILLER_PRE}(?:turn\s+)?(?:the\s+)?(?:system|computer|pc|windows)(?:\s+volume)?\s+"
            rf"(up|down|louder|quieter){_FILLER_POST}$",
            re.IGNORECASE,
        ),
        lambda m: _system_volume_step(m.group(1).lower() in ("up", "louder")),
    ),
    (
        re.compile(
            rf"^{_FILLER_PRE}(un)?mute(?:\s+(?:the\s+)?(?:system|computer|pc|windows|sound|audio))?{_FILLER_POST}$",
            re.IGNORECASE,
        ),
        lambda m: _system_mute_action(not m.group(1)),
    ),
    (
        re.compile(
            rf"^{_FILLER_PRE}(?:turn\s+)?(?:the\s+)?(?:sound|audio)\s+(off|on){_FILLER_POST}$",
            re.IGNORECASE,
        ),
        lambda m: _system_mute_action(m.group(1).lower() == "off"),
    ),
    (
        re.compile(
            rf"^{_FILLER_PRE}(?:take\s+(?:a\s+)?)?(?:screenshot|screen\s*shot|bildschirmfoto){_FILLER_POST}$",
            re.IGNORECASE,
        ),
        lambda m: Action(name="screenshot", signature="screenshot", run=wa.take_screenshot),
    ),
    (
        re.compile(
            rf"^{_FILLER_PRE}(?:capture|grab)\s+(?:the\s+|my\s+)?screen{_FILLER_POST}$",
            re.IGNORECASE,
        ),
        lambda m: Action(name="screenshot", signature="screenshot", run=wa.take_screenshot),
    ),
    (
        re.compile(
            rf"^{_FILLER_PRE}(?:minimize|minimise)(?:\s+(?:the\s+|this\s+)?(?:window|app))?{_FILLER_POST}$",
            re.IGNORECASE,
        ),
        lambda m: Action(name="window_minimize", signature="window_minimize", run=wa.minimize_window),
    ),
    (
        re.compile(
            rf"^{_FILLER_PRE}(?:maximize|maximise)(?:\s+(?:the\s+|this\s+)?(?:window|app))?{_FILLER_POST}$",
            re.IGNORECASE,
        ),
        lambda m: Action(name="window_maximize", signature="window_maximize", run=wa.maximize_window),
    ),
    (
        re.compile(
            rf"^{_FILLER_PRE}restore(?:\s+(?:the\s+|this\s+)?window)?{_FILLER_POST}$",
            re.IGNORECASE,
        ),
        lambda m: Action(name="window_restore", signature="window_restore", run=wa.restore_window),
    ),
    (
        re.compile(
            rf"^{_FILLER_PRE}(?:snap|move|dock|put)(?:\s+(?:the\s+|this\s+)?)(?:window\s+)?"
            rf"(?:to\s+(?:the\s+)?)?(left|right){_FILLER_POST}$",
            re.IGNORECASE,
        ),
        _snap_action,
    ),
    (
        re.compile(
            rf"^{_FILLER_PRE}window\s+(?:to\s+(?:the\s+)?)?(left|right){_FILLER_POST}$",
            re.IGNORECASE,
        ),
        _snap_action,
    ),
    (
        re.compile(
            rf"^{_FILLER_PRE}(?:pause|stop|halt|quiet){_MUSIC_NOUN}{_FILLER_POST}$",
            re.IGNORECASE,
        ),
        lambda m: Action(name="spotify_pause", signature="spotify_pause", run=sp.pause),
    ),
    (
        re.compile(
            rf"^{_FILLER_PRE}(?:next|skip)(?:\s+(?:this|it))?{_MUSIC_NOUN}{_FILLER_POST}$",
            re.IGNORECASE,
        ),
        lambda m: Action(name="spotify_next", signature="spotify_next", run=sp.next_track),
    ),
    (
        re.compile(
            rf"^{_FILLER_PRE}(?:previous|last|go\s+back|back){_MUSIC_NOUN}{_FILLER_POST}$",
            re.IGNORECASE,
        ),
        lambda m: Action(name="spotify_previous", signature="spotify_previous", run=sp.previous_track),
    ),
    (
        re.compile(rf"\b{_OPEN_TRIGGER}\s+{_APP_WORDS}\b", re.IGNORECASE),
        lambda m: Action(
            name="open_app",
            signature=f"open_app:{m.group(1).lower()}",
            run=lambda: wa.open_app(m.group(1)),
            display_arg=m.group(1),
        ),
    ),
    (
        # "close <known app>" fires on partials, same as open_app.
        re.compile(rf"\b{_CLOSE_TRIGGER}\s+{_APP_WORDS}\b", re.IGNORECASE),
        lambda m: Action(
            name="close_app",
            signature=f"close_app:{m.group(1).lower()}",
            run=lambda: wa.close_app(m.group(1)),
            display_arg=m.group(1),
        ),
    ),
    (
        # Accepts both a literal-dot domain ("github.com") and a spoken one
        # ("animus infra dot com") - _despeak_dots is applied unconditionally
        # since it's a no-op on text that's already a proper literal domain.
        re.compile(
            rf"\b(?:navigate to|go to)\s+("
            rf"(?:[a-z0-9-]+\s+){{1,3}}dot\s+(?:{_TLDS})(?:\s+dot\s+(?:{_TLDS}))*"
            rf"|[a-z0-9][\w.\-/:]*\.[a-z]{{2,}}[\w.\-/]*"
            rf")",
            re.IGNORECASE,
        ),
        lambda m: Action(
            name="navigate_url",
            signature=f"navigate_url:{_despeak_dots(m.group(1)).lower()}",
            run=lambda: wa.navigate_url(_despeak_dots(m.group(1))),
            display_arg=_despeak_dots(m.group(1)),
        ),
    ),
    (
        re.compile(r"\btake (?:a )?(?:photo|picture|selfie)\b", re.IGNORECASE),
        lambda m: Action(
            name="take_photo",
            signature="take_photo",
            run=lambda: wa.take_photo_countdown().message,
        ),
    ),
]

# "write <text>" and "search ... for <query>" both capture open-ended,
# growing free text rather than a fixed short word (like an app name) - firing
# them mid-sentence (like the patterns above) means every interim update
# re-fires with a different, incomplete argument: e.g. "search x" -> "search
# x for" -> "search x for latest news" each produce a distinct dedup
# signature and were all three observed firing as separate browser searches
# during live testing. Both are only matched once the utterance is actually
# done speaking (speech_final), using the final, complete text.


def _build_targeted_search(m: re.Match) -> Action:
    query = m.group(2).strip()
    domain = _looks_like_domain(query)
    if domain:
        # "search github.com" / "search github dot com" - the whole query is
        # just a domain, so the user almost certainly means "go there", not
        # "Google the literal text 'github.com'".
        return Action(
            name="navigate_url",
            signature=f"navigate_url:{domain.lower()}",
            run=lambda: wa.navigate_url(domain),
            display_arg=domain,
        )
    browser = "default" if m.group(1).lower() == "google" else m.group(1).strip()
    return Action(
        name="browser_search",
        signature=f"browser_search:{m.group(1).lower()}:{query.lower()}",
        run=lambda: wa.browser_search(query=query, browser=browser),
        display_arg=query,
    )


def _build_generic_search(m: re.Match) -> Action:
    query = m.group(1).strip()
    domain = _looks_like_domain(query)
    if domain:
        return Action(
            name="navigate_url",
            signature=f"navigate_url:{domain.lower()}",
            run=lambda: wa.navigate_url(domain),
            display_arg=domain,
        )
    return Action(
        name="browser_search",
        signature=f"browser_search:default:{query.lower()}",
        run=lambda: wa.browser_search(query=query, browser="default"),
        display_arg=query,
    )


def _simple(name: str, run: Callable[[], str], display_arg: str = "") -> Callable[[re.Match], Action]:
    """Builder for fixed commands with no captured arguments."""
    return lambda m: Action(name=name, signature=name, run=run, display_arg=display_arg)


# Spotify / music. All final-only: these are short whole-utterance commands,
# and a growing partial like "play" (-> resume) would otherwise fire before
# "play fire force opening" (-> search & play) is complete.
_SPOTIFY_FINAL_PATTERNS: list[tuple[re.Pattern, Callable[[re.Match], Action]]] = [
    (
        re.compile(
            rf"^{_FILLER_PRE}(?:pause|stop|halt|quiet){_MUSIC_NOUN}{_FILLER_POST}$",
            re.IGNORECASE,
        ),
        _simple("spotify_pause", sp.pause),
    ),
    (
        re.compile(
            rf"^{_FILLER_PRE}(?:resume|continue|unpause|play|keep\s+playing){_MUSIC_NOUN}{_FILLER_POST}$",
            re.IGNORECASE,
        ),
        _simple("spotify_resume", sp.resume),
    ),
    (
        re.compile(
            rf"^{_FILLER_PRE}(?:next|skip)(?:\s+(?:this|it))?{_MUSIC_NOUN}{_FILLER_POST}$",
            re.IGNORECASE,
        ),
        _simple("spotify_next", sp.next_track),
    ),
    (
        re.compile(
            rf"^{_FILLER_PRE}(?:previous|last|go\s+back|back){_MUSIC_NOUN}{_FILLER_POST}$", re.IGNORECASE
        ),
        _simple("spotify_previous", sp.previous_track),
    ),
    (
        re.compile(
            rf"^{_FILLER_PRE}(?:set\s+)?(?:the\s+)?volume\s+(?:to\s+)?(\d{{1,3}})(?:\s*(?:percent|%))?"
            rf"{_FILLER_POST}$",
            re.IGNORECASE,
        ),
        lambda m: Action(
            name="spotify_volume",
            signature=f"spotify_volume:{m.group(1)}",
            run=lambda: sp.set_volume(int(m.group(1))),
            display_arg=f"{m.group(1)}%",
        ),
    ),
    (
        re.compile(
            rf"^{_FILLER_PRE}(?:turn\s+(?:it|the\s+volume|the\s+music)\s+up|louder|volume\s+up|turn\s+up){_FILLER_POST}$",
            re.IGNORECASE,
        ),
        _simple("spotify_volume_up", lambda: sp.change_volume(+15), "UP"),
    ),
    (
        re.compile(
            rf"^{_FILLER_PRE}(?:turn\s+(?:it|the\s+volume|the\s+music)\s+down|quieter|softer|volume\s+down|turn\s+down)"
            rf"{_FILLER_POST}$",
            re.IGNORECASE,
        ),
        _simple("spotify_volume_down", lambda: sp.change_volume(-15), "DOWN"),
    ),
    (
        re.compile(rf"^{_FILLER_PRE}(?:turn\s+)?shuffle\s+(on|off){_FILLER_POST}$", re.IGNORECASE),
        lambda m: Action(
            name="spotify_shuffle",
            signature=f"spotify_shuffle:{m.group(1).lower()}",
            run=lambda: sp.set_shuffle(m.group(1).lower() == "on"),
            display_arg=m.group(1),
        ),
    ),
    (
        # "play <anything>" - search Spotify and play the top hit. Comes after
        # the bare "play"/"play the music" resume pattern above.
        re.compile(rf"\bplay\b[.,:;]?\s+(.+?){_SENTENCE_END}", re.IGNORECASE),
        lambda m: Action(
            name="spotify_play",
            signature=f"spotify_play:{m.group(1).lower()}",
            run=lambda: sp.play(m.group(1).strip()),
            display_arg=m.group(1),
        ),
    ),
]


_FINAL_ONLY_PATTERNS: list[tuple[re.Pattern, Callable[[re.Match], Action]]] = [
    (
        re.compile(
            rf"^{_FILLER_PRE}(?:set\s+)?(?:a\s+)?timer\s+(?:for\s+)?(\d+)\s*"
            rf"(seconds|minutes|hours){_FILLER_POST}$",
            re.IGNORECASE,
        ),
        _timer_action,
    ),
    (
        re.compile(
            rf"^{_FILLER_PRE}(?:switch\s+to|focus|show)\s+(?:the\s+)?"
            rf"([a-z0-9][\w+\-]*(?:\s+[\w+\-]+){{0,2}}){_FILLER_POST}$",
            re.IGNORECASE,
        ),
        _focus_action,
    ),
    (
        re.compile(
            rf"^{_FILLER_PRE}find\s+(?:the\s+)?file\s+(.+?){_FILLER_POST}$",
            re.IGNORECASE,
        ),
        _file_action,
    ),
    (
        # Final-only: a growing partial "system volume 3" must not fire
        # before the speaker finishes "system volume 30".
        re.compile(
            rf"^{_FILLER_PRE}(?:set\s+)?(?:the\s+)?(?:system|computer|pc|windows)\s+volume\s+"
            rf"(?:to\s+)?(\d{{1,3}})(?:\s*(?:percent|%))?{_FILLER_POST}$",
            re.IGNORECASE,
        ),
        _system_volume_action,
    ),
    *_SPOTIFY_FINAL_PATTERNS,
    (
        # "open <browser> and search <query>" - ties the search to the named
        # browser. Tried before the generic "search ..." pattern below,
        # which would otherwise still match the "search <query>" tail of
        # this same phrase but silently ignore that a browser was named and
        # fall back to "default" instead of actually using it.
        re.compile(
            rf"\bopen\s+{_KNOWN_SEARCH_TARGETS}\s+and\s+{_SEARCH_VERB}(.+?){_SENTENCE_END}",
            re.IGNORECASE,
        ),
        lambda m: Action(
            name="browser_search",
            signature=f"browser_search:{m.group(1).lower()}:{m.group(2).lower()}",
            run=lambda: wa.browser_search(
                query=m.group(2).strip(),
                browser="default" if m.group(1).lower() == "google" else m.group(1).strip(),
            ),
            display_arg=m.group(2),
        ),
    ),
    (
        # Explicit target named: "search <chrome/arc/x/...> for <query>".
        # "google" is accepted here too even though it isn't a browser app -
        # browser_search always searches via Google regardless, so it just
        # keeps "google" out of the captured query text below.
        re.compile(
            rf"\b(?:search|look)\s+(?:in\s+|on\s+)?{_KNOWN_SEARCH_TARGETS}\s+for\s+(.+?){_SENTENCE_END}",
            re.IGNORECASE,
        ),
        _build_targeted_search,
    ),
    (
        # Natural phrasing with no named target: "search for <query>",
        # "search <query>", "look for/up <query>", "google <query>", "find
        # <query>" - defaults to the system's default browser.
        re.compile(rf"\b{_SEARCH_VERB}(.+?){_SENTENCE_END}", re.IGNORECASE),
        _build_generic_search,
    ),
    (
        _WRITE_PATTERN,
        lambda m: Action(
            name="write_in_app",
            signature="write_in_app",
            run=lambda: wa.write_in_app(m.group(1).strip()),
        ),
    ),
    (
        # Fallback for apps not listed in _APP_WORDS: "open <one to three
        # words>". Final-only because a growing partial like "open spo"
        # would otherwise fire with a truncated name. The capture stops at
        # "and"/"please"/"for me"/"now" so compound commands ("open chrome
        # and search for ...") don't swallow their tail. Resolving the name
        # to a real program (PATH, Start Menu shortcut, Store app) happens
        # in windows_actions.open_app. Last in the list so every more
        # specific pattern gets its chance first.
        re.compile(
            rf"\b{_OPEN_TRIGGER}\s+(?!(?:the|up|a|an)\b)([a-z0-9][\w+\-]*(?:\s+[\w+\-]+){{0,2}}?)"
            rf"(?=\s+(?:and|please|for\s+me|now)\b|{_SENTENCE_END})",
            re.IGNORECASE,
        ),
        lambda m: Action(
            name="open_app",
            signature=f"open_app:{m.group(1).lower()}",
            run=lambda: wa.open_app(m.group(1)),
            display_arg=m.group(1),
        ),
    ),
    (
        # Generic "close <one to three words>" - final-only, same reasoning
        # as the generic open above.
        re.compile(
            rf"\b{_CLOSE_TRIGGER}\s+(?!(?:the|a|an)\b)([a-z0-9][\w+\-]*(?:\s+[\w+\-]+){{0,2}}?)"
            rf"(?=\s+(?:and|please|for\s+me|now)\b|{_SENTENCE_END})",
            re.IGNORECASE,
        ),
        lambda m: Action(
            name="close_app",
            signature=f"close_app:{m.group(1).lower()}",
            run=lambda: wa.close_app(m.group(1)),
            display_arg=m.group(1),
        ),
    ),
]

# Strips conversational punctuation Deepgram's smart-formatting can add
# ("Open the notes app, please!") before regex matching. Deliberately
# leaves '.', ':', '/', '-' untouched - those are load-bearing for
# navigate_url's domain-token detection (e.g. "github.com"); stripping all
# punctuation indiscriminately would break that instead of helping.
_STRIP_CHARS_PATTERN = re.compile(r"[,;!?\"'`]")


def _redact_secrets(exc: BaseException) -> str:
    """Strip API keys out of exception text (requests embeds `?key=` in 429s)."""
    return re.sub(r"([?&]key=)[^&\s]+", r"\1***", str(exc), flags=re.IGNORECASE)


# Cheap phonetic patches for names Whisper routinely mangles with a German
# accent. Applied before regex matching so "play Strakitz" still finds
# Stray Kids without waiting on the LLM rewrite.
_STT_FIXES = (
    (re.compile(r"\bjarvis\.play\b"), "jarvis play"),
    (re.compile(r"\bplays\b"), "play"),
    (re.compile(r"\bthe weekend\b"), "the weeknd"),
    (re.compile(r"\bsave your cheers\b"), "save your tears"),
    (re.compile(r"\bstr[aeiy]+[\s-]?k(?:itz|erts|ids|ets|idz|its)\b"), "stray kids"),
    (re.compile(r"\b(?:strikets|strakitz|straykerts|straychids)\b"), "stray kids"),
    (re.compile(r"\b(?:straight|strait)\s+kids\b"), "stray kids"),
    (re.compile(r"\b(?:leaf|leave)\s+straight\s*heads\b"), "stray kids"),
    (re.compile(r"\bstraight\s*heads\b"), "stray kids"),
    (re.compile(r"\bmegav(?:us|erse|as)\b"), "megaverse"),
    (re.compile(r"\binfano\b|\bin fenno\b"), "inferno"),
)

def _apply_stt_fixes(text: str) -> str:
    for pattern, replacement in _STT_FIXES:
        text = pattern.sub(replacement, text)
    return text


def _normalize_transcript(text: str) -> str:
    normalized = _STRIP_CHARS_PATTERN.sub("", text.lower())
    # "play. Save your tears" - Whisper puts a period between the verb and
    # the title. A dot with a space after it is sentence punctuation;
    # "github.com" has no space and stays intact.
    normalized = re.sub(r"[.!?]+\s+", " ", normalized)
    normalized = re.sub(r"\s+", " ", normalized).strip()
    # Whisper/Deepgram often leave a trailing "stop." / "skip." - the music
    # commands are anchored to end-of-string, so a leftover period made
    # "Jarvis, stop." miss the pause pattern and fall through to Gemini.
    # Internal dots ("github.com", "mrs. green") stay.
    normalized = re.sub(r"[.!?]+$", "", normalized).strip()
    return _rewrite_spoken_german(_apply_stt_fixes(normalized))


_DE_LEADING_FILLER = re.compile(
    r"^(?:(?:bitte|mal|doch|einfach|kurz|eben|einmal|jarvis|"
    r"kannst du(?: mir)?|könntest du(?: mir)?|koenntest du(?: mir)?|"
    r"würdest du(?: mir)?|wuerdest du(?: mir)?|hallo|hey)\s+)+"
)
# Spoken padding that sits between the verb and the object ("minimiere mal
# bitte das Fenster"). Stripped as whole words so a search for "Screenshot"
# still contains that word.
_DE_INLINE_FILLER = re.compile(r"\b(?:bitte|mal|doch|einfach|kurz|eben|einmal|für mich|for me)\b")
_DE_ARTICLE = r"den|die|das|dem|der|ein|eine|einen|einem"


def _shuffle_state(match: re.Match) -> str:
    state = {"an": "on", "ein": "on", "on": "on", "aus": "off", "off": "off"}[match.group(1)]
    return f"shuffle {state}"


def _pc_volume_direction(match: re.Match) -> str:
    return "system volume up" if match.group(1) == "lauter" else "system volume down"


def _snap_side(match: re.Match) -> str:
    return "snap left" if match.group(1) == "links" else "snap right"


def _apply_desktop_rewrites(text: str) -> str:
    """Map one clause onto a desktop command. Anchored so a search that only
    mentions the word "screenshot" is left alone."""
    text = re.sub(
        r"^(?:wechsle|wechsel(?:n)?|fokussiere|fokussier|fokus)\s+(?:zu(?:m)?\s+|auf\s+)?",
        "switch to ",
        text,
    )
    text = re.sub(
        r"^(?:(?:mach(?:e)?|nimm)\s+)?(?:ein(?:en)?\s+)?(?:screenshot|screen\s*shot|bildschirmfoto)"
        r"(?:\s+(?:machen|aufnehmen))?$",
        "take a screenshot",
        text,
    )
    text = re.sub(r"^bildschirm\s+aufnehmen$", "take a screenshot", text)
    text = re.sub(r"^(?:mach(?:e)?\s+)?(?:das\s+)?(?:fenster\s+)?vollbild$", "maximize", text)
    text = re.sub(r"^(?:mach(?:e)?\s+)?(?:das\s+)?fenster\s+kleiner$", "minimize", text)
    text = re.sub(r"^(?:mach(?:e)?\s+)?(?:das\s+)?fenster\s+(?:größer|groesser)$", "maximize", text)
    text = re.sub(
        r"^(?:schieb(?:e)?|andock(?:e|en)?|dock(?:e)?)\s+(?:das\s+)?fenster\s+(?:nach\s+)?(links|rechts)$",
        _snap_side,
        text,
    )
    text = re.sub(
        r"^(?:das\s+)?fenster\s+(?:nach\s+)?(links|rechts)(?:\s+(?:schieben|andocken|docken))?$",
        _snap_side,
        text,
    )
    text = re.sub(r"^(?:nach\s+)?(links|rechts)\s+(?:andocken|docken|schieben)$", _snap_side, text)
    text = re.sub(r"^(?:andock(?:e|en)?|dock(?:e)?)\s+(?:nach\s+)?(links|rechts)$", _snap_side, text)
    text = re.sub(r"^(?:das\s+)?fenster\s+(?:minimieren|verkleinern)$", "minimize", text)
    text = re.sub(r"^(?:das\s+)?fenster\s+(?:maximieren|vergrößern|vergroessern)$", "maximize", text)
    text = re.sub(
        r"^(?:minimiere|minimieren|verkleinere|verkleinern)(?:\s+(?:das\s+)?fenster)?$",
        "minimize",
        text,
    )
    text = re.sub(
        r"^(?:maximiere|maximieren|vergrößere|vergroessere|vergrößern|vergroessern)"
        r"(?:\s+(?:das\s+)?fenster)?$",
        "maximize",
        text,
    )
    text = re.sub(r"^(?:stell(?:e)?\s+)?(?:das\s+)?fenster\s+wieder\s+her$", "restore", text)
    text = re.sub(r"^(?:fenster\s+)?wiederherstellen$", "restore", text)
    text = re.sub(
        r"^(?:(?:mach|dreh)(?:e)?\s+)?(?:den\s+|die\s+|das\s+)?"
        r"(?:computer|pc|windows|system)\s+(lauter|leiser)$",
        _pc_volume_direction,
        text,
    )
    text = re.sub(
        r"^(?:stelle\s+)?(?:die\s+)?(?:"
        r"system\s*(?:lautstärke|lautstaerke)|"
        r"(?:computer|pc|windows|system)\s+(?:lautstärke|lautstaerke)"
        r")\s+(?:auf\s+)?(\d{1,3})(?:\s*(?:prozent|%))?$",
        r"system volume \1",
        text,
    )
    text = re.sub(r"^(?:ton|sound|audio|lautstärke|lautstaerke)\s+(?:aus|stumm)$", "mute", text)
    text = re.sub(r"^(?:ton|sound|audio)\s+(?:an|ein)$", "unmute", text)
    text = re.sub(r"^stumm\s*schalt\w*$", "mute", text)
    text = re.sub(r"^entstumm\w*$", "unmute", text)
    text = re.sub(r"^(?:mach(?:e)?\s+)?(?:stumm|lautlos)$", "mute", text)
    text = re.sub(r"^dreh(?:e)?\s+(?:es\s+)?(lauter|leiser)$", _pc_volume_direction, text)
    text = re.sub(
        r"^(?:dreh(?:e)?|mach(?:e)?)\s+(?:den\s+)?ton\s+(lauter|leiser)$",
        _pc_volume_direction,
        text,
    )
    return text


def _apply_answer_rewrites(text: str) -> str:
    """Map one clause onto a spoken answer, timer, or cancel."""
    text = re.sub(
        r"^(?:sag(?:en|e)?(?:\s+mir)?\s+)?(?:wie\s+sp[aä]t(?:\s+(?:ist\s+es|es\s+ist))?|"
        r"wie\s*viel\s+uhr(?:\s+(?:ist\s+es|es\s+ist))?|uhrzeit)$",
        "what time is it",
        text,
    )
    text = re.sub(
        r"^(?:wie\s+voll\s+ist\s+der\s+akku|wie\s+viel\s+akku|akku(?:stand)?|batterie(?:stand)?)$",
        "battery",
        text,
    )
    text = re.sub(
        r"^(?:was\s+l[aä]uft(?:\s+gerade)?|was\s+laeuft(?:\s+gerade)?|"
        r"welcher\s+(?:song|titel)|was\s+spielst\s+du(?:\s+gerade)?)$",
        "whats playing",
        text,
    )
    text = re.sub(
        r"^(?:timer\s+(?:abbrechen|stoppen|aus)|(?:stopp(?:e)?|stoppe)\s+den\s+timer)$",
        "cancel timer",
        text,
    )
    text = re.sub(
        r"^(?:stell(?:e)?\s+)?(?:einen\s+)?(?:timer|wecker)\s+(?:auf\s+|fuer\s+|für\s+)?(\d+)\s*"
        r"(sekunden|sekunde|minuten|minute|stunden|stunde)$",
        lambda m: (
            f"timer {m.group(1)} "
            + {
                "sekunde": "seconds",
                "sekunden": "seconds",
                "minute": "minutes",
                "minuten": "minutes",
                "stunde": "hours",
                "stunden": "hours",
            }[m.group(2)]
        ),
        text,
    )
    return text


def _rewrite_spoken_german(text: str) -> str:
    """Map common German imperatives onto the English command patterns.

    Applied to every transcript. The replacements are German function words
    and verbs, so an English command is left unchanged. Song titles and
    search text after the verb stay as spoken.
    """
    text = _DE_LEADING_FILLER.sub("", text)
    text = _DE_INLINE_FILLER.sub(" ", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"(?:\s+(?:bitte|jetzt|danke))+$", "", text)

    # Desktop and answer commands are recognized per clause, so
    # "öffne Chrome und mach einen Screenshot" still sees the screenshot half
    # on its own. Search compounds stay intact because their clauses are not rewritten.
    clauses = [clause.strip() for clause in re.split(r"\s+(?:und|and)\s+", text) if clause.strip()]
    text = " and ".join(
        _apply_answer_rewrites(_apply_desktop_rewrites(clause)) for clause in clauses
    )

    text = re.sub(
        r"\b(?:mach(?:e)?|nimm)\s+(?:bitte\s+)?(?:ein(?:en)?\s+)?(?:foto|photo|bild|selfie)\b",
        "take a photo",
        text,
    )
    text = re.sub(r"\b(?:foto|photo|selfie)\s+machen\b", "take a photo", text)

    text = re.sub(
        r"\bmach(?:e)?\s+((?:[a-z0-9äöüß\-]+\s+){0,3}[a-z0-9äöüß\-]+)\s+auf\b",
        r"open \1",
        text,
    )
    text = re.sub(
        r"\bmach(?:e)?\s+((?:[a-z0-9äöüß\-]+\s+){0,3}[a-z0-9äöüß\-]+)\s+zu\b",
        r"close \1",
        text,
    )
    text = re.sub(
        r"\b(?:mach(?:e)?\s+)?(?:es\s+)?(?:lauter|lautstärke\s+(?:hoch|lauter)|lautstaerke\s+(?:hoch|lauter))\b",
        "louder",
        text,
    )
    text = re.sub(
        r"\b(?:mach(?:e)?\s+)?(?:es\s+)?(?:leiser|lautstärke\s+(?:runter|leiser)|lautstaerke\s+(?:runter|leiser))\b",
        "quieter",
        text,
    )
    text = re.sub(r"\b(louder|quieter)\s+machen\b", r"\1", text)
    text = re.sub(
        r"\b(?:stelle\s+)?(?:die\s+)?(?:lautstärke|lautstaerke)\s+(?:auf\s+)?(\d{1,3})"
        r"(?:\s*(?:prozent|percent|%))?",
        r"volume \1",
        text,
    )
    text = re.sub(
        r"\b(?:zufall(?:swiedergabe)?|shuffle)\s+(an|ein|aus|on|off)\b",
        _shuffle_state,
        text,
    )

    text = re.sub(
        r"\b(?:spiel(?:e)?\s+weiter|mach\s+weiter|weiterspielen|fortsetzen)\b",
        "play",
        text,
    )
    text = re.sub(r"\bspiel(?:e)?\s+(.+?)\s+ab\b", r"play \1", text)
    text = re.sub(r"\bgeh(?:e)?\s+(?:zurück|zurueck)\b", "previous", text)
    text = re.sub(
        r"\b(?:geh(?:e)?|gehen|navigiere)\s+(?:mal\s+)?(?:zu|auf|nach)\b",
        "go to",
        text,
    )
    text = re.sub(
        r"\b(?:such(?:e)?|suchen)\s+(?:mal\s+)?(?:in|mit|auf|bei)\s+([a-z0-9]+)\s+nach\b",
        r"search \1 for",
        text,
    )
    text = re.sub(
        r"\b(?:such(?:e)?|finde)\s+(?:mir\s+)?(?:die\s+|eine\s+)?datei\s+",
        "find file ",
        text,
    )
    text = re.sub(r"\b(?:such(?:e)?|suchen)\s+(?:mal\s+)?nach\b", "search for", text)
    text = re.sub(r"\bgoogle(?:\s+mal)?\s+nach\b", "google", text)

    # "chrome öffnen" (verb last) before the bare infinitive becomes "open".
    def _verb_final(match: re.Match) -> str:
        verb = match.group(2)
        kind = "close" if verb.startswith("schlie") or verb.startswith("beend") else "open"
        return f"{kind} {match.group(1)}"

    text = re.sub(
        r"\b((?:[a-z0-9äöüß\-]+\s+){0,2}[a-z0-9äöüß\-]+)\s+"
        r"(öffnen|oeffnen|starten|schließen|schliessen|beenden)\b",
        _verb_final,
        text,
    )
    text = re.sub(r"\b(?:öffne|oeffne|öffnen|oeffnen|starte|starten)\b", "open", text)
    text = re.sub(r"\b(?:schließ(?:e|en)?|schliess(?:e|en)?|beende|beenden)\b", "close", text)
    text = re.sub(r"\bschreib(?:e)?\b", "write", text)
    text = re.sub(r"\bspiel(?:e)?\b", "play", text)
    text = re.sub(r"\b(?:suchen|suche|such)\b", "search", text)
    text = re.sub(r"\bfinde\b", "find", text)
    text = re.sub(r"\b(search for|google|find)\s+(?:den|die|das|dem)\s+", r"\1 ", text)
    text = re.sub(r"\bstopp\b", "stop", text)
    text = re.sub(r"\b(?:nächst\w*|naechst\w*|überspring(?:e)?|ueberspring(?:e)?)\b", "next", text)
    text = re.sub(r"\bzurück\b|\bzurueck\b", "previous", text)
    text = re.sub(r"\bvorherige\w*\b", "previous", text)
    text = re.sub(r"^letzte[rsn]?$", "previous", text)
    text = re.sub(r"\bletzte[rsn]?\s+(?:lied|song|titel|track)\b", "previous", text)
    text = re.sub(
        r"\bweiter\s+(?:(?:die|der|das|den)\s+)?(?:musik|lied|song|titel|track)\b",
        "next",
        text,
    )
    text = re.sub(r"^weiter$", "next", text)

    text = re.sub(rf"\bpunkt\s+({_TLDS})\b", r"dot \1", text)
    text = re.sub(r"\bfür mich\b", "for me", text)
    text = re.sub(r"\bund\b", "and", text)

    text = re.sub(
        rf"\b(open|close)(?:\s+(?:{_DE_ARTICLE}|mal|bitte|doch))+",
        r"\1",
        text,
    )
    text = re.sub(
        r"\b(pause|stop|halt|next|previous)\s+(?:(?:die|der|das|den|dem)\s+)?"
        r"(?:musik|lied|song|titel|track)\b",
        r"\1",
        text,
    )
    text = re.sub(
        r"\bplay\s+(?:(?:die|der|das|den)\s+)?(?:musik|lied|titel|song|track)$",
        "play",
        text,
    )
    text = re.sub(r"(?:\s+(?:bitte|jetzt))+$", "", text)
    return re.sub(r"\s+", " ", text).strip()


# Bracketed, uppercase confirmation-badge style, e.g. "[LAUNCHED NOTEPAD]".
_ACTION_LABELS = {
    "open_app": "[LAUNCHED {arg}]",
    "close_app": "[CLOSING {arg}]",
    "navigate_url": "[NAVIGATING TO {arg}]",
    "browser_search": "[SEARCHING FOR {arg}]",
    "take_photo": "[CAPTURING PHOTO]",
    "write_in_app": "[WRITING TEXT]",
    "spotify_play": "[PLAYING {arg}]",
    "spotify_pause": "[PAUSED]",
    "spotify_resume": "[RESUMING]",
    "spotify_next": "[NEXT TRACK]",
    "spotify_previous": "[PREVIOUS TRACK]",
    "spotify_volume": "[VOLUME {arg}]",
    "spotify_volume_up": "[VOLUME UP]",
    "spotify_volume_down": "[VOLUME DOWN]",
    "spotify_shuffle": "[SHUFFLE {arg}]",
    "system_volume": "[SYSTEM VOLUME {arg}]",
    "system_volume_up": "[SYSTEM VOLUME UP]",
    "system_volume_down": "[SYSTEM VOLUME DOWN]",
    "system_mute": "[MUTED]",
    "system_unmute": "[UNMUTED]",
    "screenshot": "[SCREENSHOT]",
    "window_minimize": "[MINIMIZED]",
    "window_maximize": "[MAXIMIZED]",
    "window_restore": "[RESTORED]",
    "window_left": "[WINDOW LEFT]",
    "window_right": "[WINDOW RIGHT]",
    "answer_time": "[TIME]",
    "answer_battery": "[BATTERY]",
    "answer_now_playing": "[NOW PLAYING]",
    "timer_set": "[TIMER {arg}]",
    "timer_cancel": "[TIMER CANCELLED]",
    "focus_window": "[FOCUS {arg}]",
    "open_folder": "[FOLDER {arg}]",
    "find_file": "[FILE {arg}]",
}


# "open spotify and chrome" / "close chrome and firefox and discord": a
# list of apps sharing one verb. Split into "open spotify", "open chrome"
# so each becomes its own action. Segments that start with their own verb
# are NOT split off ("open chrome and search for cats" must stay intact so
# the compound browser_search pattern can tie the search to chrome).
_CONJUNCTION_SPLIT = re.compile(r"\s+(?:and|plus|also|then)\s+", re.IGNORECASE)
_APP_VERB = re.compile(rf"\b({_OPEN_TRIGGER}|{_CLOSE_TRIGGER})\s+", re.IGNORECASE)
_STARTS_WITH_APP_VERB = re.compile(rf"^(?:{_OPEN_TRIGGER}|{_CLOSE_TRIGGER})\s+", re.IGNORECASE)
_STARTS_WITH_OTHER_VERB = re.compile(
    rf"^(?:{_SEARCH_VERB}|write|take|navigate|go|play|pause|stop|resume|skip|next|previous|volume|shuffle|"
    rf"mute|unmute|minimize|minimise|maximize|maximise|restore|snap|screenshot|system|"
    rf"switch|focus|show|find|timer|cancel|battery|what)\b",
    re.IGNORECASE,
)
# A finished command after "and" is its own action. "open chrome and search
# for cats" does not match, so that phrase stays whole and keeps the browser.
_STANDALONE_COMMAND = re.compile(
    r"^(?:"
    r"take a (?:screenshot|photo)|"
    r"minimize|maximize|restore|"
    r"snap (?:left|right)|"
    r"mute|unmute|"
    r"system volume (?:up|down|\d{1,3})|"
    r"(?:pause|stop|next|previous|play)|"
    r"what time is it|whats playing|battery|cancel timer|"
    r"timer \d+ (?:seconds|minutes|hours)|"
    r"switch to .+|focus .+|show .+|find file .+"
    r")$",
    re.IGNORECASE,
)


def _expand_conjunctions(text: str) -> list[str]:
    """Returns the list of standalone commands contained in `text`.

    - part after "and" has no verb, previous part was open/close -> the verb
      is carried over ("open spotify and chrome" -> "open spotify", "open
      chrome")
    - part starts with its own open/close verb -> standalone command
      ("open spotify and close chrome")
    - part starts with any other verb (search, write, ...) or nothing
      applies -> re-joined, so compound patterns still see the whole phrase
    """
    parts = _CONJUNCTION_SPLIT.split(text)
    if len(parts) == 1:
        return [text]
    out = [parts[0]]
    verb_match = _APP_VERB.search(parts[0])
    verb = verb_match.group(1) if verb_match else None
    for part in parts[1:]:
        if _STARTS_WITH_APP_VERB.match(part):
            out.append(part)
            verb = _APP_VERB.search(part).group(1)
        elif _STANDALONE_COMMAND.match(part):
            out.append(part)
            verb = None
        elif verb and not _STARTS_WITH_OTHER_VERB.match(part):
            out.append(f"{verb} {part}")
        else:
            out[-1] = f"{out[-1]} and {part}"
            verb = None  # a search/write tail ends the app list
    return out


def _match_regex(text: str, speech_final: bool, already_fired: Callable[[str], bool]) -> Optional[Action]:
    """Returns the first NEW (not already-fired) action found.

    A match whose signature already fired earlier in this utterance is
    skipped rather than returned - otherwise a compound command like "open
    chrome and search for python tutorials" would never get past its own
    "open chrome" prefix: that substring always matches open_app first, and
    once it has already fired, returning it again just gets silently
    deduped one level up, so the trailing "search for ..." is never even
    considered on the same call.
    """
    for pattern, build in _PATTERNS:
        match = pattern.search(text)
        if match:
            action = build(match)
            if not already_fired(action.signature):
                return action
    if speech_final:
        for pattern, build in _FINAL_ONLY_PATTERNS:
            match = pattern.search(text)
            if match:
                action = build(match)
                if not already_fired(action.signature):
                    return action
    return None


class LLMFallbackClassifier:
    """Optional secondary pass for phrasing the regex layer doesn't cover.

    Only invoked when `_match_regex` returns None. Keeps the assistant
    responsive to the exact demo phrases without paying an LLM round trip on
    every single transcript update.
    """

    SYSTEM_PROMPT = (
        "You classify a spoken command fragment into at most one action. "
        "Respond with strict JSON only, one of: "
        '{"action": null} or '
        '{"action": "open_app", "app": "<name>"} or '
        '{"action": "close_app", "app": "<name>"} or '
        '{"action": "navigate_url", "url": "<url>"} or '
        '{"action": "browser_search", "browser": "<name>", "query": "<text>"} or '
        '{"action": "take_photo"} or '
        '{"action": "write_in_app", "text": "<text>"} or '
        '{"action": "spotify_play", "query": "<song or artist>"} or '
        '{"action": "spotify_pause"} or '
        '{"action": "spotify_resume"} or '
        '{"action": "spotify_next"} or '
        '{"action": "spotify_previous"} or '
        '{"action": "system_volume", "percent": 40} or '
        '{"action": "system_volume", "direction": "up"} or '
        '{"action": "system_volume", "direction": "down"} or '
        '{"action": "system_mute", "muted": true} or '
        '{"action": "screenshot"} or '
        '{"action": "window", "op": "minimize"} or '
        '{"action": "answer_time"} or '
        '{"action": "answer_battery"} or '
        '{"action": "answer_now_playing"} or '
        '{"action": "timer", "seconds": 300} or '
        '{"action": "timer_cancel"} or '
        '{"action": "focus_window", "app": "<name>"} or '
        '{"action": "open_folder", "folder": "downloads"} or '
        '{"action": "find_file", "query": "<file name>"}. '
        "window op is one of minimize, maximize, restore, left, right. "
        "system_volume and system_mute are the PC, not Spotify. "
        "Bare volume/louder/lauter without computer, system, pc, or windows stays Spotify. "
        "The speaker may use German or English, including casual paraphrases. "
        "Keep search queries, dictated text, and song titles in the original wording. "
        "German app names: Rechner/Taschenrechner=calculator, Kamera=camera, "
        "Dateien=explorer, Konsole=terminal, Notizblock=notepad. "
        "When the user is telling you to do something this schema can do, pick that "
        "action. Respond {\"action\": null} only when there is no request, the "
        "utterance is only a wake word, or it is small talk."
    )

    def __init__(self) -> None:
        self.provider = os.environ.get("LLM_PROVIDER", "anthropic").lower()
        self.anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "")
        self.anthropic_model = os.environ.get("ANTHROPIC_MODEL", "claude-haiku-4-5-20251001")
        self.gemini_key = os.environ.get("GEMINI_API_KEY", "")
        self.gemini_model = os.environ.get("GEMINI_MODEL", "gemini-2.5-flash")

    def enabled(self) -> bool:
        if self.provider == "anthropic":
            return bool(self.anthropic_key)
        if self.provider == "gemini":
            return bool(self.gemini_key)
        return False

    def classify(self, text: str) -> Optional[Action]:
        if not self.enabled():
            return None
        try:
            raw = self._call_llm(text, self.SYSTEM_PROMPT, timeout=8)
            raw = raw.strip()
            if raw.startswith("```"):
                raw = re.sub(r"^```(?:json)?\s*", "", raw)
                raw = re.sub(r"\s*```$", "", raw)
            print(f"[intent] LLM classify {text!r} -> {raw[:160]}")
            data = json.loads(raw)
        except Exception as exc:  # network/parsing failures should never crash the pipeline
            print(f"[intent] LLM fallback failed: {_redact_secrets(exc)}")
            return None
        return self._action_from_json(data)

    def _call_llm(self, text: str, system: str, timeout: int = 4, max_tokens: int = 200) -> str:
        import requests

        if self.provider == "anthropic":
            resp = requests.post(
                "https://api.anthropic.com/v1/messages",
                headers={
                    "x-api-key": self.anthropic_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": self.anthropic_model,
                    "max_tokens": max_tokens,
                    "system": system,
                    "messages": [{"role": "user", "content": text}],
                },
                timeout=timeout,
            )
            resp.raise_for_status()
            return resp.json()["content"][0]["text"]

        # Gemini
        resp = requests.post(
            f"https://generativelanguage.googleapis.com/v1beta/models/{self.gemini_model}:generateContent",
            params={"key": self.gemini_key},
            json={
                "systemInstruction": {"parts": [{"text": system}]},
                "contents": [{"parts": [{"text": text}]}],
                "generationConfig": {"maxOutputTokens": max_tokens, "temperature": 0},
            },
            timeout=timeout,
        )
        resp.raise_for_status()
        return resp.json()["candidates"][0]["content"]["parts"][0]["text"]

    @staticmethod
    def _action_from_json(data: dict) -> Optional[Action]:
        action = data.get("action")
        if action == "open_app":
            app = data["app"]
            return Action("open_app", f"open_app:{app.lower()}", lambda: wa.open_app(app), display_arg=app)
        if action == "close_app":
            app = data["app"]
            return Action("close_app", f"close_app:{app.lower()}", lambda: wa.close_app(app), display_arg=app)
        if action == "navigate_url":
            url = data["url"]
            return Action(
                "navigate_url", f"navigate_url:{url.lower()}", lambda: wa.navigate_url(url), display_arg=url
            )
        if action == "browser_search":
            browser, query = data.get("browser", "default"), data["query"]
            return Action(
                "browser_search",
                f"browser_search:{browser.lower()}:{query.lower()}",
                lambda: wa.browser_search(query=query, browser=browser),
                display_arg=query,
            )
        if action == "take_photo":
            return Action("take_photo", "take_photo", lambda: wa.take_photo_countdown().message)
        if action == "write_in_app":
            text = data["text"]
            return Action(
                "write_in_app",
                f"write_in_app:{hash(text) & 0xFFFF}",
                lambda: wa.write_in_app(text),
            )
        if action == "spotify_play":
            query = data["query"]
            return Action(
                "spotify_play",
                f"spotify_play:{query.lower()}",
                lambda: sp.play(query),
                display_arg=query,
            )
        if action == "spotify_pause":
            return Action("spotify_pause", "spotify_pause", sp.pause)
        if action == "spotify_resume":
            return Action("spotify_resume", "spotify_resume", sp.resume)
        if action == "spotify_next":
            return Action("spotify_next", "spotify_next", sp.next_track)
        if action == "spotify_previous":
            return Action("spotify_previous", "spotify_previous", sp.previous_track)
        if action == "system_volume":
            percent = data.get("percent")
            if percent is not None and str(percent).strip() != "":
                try:
                    level = _clamp_spoken_percent(str(percent).strip())
                except ValueError:
                    return None
                return Action(
                    "system_volume",
                    f"system_volume:{level}",
                    lambda: wa.set_system_volume(level),
                    display_arg=f"{level}%",
                )
            direction = str(data.get("direction", "")).lower()
            if direction in ("up", "down"):
                return _system_volume_step(direction == "up")
            return None
        if action == "system_mute":
            muted = data.get("muted", True)
            if isinstance(muted, str):
                muted = muted.strip().lower() not in ("false", "0", "no", "off")
            return _system_mute_action(bool(muted))
        if action == "screenshot":
            return Action("screenshot", "screenshot", wa.take_screenshot)
        if action == "window":
            op = str(data.get("op", "")).lower()
            windows = {
                "minimize": ("window_minimize", wa.minimize_window),
                "maximize": ("window_maximize", wa.maximize_window),
                "restore": ("window_restore", wa.restore_window),
                "left": ("window_left", lambda: wa.snap_window("left")),
                "right": ("window_right", lambda: wa.snap_window("right")),
            }
            chosen = windows.get(op)
            if chosen is None:
                return None
            name, run = chosen
            return Action(name, name, run)
        if action == "answer_time":
            return Action("answer_time", "answer_time", answers.tell_time)
        if action == "answer_battery":
            return Action("answer_battery", "answer_battery", answers.tell_battery)
        if action == "answer_now_playing":
            return Action("answer_now_playing", "answer_now_playing", answers.tell_now_playing)
        if action == "timer":
            try:
                seconds = max(1, int(data.get("seconds", 0)))
            except (TypeError, ValueError):
                return None
            return Action("timer_set", f"timer_set:{seconds}", lambda: answers.start_timer(seconds), display_arg=f"{seconds}s")
        if action == "timer_cancel":
            return Action("timer_cancel", "timer_cancel", answers.cancel_timer)
        if action == "focus_window":
            app = str(data.get("app", "")).strip()
            if not app:
                return None
            return Action("focus_window", f"focus_window:{app.lower()}", lambda: wa.focus_window(app), display_arg=app)
        if action == "open_folder":
            folder = str(data.get("folder", "")).strip()
            if not folder:
                return None
            return Action(
                "open_folder", f"open_folder:{folder.lower()}", lambda: wa.open_folder(folder), display_arg=folder
            )
        if action == "find_file":
            query = str(data.get("query", "")).strip()
            if not query:
                return None
            return Action("find_file", f"find_file:{query.lower()}", lambda: wa.find_file(query), display_arg=query)
        return None


def _worth_asking_llm(text: str) -> bool:
    """False for a bare wake word or acknowledgement — nothing to execute."""
    cleaned = re.sub(r"[^a-zäöüß0-9\s]", "", text.lower())
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned not in {
        "",
        "please",
        "now",
        "thanks",
        "thank you",
        "okay",
        "ok",
        "jarvis",
        "hey jarvis",
        "hallo",
        "hey",
        "hallo jarvis",
        "danke",
        "ja",
        "nein",
        "yes",
        "no",
    }


class IntentEngine:
    def __init__(self, on_action_started: ActionCallback, use_llm_fallback: bool = True) -> None:
        self._on_action_started = on_action_started
        answers.set_done_callback(lambda: self._on_action_started(answers.timer_done_label()))
        self._fired: dict[int, set[str]] = {}
        self._executor = ThreadPoolExecutor(max_workers=2)
        self._llm = LLMFallbackClassifier() if use_llm_fallback else None

    def _already_fired(self, utterance_id: int, signature: str) -> bool:
        return signature in self._fired.get(utterance_id, set())

    def _mark_fired(self, utterance_id: int, signature: str) -> None:
        self._fired.setdefault(utterance_id, set()).add(signature)

    def on_transcript(self, text: str, is_final: bool, speech_final: bool, utterance_id: int) -> None:
        # Lowercased/punctuation-stripped for trigger matching only - see
        # _normalize_transcript. Casing/punctuation doesn't matter for any
        # of the other captured args (Google search is case-insensitive,
        # and confirmation badges are uppercased anyway), so this is safe
        # everywhere except write_in_app's dictated content, handled below.
        normalized = _normalize_transcript(text)

        if speech_final:
            commands = macros.match(normalized)
            if commands:
                self._mark_fired(utterance_id, f"macro:{normalized}")
                print(f"[intent] macro {normalized!r} -> {commands}")
                for command in commands:
                    for part in _expand_conjunctions(_normalize_transcript(command)):
                        self._handle_command(part, command, True, True, utterance_id)
                self._fired.pop(utterance_id, None)
                return

        # "open spotify and chrome" -> two commands, each handled on its own.
        for command in _expand_conjunctions(normalized):
            self._handle_command(command, text, is_final, speech_final, utterance_id)

        # Regex missed the whole utterance. Ask the LLM with the original
        # wording — the rewritten text is for the patterns, and a half-rewritten
        # clause ("mal louder machen") is harder to classify than what was said.
        if speech_final and not self._fired.get(utterance_id) and _worth_asking_llm(text):
            if self._llm is not None:
                action = self._llm.classify(text.strip())
                if action is not None:
                    self._fire(action, text, speech_final, utterance_id)
            if not self._fired.get(utterance_id):
                print(f"[intent] unmatched: {text.strip()!r}")

        if speech_final:
            self._fired.pop(utterance_id, None)

    def _handle_command(
        self, normalized: str, original_text: str, is_final: bool, speech_final: bool, utterance_id: int
    ) -> None:
        already_fired = lambda sig: self._already_fired(utterance_id, sig)  # noqa: E731
        # A single command can hold several actions ("open spotify and play
        # X" -> open_app, then spotify_play): keep asking for the next NEW
        # action until nothing is left. Capped so a pathological transcript
        # can't spin.
        for _ in range(4):
            action = _match_regex(normalized, speech_final, already_fired)
            if action is None:
                break
            if not self._fire(action, original_text, speech_final, utterance_id):
                break

    def _fire(self, action: Action, original_text: str, speech_final: bool, utterance_id: int) -> bool:
        """Executes `action` unless it's deferred/deduped. Returns True if it
        fired (so the caller may look for a further action in the same
        command)."""
        # Same reasoning as _FINAL_ONLY_PATTERNS: open-ended captured text
        # (dictation, search queries) must not fire on a growing partial
        # transcript, regardless of which matcher (regex or LLM) found it.
        if action.name in ("write_in_app", "browser_search", "spotify_play") and not speech_final:
            return False

        if action.name == "write_in_app":
            # Re-extract from the ORIGINAL text: typing everything in
            # lowercase with punctuation stripped would be a real
            # regression for dictated notes, unlike the other actions.
            original_match = _WRITE_PATTERN.search(original_text)
            if original_match:
                original_content = original_match.group(1).strip()
                action = Action(
                    name="write_in_app",
                    signature="write_in_app",
                    run=lambda: wa.write_in_app(original_content),
                )
        if self._already_fired(utterance_id, action.signature):
            return False

        self._mark_fired(utterance_id, action.signature)
        label = _ACTION_LABELS.get(action.name, "[WORKING]").format(arg=action.display_arg.upper())
        print(f"[intent] #{utterance_id} firing {action.signature}  {label}")
        self._on_action_started(label)
        self._executor.submit(self._execute, action, label)
        return True

    def _execute(self, action: Action, fallback_label: str) -> None:
        try:
            result_label = action.run()
        except wa.ActionError as exc:
            result_label = f"Couldn't do that: {exc}"
        except Exception as exc:  # noqa: BLE001 - surface any failure to the pill, don't crash the pipeline
            result_label = f"Error: {exc}"
        if result_label and result_label != fallback_label:
            print(f"[intent] result: {result_label}")
            self._on_action_started(result_label)

    def shutdown(self, wait: bool = True) -> None:
        # wait=True by default: the executor's worker threads must be fully
        # joined *before* QApplication starts tearing down, otherwise a
        # worker thread finishing an action (and emitting a Qt signal) can
        # race the interpreter's native cleanup and crash the process. Call
        # this before app.quit(), not after.
        answers.cancel_timer()
        self._executor.shutdown(wait=wait)
