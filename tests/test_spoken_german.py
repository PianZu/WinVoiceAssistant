"""German speech commands rewrite onto the existing English matchers."""

from __future__ import annotations

import os
import unittest

import intent_engine as ie
import listening_control
import voice_stream
import windows_actions as wa


def _actions(text: str, speech_final: bool = True) -> list[ie.Action]:
    found: list[ie.Action] = []
    for part in ie._expand_conjunctions(ie._normalize_transcript(text)):
        fired: set[str] = set()
        for _ in range(4):
            action = ie._match_regex(part, speech_final, lambda sig: sig in fired)
            if action is None:
                break
            fired.add(action.signature)
            found.append(action)
    return found


class GermanCommands(unittest.TestCase):
    def test_open_and_close(self) -> None:
        opened = _actions("öffne Chrome", speech_final=False)
        self.assertEqual([a.signature for a in opened], ["open_app:chrome"])

        closed = _actions("schließ Spotify", speech_final=False)
        self.assertEqual([a.signature for a in closed], ["close_app:spotify"])

    def test_articles_and_politeness(self) -> None:
        actions = _actions("kannst du bitte den Rechner starten")
        self.assertEqual(actions[0].name, "open_app")
        self.assertEqual(wa.resolve_app(actions[0].display_arg), "calc")

        camera = _actions("mach die Kamera auf", speech_final=False)
        self.assertEqual(wa.resolve_app(camera[0].display_arg), "camera")

    def test_search_play_and_dictation(self) -> None:
        search = _actions("such nach Python Tutorials")
        self.assertEqual(search[0].name, "browser_search")
        self.assertIn("python tutorials", search[0].display_arg.lower())

        play = _actions("spiel Blaze ab")
        self.assertEqual(play[0].signature, "spotify_play:blaze")

        write = _actions("schreib Hallo Welt")
        self.assertEqual(write[0].name, "write_in_app")
        captured = ie._WRITE_PATTERN.search("Schreib mal Hallo Welt")
        self.assertIsNotNone(captured)
        assert captured is not None
        self.assertEqual(captured.group(1), "Hallo Welt")

    def test_transport_and_volume(self) -> None:
        self.assertEqual(_actions("Pause die Musik")[0].name, "spotify_pause")
        self.assertEqual(_actions("nächster Song")[0].name, "spotify_next")
        self.assertEqual(_actions("spiel weiter")[0].name, "spotify_resume")
        self.assertEqual(_actions("Lautstärke auf 30")[0].signature, "spotify_volume:30")
        self.assertEqual(_actions("leiser")[0].name, "spotify_volume_down")
        self.assertEqual(_actions("Zufall an")[0].signature, "spotify_shuffle:on")

    def test_compound_and_domain(self) -> None:
        names = [a.signature for a in _actions("öffne Spotify und Chrome", speech_final=False)]
        self.assertEqual(names, ["open_app:spotify", "open_app:chrome"])

        url = _actions("gehe zu github punkt de", speech_final=False)
        self.assertEqual(url[0].name, "navigate_url")
        self.assertEqual(url[0].display_arg.lower(), "github.de")

    def test_english_commands_still_match(self) -> None:
        self.assertEqual(_actions("open chrome", speech_final=False)[0].signature, "open_app:chrome")
        self.assertEqual(_actions("play stray kids")[0].signature, "spotify_play:stray kids")
        self.assertEqual(_actions("search for cats")[0].name, "browser_search")

    def test_wake_phrase_hallo_jarvis(self) -> None:
        gate = listening_control.WakeWordGate()
        remainder = gate.process("hallo Jarvis öffne Chrome", utterance_id=1, speech_final=True)
        self.assertIsNotNone(remainder)
        assert remainder is not None
        self.assertIn("öffne", remainder.lower())


class WhisperModelSelection(unittest.TestCase):
    def test_english_only_checkpoint_is_swapped_for_german(self) -> None:
        previous_lang = os.environ.get("STT_LANGUAGE")
        previous_model = os.environ.get("WHISPER_MODEL")
        os.environ["STT_LANGUAGE"] = "de"
        os.environ["WHISPER_MODEL"] = "small.en"
        try:
            self.assertEqual(voice_stream.resolve_stt_language(), "de")
            self.assertEqual(voice_stream.resolve_whisper_model(), "small")
        finally:
            if previous_lang is None:
                os.environ.pop("STT_LANGUAGE", None)
            else:
                os.environ["STT_LANGUAGE"] = previous_lang
            if previous_model is None:
                os.environ.pop("WHISPER_MODEL", None)
            else:
                os.environ["WHISPER_MODEL"] = previous_model


if __name__ == "__main__":
    unittest.main()
