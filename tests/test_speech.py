"""Spoken confirmation phrases and echo filtering."""

from __future__ import annotations

import os
import unittest

import speech


class Phrases(unittest.TestCase):
    def test_german_confirmations(self) -> None:
        self.assertEqual(speech.phrase_for_label("[LAUNCHED CHROME]"), "Chrome wird geöffnet.")
        self.assertEqual(speech.phrase_for_label("[LAUNCHED CAMERA]"), "Die Kamera wird geöffnet.")
        self.assertEqual(speech.phrase_for_label("[CLOSING SPOTIFY]"), "Spotify wird geschlossen.")
        self.assertEqual(
            speech.phrase_for_label("[SEARCHING FOR PYTHON TUTORIALS]"),
            "Ich suche nach Python Tutorials.",
        )
        self.assertEqual(speech.phrase_for_label("[NAVIGATING TO GITHUB.COM]"), "Ich öffne github.com.")
        self.assertEqual(speech.phrase_for_label("[PLAYING BLAZE]"), "Ich spiele Blaze.")
        self.assertEqual(speech.phrase_for_label("[PAUSED]"), "Pause.")
        self.assertEqual(speech.phrase_for_label("[VOLUME 30%]"), "Lautstärke 30.")
        self.assertEqual(speech.phrase_for_label("[VOLUME UP]"), "Lauter.")
        self.assertEqual(speech.phrase_for_label("[SHUFFLE ON]"), "Zufall an.")
        self.assertEqual(speech.phrase_for_label("[CAPTURING PHOTO]"), "Foto wird aufgenommen.")
        self.assertEqual(speech.phrase_for_label("Couldn't do that: missing"), "Das hat nicht geklappt.")

    def test_result_text_stays_silent(self) -> None:
        self.assertIsNone(speech.phrase_for_label("Opening Chrome"))
        self.assertIsNone(speech.phrase_for_label("Saved photo to C:\\Pictures\\a.jpg"))

    def test_english_when_language_is_en(self) -> None:
        self.assertEqual(speech.phrase_for_label("[LAUNCHED CHROME]", "en"), "Opening Chrome.")
        self.assertEqual(speech.phrase_for_label("[PAUSED]", "en"), "Paused.")

    def test_own_voice_is_an_echo(self) -> None:
        spoken = "Chrome wird geöffnet."
        self.assertTrue(speech.looks_like_echo("chrome wird geöffnet", spoken))
        self.assertFalse(speech.looks_like_echo("öffne firefox", spoken))


class DisabledSpeaker(unittest.TestCase):
    def test_speak_is_a_no_op_when_muted(self) -> None:
        previous = os.environ.get("TTS_ENABLED")
        os.environ["TTS_ENABLED"] = "0"
        try:
            speaker = speech.Speaker()
            speaker.start()
            self.assertFalse(speaker.speak("Hallo"))
        finally:
            if previous is None:
                os.environ.pop("TTS_ENABLED", None)
            else:
                os.environ["TTS_ENABLED"] = previous


if __name__ == "__main__":
    unittest.main()
