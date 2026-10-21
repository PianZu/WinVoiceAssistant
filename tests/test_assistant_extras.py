"""Spoken answers, folders, file search, focus, macros, and the wake-word scorer."""

from __future__ import annotations

import os
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import answers
import intent_engine as ie
import listening_control
import macros
import wake_model
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


class SpokenAnswers(unittest.TestCase):
    def test_time_and_battery_sentences(self) -> None:
        morning = datetime(2026, 10, 21, 9, 0)
        later = datetime(2026, 10, 21, 9, 5)
        self.assertEqual(answers.current_time_sentence("de", morning), "Es ist 9 Uhr.")
        self.assertEqual(answers.current_time_sentence("de", later), "Es ist 9 Uhr 5.")
        self.assertEqual(answers.current_time_sentence("en", later), "It's 9:05 AM.")
        self.assertEqual(answers.battery_sentence("de", 80, True), "Der Akku ist bei 80 Prozent und lädt.")
        self.assertEqual(answers.battery_sentence("en", None, None), "I can't see a battery.")
        percent, _charging = answers.read_battery()
        self.assertTrue(percent is None or 0 <= percent <= 100)

    def test_timer_sentences_and_cancel(self) -> None:
        previous = os.environ.get("STT_LANGUAGE")
        os.environ["STT_LANGUAGE"] = "en"
        try:
            self.assertEqual(answers.timer_set_sentence(300, "de"), "Timer auf 5 Minuten.")
            self.assertEqual(answers.start_timer(60), "SAY:Timer set for 1 minute.")
            self.assertEqual(answers.cancel_timer(), "SAY:Timer cancelled.")
            self.assertEqual(answers.cancel_timer(), "SAY:No timer is running.")
        finally:
            answers.cancel_timer()
            if previous is None:
                os.environ.pop("STT_LANGUAGE", None)
            else:
                os.environ["STT_LANGUAGE"] = previous

    def test_timer_done_notifies_once(self) -> None:
        done: list[str] = []
        answers.set_done_callback(lambda: done.append("done"))
        try:
            answers._fire_timer()
            self.assertEqual(done, ["done"])
        finally:
            answers.set_done_callback(lambda: None)
            answers.cancel_timer()


class Commands(unittest.TestCase):
    def test_german_and_english_phrases(self) -> None:
        self.assertEqual(_actions("wie spät ist es")[0].name, "answer_time")
        self.assertEqual(_actions("what time is it")[0].name, "answer_time")
        self.assertEqual(_actions("Akku")[0].name, "answer_battery")
        self.assertEqual(_actions("was läuft")[0].name, "answer_now_playing")
        self.assertEqual(_actions("what's playing")[0].name, "answer_now_playing")
        self.assertEqual(_actions("Timer auf 5 Minuten")[0].signature, "timer_set:300")
        self.assertEqual(_actions("timer 5 minutes", speech_final=False), [])
        self.assertEqual(_actions("timer abbrechen")[0].name, "timer_cancel")
        self.assertEqual(_actions("öffne Downloads")[0].signature, "open_folder:downloads")
        self.assertEqual(_actions("öffne Dokumente")[0].signature, "open_folder:dokumente")
        self.assertEqual(_actions("wechsle zu chrome")[0].signature, "focus_window:chrome")
        self.assertEqual(_actions("such die datei rechnung")[0].signature, "find_file:rechnung")
        self.assertEqual(_actions("find the file budget")[0].name, "find_file")
        self.assertEqual(_actions("such nach Screenshot Anleitung")[0].name, "browser_search")
        self.assertEqual(_actions("find cats")[0].name, "browser_search")
        search = _actions("open chrome and search for cats")
        self.assertEqual(search[0].name, "open_app")
        self.assertEqual(search[1].name, "browser_search")
        self.assertIn("chrome", search[1].signature)

    def test_focus_and_file_search_do_not_touch_the_desktop(self) -> None:
        previous = wa.DRY_RUN
        previous_lang = os.environ.get("STT_LANGUAGE")
        wa.DRY_RUN = True
        os.environ["STT_LANGUAGE"] = "de"
        try:
            self.assertEqual(wa.focus_window("chrome"), "SAY:Chrome ist vorn.")
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                (root / "notes.txt").write_text("x", encoding="utf-8")
                (root / "rechnung-2024.pdf").write_text("x", encoding="utf-8")
                message = wa.find_file("rechnung", roots=[root])
            self.assertIn("rechnung-2024.pdf", message)
        finally:
            wa.DRY_RUN = previous
            if previous_lang is None:
                os.environ.pop("STT_LANGUAGE", None)
            else:
                os.environ["STT_LANGUAGE"] = previous_lang


class Macros(unittest.TestCase):
    def tearDown(self) -> None:
        macros.load()

    def test_example_phrase_and_plain_open(self) -> None:
        self.assertEqual(
            macros.match("starte arbeitsmodus"),
            ["open slack", "open chrome", "open spotify"],
        )
        self.assertEqual(macros.match("open arbeitsmodus"), macros.match("arbeitsmodus"))
        self.assertIsNone(macros.match("open chrome"))

    def test_user_file_replaces_the_example(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "macros.txt"
            path.write_text("feierabend = open notepad\n", encoding="utf-8")
            macros.load(path)
            self.assertEqual(macros.match("feierabend"), ["open notepad"])
            self.assertIsNone(macros.match("arbeitsmodus"))

    def test_phrase_runs_each_command(self) -> None:
        labels: list[str] = []
        previous = wa.DRY_RUN
        wa.DRY_RUN = True
        engine = ie.IntentEngine(on_action_started=labels.append, use_llm_fallback=False)
        try:
            engine.on_transcript("starte Arbeitsmodus", True, True, 7)
        finally:
            engine.shutdown(wait=True)
            wa.DRY_RUN = previous
            answers.cancel_timer()
        self.assertIn("[LAUNCHED SLACK]", labels)
        self.assertIn("[LAUNCHED CHROME]", labels)
        self.assertIn("[LAUNCHED SPOTIFY]", labels)


class WakeScore(unittest.TestCase):
    def test_one_loud_frame_wakes(self) -> None:
        seen: list[int] = []

        def predict(frame):
            seen.append(len(frame))
            return {"hey_jarvis": 0.9}

        detector = wake_model.HeyJarvisDetector(predict=predict)
        quiet = b"\x00\x00" * 100
        self.assertFalse(detector.consume(quiet))
        self.assertTrue(detector.consume(b"\x00\x00" * 1180))
        self.assertEqual(seen, [wake_model.FRAME])

    def test_disabled_detector_stays_asleep(self) -> None:
        self.assertFalse(wake_model.HeyJarvisDetector().consume(b"\x00\x00" * wake_model.FRAME))

    def test_strip_keeps_text_when_the_name_is_missing(self) -> None:
        gate = listening_control.WakeWordGate()
        self.assertEqual(gate.strip("hey jarvis öffne chrome"), "öffne chrome")
        self.assertEqual(gate.strip("öffne chrome"), "öffne chrome")


if __name__ == "__main__":
    unittest.main()
