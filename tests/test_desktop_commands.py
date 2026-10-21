"""System volume, screenshot, and foreground-window commands."""

from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import intent_engine as ie
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


class DesktopCommands(unittest.TestCase):
    def test_system_volume_is_separate_from_spotify(self) -> None:
        self.assertEqual(_actions("Lautstärke auf 30")[0].signature, "spotify_volume:30")
        self.assertEqual(_actions("leiser")[0].name, "spotify_volume_down")
        self.assertEqual(_actions("louder")[0].name, "spotify_volume_up")

        german = _actions("Systemlautstärke auf 40")
        self.assertEqual(german[0].signature, "system_volume:40")
        self.assertEqual(_actions("Computer lauter")[0].name, "system_volume_up")
        self.assertEqual(_actions("mach den PC leiser")[0].name, "system_volume_down")
        self.assertEqual(_actions("set the system volume to 15")[0].signature, "system_volume:15")
        self.assertEqual(_actions("Systemlautstärke auf 250")[0].signature, "system_volume:100")

    def test_numeric_volume_waits_until_the_utterance_ends(self) -> None:
        self.assertEqual(_actions("system volume 4", speech_final=False), [])
        self.assertEqual(_actions("system volume 40")[0].signature, "system_volume:40")

    def test_mute_screenshot_and_window(self) -> None:
        self.assertEqual(_actions("Stumm")[0].name, "system_mute")
        self.assertEqual(_actions("Ton an")[0].name, "system_unmute")
        self.assertEqual(_actions("mute the computer")[0].name, "system_mute")
        self.assertEqual(_actions("unmute")[0].name, "system_unmute")

        self.assertEqual(_actions("mach einen Screenshot")[0].name, "screenshot")
        self.assertEqual(_actions("Bildschirmfoto")[0].name, "screenshot")
        self.assertEqual(_actions("take a screenshot", speech_final=False)[0].name, "screenshot")

        self.assertEqual(_actions("minimiere das Fenster")[0].name, "window_minimize")
        self.assertEqual(_actions("Maximiere")[0].name, "window_maximize")
        self.assertEqual(_actions("Vollbild")[0].name, "window_maximize")
        self.assertEqual(_actions("Fenster wiederherstellen")[0].name, "window_restore")
        self.assertEqual(_actions("Fenster nach links")[0].name, "window_left")
        self.assertEqual(_actions("snap the window to the right")[0].name, "window_right")

    def test_casual_phrasing_still_runs(self) -> None:
        self.assertEqual(_actions("minimiere mal bitte das Fenster")[0].name, "window_minimize")
        self.assertEqual(_actions("kannst du mal einen Screenshot machen")[0].name, "screenshot")
        self.assertEqual(_actions("schieb das Fenster nach links")[0].name, "window_left")
        self.assertEqual(_actions("dreh den Ton lauter")[0].name, "system_volume_up")
        self.assertEqual(_actions("lauter")[0].name, "spotify_volume_up")

        names = [action.name for action in _actions("öffne Chrome und mach einen Screenshot")]
        self.assertEqual(names, ["open_app", "screenshot"])

        search = _actions("open chrome and search for cats")
        self.assertEqual(search[0].name, "open_app")
        self.assertEqual(search[1].name, "browser_search")
        self.assertIn("chrome", search[1].signature)

    def test_a_search_that_mentions_screenshot_stays_a_search(self) -> None:
        search = _actions("such nach Screenshot Anleitung")
        self.assertEqual(search[0].name, "browser_search")
        self.assertIn("screenshot", search[0].display_arg.lower())

    def test_minimize_runs_beside_open_and_is_not_an_app(self) -> None:
        actions = _actions("open chrome and minimize", speech_final=False)
        self.assertEqual([action.name for action in actions], ["open_app", "window_minimize"])
        self.assertTrue(all(":minimize" not in action.signature for action in actions))


class LlmDesktopActions(unittest.TestCase):
    def test_json_maps_without_running(self) -> None:
        volume = ie.LLMFallbackClassifier._action_from_json({"action": "system_volume", "percent": 12})
        assert volume is not None
        self.assertEqual(volume.signature, "system_volume:12")

        step = ie.LLMFallbackClassifier._action_from_json({"action": "system_volume", "direction": "down"})
        assert step is not None
        self.assertEqual(step.name, "system_volume_down")

        shot = ie.LLMFallbackClassifier._action_from_json({"action": "screenshot"})
        assert shot is not None
        self.assertEqual(shot.name, "screenshot")

        window = ie.LLMFallbackClassifier._action_from_json({"action": "window", "op": "left"})
        assert window is not None
        self.assertEqual(window.name, "window_left")

        self.assertIsNone(ie.LLMFallbackClassifier._action_from_json({"action": "window", "op": "nope"}))
        self.assertIsNone(
            ie.LLMFallbackClassifier._action_from_json({"action": "system_volume", "percent": "loud"})
        )


class DesktopActionsDryRun(unittest.TestCase):
    def setUp(self) -> None:
        self._previous = wa.DRY_RUN
        wa.DRY_RUN = True

    def tearDown(self) -> None:
        wa.DRY_RUN = self._previous

    def test_actions_do_not_touch_the_desktop(self) -> None:
        self.assertEqual(wa.set_system_volume(40), "System volume 40%")
        self.assertEqual(wa.set_system_volume(250), "System volume 100%")
        self.assertEqual(wa.change_system_volume(wa.SYSTEM_VOLUME_STEP), "System volume up")
        self.assertEqual(wa.set_system_mute(True), "Muted")
        self.assertEqual(wa.minimize_window(), "Minimized")
        self.assertEqual(wa.maximize_window(), "Maximized")
        self.assertEqual(wa.restore_window(), "Restored")
        self.assertEqual(wa.snap_window("left"), "Snapped left")
        self.assertIn("screenshot_", wa.take_screenshot().lower())


class DesktopOnThisMachine(unittest.TestCase):
    def test_reads_master_volume_and_writes_the_same_level(self) -> None:
        previous = wa.DRY_RUN
        wa.DRY_RUN = False
        try:
            with wa._MasterVolume() as volume:
                level = volume.get_scalar()
                volume.set_scalar(level)
                again = volume.get_scalar()
                muted = volume.get_mute()
                volume.set_mute(muted)
                muted_again = volume.get_mute()
        finally:
            wa.DRY_RUN = previous
        self.assertGreaterEqual(level, 0.0)
        self.assertLessEqual(level, 1.0)
        self.assertAlmostEqual(again, level, delta=0.02)
        self.assertEqual(muted_again, muted)

    def test_screenshot_file_is_a_png(self) -> None:
        previous = wa.DRY_RUN
        wa.DRY_RUN = False
        try:
            with tempfile.TemporaryDirectory() as tmp:
                message = wa.take_screenshot(save_dir=Path(tmp))
                files = list(Path(tmp).glob("screenshot_*.png"))
                self.assertIn("Saved screenshot", message)
                self.assertEqual(len(files), 1)
                self.assertGreater(files[0].stat().st_size, 1000)
        finally:
            wa.DRY_RUN = previous

    def test_finds_a_foreground_window(self) -> None:
        hwnd = wa._foreground_hwnd()
        self.assertIsInstance(hwnd, int)
        self.assertGreater(hwnd, 0)


if __name__ == "__main__":
    unittest.main()
