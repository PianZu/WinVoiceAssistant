"""Speech clips close on silence and when listening stops."""

from __future__ import annotations

import unittest

import numpy as np

import voice_stream


def _block(level: int) -> bytes:
    return np.full(voice_stream.BLOCK_SIZE, level, dtype=np.int16).tobytes()


class UtteranceSegmenter(unittest.TestCase):
    def test_flush_keeps_speech_that_had_no_trailing_silence(self) -> None:
        segmenter = voice_stream.UtteranceSegmenter()
        loud = _block(8000)
        closed = None
        for _ in range(6):
            closed = segmenter.push(loud)
        self.assertIsNone(closed)
        pcm = segmenter.flush()
        self.assertIsNotNone(pcm)
        assert pcm is not None
        self.assertGreater(len(pcm), 0)
        self.assertIsNone(segmenter.flush())

    def test_silence_closes_the_phrase(self) -> None:
        segmenter = voice_stream.UtteranceSegmenter()
        for _ in range(6):
            self.assertIsNone(segmenter.push(_block(8000)))
        closed = None
        for _ in range(20):
            closed = segmenter.push(_block(0))
            if closed:
                break
        self.assertIsNotNone(closed)

    def test_room_noise_is_not_a_command(self) -> None:
        segmenter = voice_stream.UtteranceSegmenter()
        for _ in range(10):
            segmenter.push(_block(0))
        self.assertIsNone(segmenter.flush())


if __name__ == "__main__":
    unittest.main()
