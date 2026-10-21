"""
voice_stream.py

Streams microphone audio to a speech-to-text backend and reports interim +
final transcripts through a single callback interface, so intent_engine.py
doesn't need to know which backend is active.

Two backends:
  - DeepgramStreamer: wss://api.deepgram.com Nova-2, interim_results=True.
    Real streaming, ~150-300ms partials.
  - FasterWhisperStreamer: local fallback with no network dependency. It is
    NOT true streaming ASR (Whisper doesn't support that natively) - it runs
    short-window inference on a rolling audio buffer, which trades some
    latency for zero API cost / offline operation. Good enough for demoing
    the mid-sentence intent pipeline without credentials.

Both backends are driven from an asyncio event loop that should be run on a
background thread (see main.py); they call `on_transcript` which itself must
be safe to call from that thread (main.py marshals it onto the Qt thread via
a pyqtSignal before touching any widgets).
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import os
import queue
import re
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, Optional, Union

import time

import numpy as np
import sounddevice as sd

SAMPLE_RATE = 16000
CHANNELS = 1
BLOCK_MS = 30
BLOCK_SIZE = int(SAMPLE_RATE * BLOCK_MS / 1000)

# on_transcript(text: str, is_final: bool, speech_final: bool, utterance_id: int)
TranscriptCallback = Callable[[str, bool, bool, int], None]

# YouTube-trained Whisper invents these constantly over music. Drop the
# whole transcript if it's only this junk (a real command plus this is rare).
_HALLUCINATION = re.compile(
    r"^\s*(?:"
    r"thanks for watching(?:\s+.*)?|"
    r"don't forget to subscribe(?:\s+.*)?|"
    r"like and subscribe|"
    r"(?:oh,?\s+){3,}oh\.?|"
    r"(?:yeah,?\s+){3,}yeah\.?|"
    r"vielen dank fürs zuschauen(?:\s+.*)?|"
    r"untertitel(?:ung)?(?:\s+.*)?"
    r")\s*$",
    re.IGNORECASE,
)


def _is_hallucination(text: str) -> bool:
    return bool(text) and bool(_HALLUCINATION.match(text))


def resolve_stt_language() -> str:
    """BCP-47-ish short code for the speech backend. `de` or `en`."""
    return (os.environ.get("STT_LANGUAGE") or "en").strip().lower() or "en"


def resolve_whisper_model(language: Optional[str] = None) -> str:
    """Local Whisper model name. `*.en` checkpoints only speak English, so a
    German (or any non-English) language setting swaps them for the
    multilingual checkpoint with the same size (`small.en` -> `small`)."""
    language = resolve_stt_language() if language is None else language
    model = (os.environ.get("WHISPER_MODEL") or "small.en").strip() or "small.en"
    if language != "en" and model.endswith(".en"):
        multilingual = model[: -len(".en")]
        print(
            f"[whisper] {model} is English-only; using {multilingual} "
            f"because STT_LANGUAGE={language}"
        )
        return multilingual
    return model


def resolve_input_device(spec: Optional[str]) -> Optional[Union[int, str]]:
    """Turn the MIC_DEVICE setting into something sounddevice accepts.

    - unset / empty  -> None (Windows default recording device)
    - an integer     -> that device index, as listed by `python -m sounddevice`
    - any other text -> case-insensitive substring match against the names of
                        all *input* devices; the first hit wins. Matching by
                        name is preferred over a bare index because indices
                        shift whenever a USB device is (un)plugged.
    """
    if not spec or not spec.strip():
        return None
    spec = spec.strip()
    if spec.isdigit():
        return int(spec)
    needle = spec.lower()
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] > 0 and needle in dev["name"].lower():
            return idx
    available = [d["name"] for d in sd.query_devices() if d["max_input_channels"] > 0]
    raise RuntimeError(
        f"MIC_DEVICE={spec!r} matches no input device. Available inputs: {available}"
    )


class _MicSource:
    """Wraps sounddevice's callback-based capture into an asyncio queue of PCM16 bytes."""

    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self._loop = loop
        self._queue: "asyncio.Queue[bytes]" = asyncio.Queue()
        self._stream: Optional[sd.RawInputStream] = None

    def _callback(self, indata, frames, time_info, status) -> None:
        if status:
            print(f"[voice_stream] mic status: {status}")
        data = bytes(indata)
        self._loop.call_soon_threadsafe(self._queue.put_nowait, data)

    def start(self) -> None:
        device = resolve_input_device(os.environ.get("MIC_DEVICE"))
        if device is not None:
            print(f"[voice_stream] using mic: {sd.query_devices(device)['name']}")
        self._stream = sd.RawInputStream(
            device=device,
            samplerate=SAMPLE_RATE,
            blocksize=BLOCK_SIZE,
            channels=CHANNELS,
            dtype="int16",
            callback=self._callback,
        )
        self._stream.start()

    def stop(self) -> None:
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    async def get(self) -> bytes:
        return await self._queue.get()

    def wake(self) -> None:
        """Unblock get() from another thread so a stop can flush the buffer."""
        self._loop.call_soon_threadsafe(self._queue.put_nowait, b"")


class UtteranceSegmenter:
    """Turns mic blocks into one PCM clip per spoken phrase.

    A phrase closes after a short silence, or when flush() is called because
    listening stopped — otherwise the audio recorded so far was thrown away
    and the command never ran.
    """

    def __init__(self) -> None:
        self.buffer = bytearray()
        self.silence_accum = 0.0
        self.heard_speech = False
        self.speech_run = 0.0
        self._block_seconds = BLOCK_SIZE / SAMPLE_RATE
        self._window_bytes = int(FasterWhisperStreamer.WINDOW_S * SAMPLE_RATE) * 2
        self._max_bytes = int(FasterWhisperStreamer.MAX_UTTERANCE_S * SAMPLE_RATE) * 2
        self._rms_hist: deque[float] = deque(
            maxlen=max(8, int(FasterWhisperStreamer.AMBIENT_WINDOW_S / self._block_seconds))
        )

    def push(self, chunk: bytes) -> Optional[bytes]:
        if not chunk:
            return None
        self.buffer.extend(chunk)
        samples = np.frombuffer(chunk, dtype=np.int16)
        rms = float(np.sqrt(np.mean(samples.astype(np.float64) ** 2))) if len(samples) else 0.0

        # Music stays at full volume, including while you sing. A command
        # is a jump above that level, so the song itself never counts as
        # silence and the phrase still closes when you stop talking.
        if not self.heard_speech:
            self._rms_hist.append(rms)
        if self._above_ambient(rms):
            self.speech_run += self._block_seconds
            self.silence_accum = 0.0
            if self.speech_run >= FasterWhisperStreamer.SPEECH_HOLD_S:
                if not self.heard_speech:
                    preroll = int(0.35 * SAMPLE_RATE) * 2
                    if len(self.buffer) > preroll:
                        del self.buffer[:-preroll]
                self.heard_speech = True
        else:
            self.speech_run = 0.0
            if self.heard_speech:
                self.silence_accum += self._block_seconds

        if not self.heard_speech and len(self.buffer) > self._window_bytes:
            del self.buffer[:-self._window_bytes]
            return None

        should_finalize = self.heard_speech and (
            self.silence_accum >= FasterWhisperStreamer.SILENCE_S or len(self.buffer) >= self._max_bytes
        )
        if not should_finalize:
            return None
        return self._take()

    def flush(self) -> Optional[bytes]:
        if not self.heard_speech or not self.buffer:
            self._clear()
            return None
        return self._take()

    def _take(self) -> bytes:
        pcm = bytes(self.buffer)
        self._clear()
        return pcm

    def _clear(self) -> None:
        self.buffer.clear()
        self.silence_accum = 0.0
        self.heard_speech = False
        self.speech_run = 0.0

    def _above_ambient(self, rms: float) -> bool:
        if len(self._rms_hist) < 8:
            floor = float(FasterWhisperStreamer.SILENCE_RMS_THRESHOLD)
        else:
            ordered = sorted(self._rms_hist)
            floor = ordered[len(ordered) // 5]
        return rms > max(float(FasterWhisperStreamer.SILENCE_RMS_THRESHOLD), floor * 1.75, floor + 320)


class DeepgramStreamer:
    def __init__(self, api_key: str) -> None:
        self._api_key = api_key
        self._running = False

    async def run(self, on_transcript: TranscriptCallback) -> None:
        import websockets

        language = resolve_stt_language()
        url = (
            "wss://api.deepgram.com/v1/listen"
            f"?model=nova-2&language={language}&interim_results=true&endpointing=300"
            f"&encoding=linear16&sample_rate={SAMPLE_RATE}&channels={CHANNELS}"
        )
        headers = {"Authorization": f"Token {self._api_key}"}

        loop = asyncio.get_running_loop()
        mic = _MicSource(loop)
        self._running = True
        utterance_id = 0

        # Start capturing *before* the network handshake, not after: in
        # push-to-talk mode a fresh connection is opened on every hotkey
        # press, and the handshake's round trip (plus TLS) can easily eat
        # the first word or two of a short press-and-speak utterance if the
        # mic only starts once connect() returns. Audio queues up locally in
        # the meantime and sender() drains it as soon as the socket is up.
        mic.start()
        try:
            async with websockets.connect(url, additional_headers=headers) as ws:

                async def sender():
                    while self._running:
                        chunk = await mic.get()
                        await ws.send(chunk)

                async def receiver():
                    nonlocal utterance_id
                    async for message in ws:
                        payload = json.loads(message)
                        if payload.get("type") != "Results":
                            continue
                        alt = payload["channel"]["alternatives"][0]
                        text = alt.get("transcript", "")
                        if not text:
                            continue
                        is_final = payload.get("is_final", False)
                        speech_final = payload.get("speech_final", False)
                        on_transcript(text, is_final, speech_final, utterance_id)
                        if speech_final:
                            utterance_id += 1

                receiver_task = asyncio.ensure_future(receiver())
                try:
                    await sender()  # returns as soon as stop() flips self._running
                finally:
                    # Close immediately rather than waiting for Deepgram's own
                    # idle timeout (~10s) to notice the mic went silent - that
                    # delay was showing up as a "1011 internal error" log line
                    # on every single PTT release.
                    await ws.close()
                    receiver_task.cancel()
                    with contextlib.suppress(asyncio.CancelledError):
                        await receiver_task
        finally:
            mic.stop()

    def stop(self) -> None:
        self._running = False


class FasterWhisperStreamer:
    """Local Whisper. One transcription when the user stops speaking.

    A rolling interim pass was slower than the utterance itself on this
    CPU: the final result waited behind that first pass, so a short
    command landed many seconds late. Commands now wait out a short
    silence, then run Whisper exactly once.
    """

    WINDOW_S = 1.0  # quiet audio kept so the start of a word isn't clipped
    SILENCE_S = 0.3
    SILENCE_RMS_THRESHOLD = 300  # int16 RMS, floor when the room is quiet
    # Music used to hold an utterance open until this fired.
    MAX_UTTERANCE_S = 3.5
    SPEECH_HOLD_S = 0.12  # ignore single drum hits
    AMBIENT_WINDOW_S = 1.5

    def __init__(self, model_size: str = "small", language: str = "en", wake_word: bool = False) -> None:
        from faster_whisper import WhisperModel

        self._model_size = model_size
        threads = min(8, os.cpu_count() or 4)
        self._language = language
        print(f"[whisper] loading {model_size} language={language} (CPU int8, {threads} threads)")
        self._model = WhisperModel(
            model_size, device="cpu", compute_type="int8", cpu_threads=threads
        )
        print(f"[whisper] {model_size} ready")
        self._use_wake = wake_word
        self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="whisper")
        self._running = False
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._mic: Optional[_MicSource] = None

    async def run(self, on_transcript: TranscriptCallback) -> None:
        loop = asyncio.get_running_loop()
        mic = _MicSource(loop)
        mic.start()
        self._loop = loop
        self._mic = mic
        self._running = True
        segmenter = UtteranceSegmenter()
        utterance_id = 0
        detector = None
        if self._use_wake:
            import wake_model

            global _wake_ready
            loaded = wake_model.load()
            _wake_ready = loaded.available
            detector = loaded if loaded.available else None

        async def emit(pcm: bytes) -> None:
            nonlocal utterance_id
            uid = utterance_id
            utterance_id += 1
            text = await loop.run_in_executor(self._executor, self._transcribe, pcm)
            if text:
                on_transcript(text, True, True, uid)

        try:
            if detector is not None:
                await self._run_until_stopped(mic, segmenter, emit, detector)
            else:
                while self._running:
                    chunk = await mic.get()
                    if not self._running or not chunk:
                        break
                    pcm = segmenter.push(chunk)
                    if pcm:
                        await emit(pcm)
        finally:
            try:
                # Listening stopped mid-sentence: transcribe what was already
                # said instead of dropping the buffer on the floor.
                pcm = segmenter.flush()
                if pcm:
                    await emit(pcm)
            finally:
                mic.stop()
                self._mic = None

    async def _run_until_stopped(self, mic: _MicSource, segmenter: UtteranceSegmenter, emit, detector) -> None:
        """Score audio with the wake model. Whisper runs only after a hit."""
        from collections import deque

        preroll: deque[bytes] = deque(maxlen=max(8, int(1.5 / (BLOCK_SIZE / SAMPLE_RATE))))
        asleep = True
        while self._running:
            chunk = await mic.get()
            if not self._running or not chunk:
                break
            if asleep:
                preroll.append(chunk)
                if not detector.consume(chunk):
                    continue
                print("[wake] hey jarvis")
                asleep = False
                segmenter = UtteranceSegmenter()
                for old in preroll:
                    pcm = segmenter.push(old)
                    if pcm:
                        await emit(pcm)
                        detector.reset()
                        preroll.clear()
                        asleep = True
                        segmenter = UtteranceSegmenter()
                        break
                continue
            pcm = segmenter.push(chunk)
            if pcm:
                await emit(pcm)
                detector.reset()
                preroll.clear()
                asleep = True
                segmenter = UtteranceSegmenter()
        if not asleep:
            pcm = segmenter.flush()
            if pcm:
                await emit(pcm)

    def _transcribe(self, pcm_bytes: bytes) -> str:
        audio = np.frombuffer(pcm_bytes, dtype=np.int16).astype(np.float32) / 32768.0
        started = time.perf_counter()
        segments, _ = self._model.transcribe(
            audio,
            language=self._language,
            beam_size=1,
            temperature=0.0,
            vad_filter=True,
            vad_parameters={"min_silence_duration_ms": 500, "speech_pad_ms": 400},
            condition_on_previous_text=False,
            compression_ratio_threshold=2.4,
            log_prob_threshold=-1.0,
            # Higher means "only drop a clip when Whisper is very sure it is
            # silence". 0.6 was discarding short, quiet commands.
            no_speech_threshold=0.85,
            without_timestamps=True,
            # Do NOT put example commands in initial_prompt. Whisper copies
            # them into the output whenever the audio is music/noise - that
            # is why "Hey Jarvis, open Chrome" kept firing by itself.
            initial_prompt="Jarvis.",
        )
        text = " ".join(seg.text.strip() for seg in segments).strip()
        audio_s = len(pcm_bytes) / (SAMPLE_RATE * 2)
        print(f"[whisper] {audio_s:.1f}s audio in {time.perf_counter() - started:.2f}s -> {text!r}")
        if _is_hallucination(text):
            return ""
        return text

    def stop(self) -> None:
        self._running = False
        mic = self._mic
        if mic is not None:
            mic.wake()


_wake_ready = False


def wake_model_is_ready() -> bool:
    return _wake_ready


def make_streamer(
    deepgram_api_key: Optional[str],
    backend_override: Optional[str] = None,
    wake_word: bool = False,
):
    backend = backend_override or ("deepgram" if deepgram_api_key else "whisper")
    if backend == "deepgram":
        if not deepgram_api_key:
            raise RuntimeError("STT_BACKEND=deepgram but DEEPGRAM_API_KEY is not set")
        return DeepgramStreamer(deepgram_api_key)
    return FasterWhisperStreamer(resolve_whisper_model(), resolve_stt_language(), wake_word=wake_word)
