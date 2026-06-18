"""Voice dictation support for Wattle's live TUI."""

from __future__ import annotations

import contextlib
import os
import shutil
import subprocess
import tempfile
import wave
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from wattle.auth import get_api_key_credential

VOICE_DICTATION_API_KEY_ENV = "VOICE_DICTATION_API_KEY"
VOICE_DICTATION_MODEL_ENV = "VOICE_DICTATION_MODEL"
DEFAULT_VOICE_DICTATION_MODEL = "gpt-4o-mini-transcribe"
VOICE_SAMPLE_RATE = 16_000
VOICE_CHANNELS = 1
VOICE_SAMPLE_WIDTH_BYTES = 2


class VoiceDictationError(RuntimeError):
    """Raised when voice recording or transcription cannot complete."""


@dataclass(frozen=True)
class VoiceDictationConfig:
    api_key: str
    model: str = DEFAULT_VOICE_DICTATION_MODEL


def resolve_voice_dictation_config(
    env: Mapping[str, str] | None = None,
) -> VoiceDictationConfig:
    """Return OpenAI configuration for dictation.

    ``VOICE_DICTATION_API_KEY`` is the primary interface. For local developer
    convenience, Wattle also accepts an ``openai.api_key`` credential from the
    normal auth store when the voice-specific variable is not set. OAuth tokens
    are intentionally not accepted for OpenAI's platform transcription endpoint.
    """

    source = env if env is not None else os.environ
    api_key = source.get(VOICE_DICTATION_API_KEY_ENV, "").strip()
    if not api_key:
        try:
            api_key = get_api_key_credential("openai").bearer_token.strip()
        except Exception as exc:  # noqa: BLE001
            raise VoiceDictationError(
                f"Set {VOICE_DICTATION_API_KEY_ENV} to an OpenAI API key "
                "or add openai.api_key to ~/.wattle/auth.json to use /voice."
            ) from exc
    if not api_key:
        raise VoiceDictationError(
            f"Set {VOICE_DICTATION_API_KEY_ENV} to a non-empty OpenAI API key."
        )
    model = source.get(VOICE_DICTATION_MODEL_ENV, DEFAULT_VOICE_DICTATION_MODEL).strip()
    return VoiceDictationConfig(
        api_key=api_key,
        model=model or DEFAULT_VOICE_DICTATION_MODEL,
    )


class MicrophoneRecorder:
    """Capture microphone audio into a temporary WAV file.

    Linux hosts use ``arecord`` when available. Otherwise Wattle falls back to
    ``sounddevice.RawInputStream`` lazily so importing Wattle does not require
    audio system access.
    """

    def __init__(
        self,
        *,
        samplerate: int = VOICE_SAMPLE_RATE,
        channels: int = VOICE_CHANNELS,
    ) -> None:
        self.samplerate = samplerate
        self.channels = channels
        self._stream = None
        self._chunks: list[bytes] = []
        self._process: subprocess.Popen[bytes] | None = None
        self._path: Path | None = None
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        if shutil.which("arecord"):
            self._start_arecord()
            return
        self._start_sounddevice()

    def _start_arecord(self) -> None:
        fd, path_text = tempfile.mkstemp(prefix="wattle-voice-", suffix=".wav")
        os.close(fd)
        path = Path(path_text)
        command = [
            "arecord",
            "-q",
            "-f",
            "S16_LE",
            "-r",
            str(self.samplerate),
            "-c",
            str(self.channels),
            str(path),
        ]
        try:
            self._process = subprocess.Popen(  # noqa: S603
                command,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
        except Exception as exc:  # noqa: BLE001
            path.unlink(missing_ok=True)
            message = f"Could not start arecord microphone recording: {exc}"
            raise VoiceDictationError(message) from exc
        self._path = path
        self._started = True

    def _start_sounddevice(self) -> None:
        try:
            import sounddevice as sd  # type: ignore[import-not-found]
        except Exception as exc:  # noqa: BLE001
            raise VoiceDictationError(
                "Microphone recording requires arecord or the optional sounddevice "
                "package with a working PortAudio input device."
            ) from exc

        def callback(indata: object, _frames: int, _time: object, status: object) -> None:
            if status:
                # Keep recording despite transient over/underflow flags.
                pass
            self._chunks.append(bytes(indata))

        try:
            self._stream = sd.RawInputStream(
                samplerate=self.samplerate,
                channels=self.channels,
                dtype="int16",
                callback=callback,
            )
            self._stream.start()
        except Exception as exc:  # noqa: BLE001
            self._stream = None
            raise VoiceDictationError(f"Could not start microphone recording: {exc}") from exc
        self._started = True

    def stop_to_wav(self) -> Path:
        if not self._started:
            raise VoiceDictationError("Microphone recording was not started.")
        if self._process is not None:
            return self._stop_arecord()
        return self._stop_sounddevice()

    def _stop_arecord(self) -> Path:
        process = self._process
        path = self._path
        self._process = None
        self._path = None
        self._started = False
        if process is None or path is None:
            raise VoiceDictationError("arecord recording was not started.")
        try:
            process.terminate()
            try:
                _stdout, stderr = process.communicate(timeout=2.0)
            except subprocess.TimeoutExpired:
                process.kill()
                _stdout, stderr = process.communicate(timeout=2.0)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        if process.returncode not in (0, -15):
            path.unlink(missing_ok=True)
            detail = stderr.decode(errors="replace").strip() if stderr else "unknown error"
            raise VoiceDictationError(f"arecord microphone recording failed: {detail}")
        if not path.exists() or path.stat().st_size <= 44:
            path.unlink(missing_ok=True)
            raise VoiceDictationError("No microphone audio was captured.")
        return path

    def _stop_sounddevice(self) -> Path:
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
        self._started = False
        audio = b"".join(self._chunks)
        if not audio:
            raise VoiceDictationError("No microphone audio was captured.")
        fd, path_text = tempfile.mkstemp(prefix="wattle-voice-", suffix=".wav")
        path = Path(path_text)
        try:
            with os.fdopen(fd, "wb") as fileobj, wave.open(fileobj, "wb") as wav:
                wav.setnchannels(self.channels)
                wav.setsampwidth(VOICE_SAMPLE_WIDTH_BYTES)
                wav.setframerate(self.samplerate)
                wav.writeframes(audio)
        except Exception:
            path.unlink(missing_ok=True)
            raise
        return path

    def discard(self) -> None:
        if self._process is not None:
            process = self._process
            self._process = None
            path = self._path
            self._path = None
            process.terminate()
            with contextlib.suppress(Exception):
                process.wait(timeout=1.0)
            if path is not None:
                path.unlink(missing_ok=True)
        if self._stream is not None:
            try:
                self._stream.stop()
                self._stream.close()
            finally:
                self._stream = None
        self._started = False
        self._chunks.clear()


def transcribe_audio_file(
    path: Path,
    *,
    config: VoiceDictationConfig | None = None,
) -> str:
    """Transcribe a WAV file with OpenAI and return plain text."""

    resolved = config or resolve_voice_dictation_config()
    try:
        from openai import OpenAI
    except Exception as exc:  # noqa: BLE001
        raise VoiceDictationError("The openai package is required for /voice dictation.") from exc

    try:
        client = OpenAI(api_key=resolved.api_key)
        with path.open("rb") as audio_file:
            transcription = client.audio.transcriptions.create(
                model=resolved.model,
                file=audio_file,
            )
    except Exception as exc:  # noqa: BLE001
        raise VoiceDictationError(f"OpenAI voice transcription failed: {exc}") from exc

    text = getattr(transcription, "text", None)
    if text is None and isinstance(transcription, dict):
        text = transcription.get("text")
    if not isinstance(text, str):
        raise VoiceDictationError("OpenAI transcription response did not contain text.")
    return text.strip()
