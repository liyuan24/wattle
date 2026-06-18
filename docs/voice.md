# Voice dictation

Wattle can use your microphone to dictate text into the live TUI input box. Voice dictation is useful when you want to describe a larger change out loud, then review and edit the transcribed instruction before pressing Enter.

Voice dictation currently supports OpenAI speech-to-text only.

## Requirements

Before using `/voice`, make sure you have:

1. A terminal running Wattle's live TUI.
2. A working microphone input device on the machine running Wattle.
3. An OpenAI platform API key available to Wattle.

Set the voice dictation key in your shell:

```bash
export WATTLE_VOICE_DICTATION_API_KEY="sk-..."
```

Optionally choose a transcription model:

```bash
export VOICE_DICTATION_MODEL="gpt-4o-mini-transcribe"
```

`WATTLE_VOICE_DICTATION_API_KEY` is required for voice dictation. OAuth credentials and normal chat-provider credentials in `~/.wattle/auth.json` are not used for speech-to-text.

## Start Wattle

Open Wattle in a project directory:

```bash
wattle
```

Voice dictation is only available in the live TUI input box. It is not used by headless `wattle -p` runs.

## Enable voice dictation

In the TUI, run:

```text
/voice
```

Wattle will verify that an OpenAI API key is available. If the key is missing, it shows an error explaining how to set `WATTLE_VOICE_DICTATION_API_KEY`.

You can also be explicit:

```text
/voice on
```

To turn it off:

```text
/voice off
```

Running `/voice` with no argument toggles the current state.

## Dictate into the input box

After voice dictation is enabled:

1. Put the cursor in the Wattle input box.
2. Hold the Space key.
3. Speak your instruction while continuing to hold Space.
4. Release Space.
5. Wait for Wattle to transcribe your speech into the input box.
6. Review or edit the transcribed text.
7. Press Enter to submit it like any typed instruction.

A quick tap of Space still inserts a normal space. Wattle only starts recording when Space is held long enough to be treated as a hold.

## What appears in the prompt

When voice mode is enabled, the prompt shows a voice status line such as:

```text
Voice · hold Space to dictate
```

While recording:

```text
Voice · recording; release Space to transcribe
```

While waiting for OpenAI transcription:

```text
Voice · transcribing...
```

The resulting text is inserted at the current cursor position. If needed, Wattle adds surrounding spaces so the dictated text does not run into adjacent words.

## Troubleshooting

### `/voice` says the API key is missing

Set the voice dictation key before starting Wattle:

```bash
export WATTLE_VOICE_DICTATION_API_KEY="sk-..."
wattle
```

Do not use an OpenAI OAuth token or a provider credential from `~/.wattle/auth.json` for voice dictation. The OpenAI speech-to-text endpoint requires a platform API key in `WATTLE_VOICE_DICTATION_API_KEY`.

### Recording fails

Voice dictation records on the machine where Wattle is running. If you are connected over SSH, Wattle needs access to a microphone device on the remote machine, not your laptop unless you have forwarded audio devices.

On Linux, Wattle prefers `arecord` when available. If `arecord` cannot access a microphone, configure your system audio input or install/configure a compatible audio stack. Wattle can also fall back to the Python `sounddevice` package when PortAudio and an input device are available.

### Transcription is wrong

The transcription text is only inserted into the input box. You can edit it before pressing Enter. If you consistently get poor results, try speaking closer to the microphone or setting a different OpenAI transcription model with `VOICE_DICTATION_MODEL`.
