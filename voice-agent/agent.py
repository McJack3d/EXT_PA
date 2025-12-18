import io
import os
import queue
import sys
import time
import threading

import numpy as np
import sounddevice as sd
import soundfile as sf
from dotenv import load_dotenv
from openai import OpenAI

def _record_wav_push_to_talk(sample_rate: int = 16000, channels: int = 1) -> bytes:
    print("\nPress ENTER to start recording.")
    input()
    print("Recording... press ENTER to stop.")

    audio_q: queue.Queue[np.ndarray] = queue.Queue()
    stop_event = threading.Event()

    def wait_for_enter_to_stop():
        try:
            input()
        finally:
            stop_event.set()

    def callback(indata, frames, time_info, status):
        if status:
            # Print to stderr but keep going
            print(status, file=sys.stderr)
        audio_q.put(indata.copy())

    stream = sd.InputStream(samplerate=sample_rate, channels=channels, dtype="float32", callback=callback)

    chunks = []
    with stream:
        threading.Thread(target=wait_for_enter_to_stop, daemon=True).start()
        while not stop_event.is_set():
            try:
                chunk = audio_q.get(timeout=0.1)
                chunks.append(chunk)
            except queue.Empty:
                continue

    if not chunks:
        return b""

    audio = np.concatenate(chunks, axis=0)
    audio = np.clip(audio, -1.0, 1.0)

    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _play_wav_bytes(wav_bytes: bytes):
    if not wav_bytes:
        return
    bio = io.BytesIO(wav_bytes)
    data, sr = sf.read(bio, dtype="float32")
    sd.play(data, sr)
    sd.wait()


def main():
    load_dotenv()

    if not os.getenv("OPENAI_API_KEY"):
        raise SystemExit("Missing env var: OPENAI_API_KEY (set it in your shell or .env)")

    chat_model = os.getenv("OPENAI_CHAT_MODEL", "gpt-5.2")
    transcribe_model = os.getenv("OPENAI_TRANSCRIBE_MODEL", "gpt-4o-mini-transcribe")
    tts_model = os.getenv("OPENAI_TTS_MODEL", "gpt-4o-mini-tts")
    tts_voice = os.getenv("OPENAI_TTS_VOICE", "alloy")
    sample_rate = int(os.getenv("AUDIO_SAMPLE_RATE", "16000"))

    client = OpenAI()

    system_prompt = os.getenv(
        "SYSTEM_PROMPT",
        "You are a helpful voice assistant. Keep replies concise and conversational.",
    )

    messages = [
        {"role": "system", "content": system_prompt},
    ]

    print("Voice Agent ready.")
    print("- ENTER: record a message")
    print("- Ctrl+C: quit")

    while True:
        try:
            wav_bytes = _record_wav_push_to_talk(sample_rate=sample_rate, channels=1)
            if not wav_bytes:
                print("No audio captured; try again.")
                continue

            # Transcribe
            transcript = client.audio.transcriptions.create(
                model=transcribe_model,
                file=("audio.wav", wav_bytes, "audio/wav"),
            )
            user_text = (getattr(transcript, "text", None) or "").strip()
            if not user_text:
                print("Transcription empty; try again.")
                continue

            print(f"\nYou: {user_text}")
            messages.append({"role": "user", "content": user_text})

            # Chat
            resp = client.responses.create(
                model=chat_model,
                input=messages,
            )
            assistant_text = (getattr(resp, "output_text", None) or "").strip()
            if not assistant_text:
                assistant_text = "Sorry, I didn't get that."

            print(f"\nAssistant: {assistant_text}\n")
            messages.append({"role": "assistant", "content": assistant_text})

            # TTS (wav) + play
            try:
                speech = client.audio.speech.create(
                    model=tts_model,
                    voice=tts_voice,
                    input=assistant_text,
                    format="wav",
                )
            except TypeError:
                speech = client.audio.speech.create(
                    model=tts_model,
                    voice=tts_voice,
                    input=assistant_text,
                    response_format="wav",
                )

            # SDK may return bytes directly or an object with `.read()`
            wav_out = None
            if isinstance(speech, (bytes, bytearray)):
                wav_out = bytes(speech)
            elif hasattr(speech, "read"):
                wav_out = speech.read()
            elif hasattr(speech, "content"):
                wav_out = speech.content

            if wav_out:
                _play_wav_bytes(wav_out)

        except KeyboardInterrupt:
            print("\nBye.")
            return
        except Exception as e:
            print(f"\nError: {e}")
            time.sleep(0.5)


if __name__ == "__main__":
    main()
