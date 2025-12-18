# voice-agent

Minimal push-to-talk voice chat with OpenAI:
- record mic (ENTER to start/stop)
- transcribe
- send to chat model
- synthesize speech
- play audio back

## Setup (macOS)

System deps (recommended):

```zsh
brew install portaudio libsndfile
```

Python deps:

```zsh
cd /Users/alexandrebredillot/Documents/GitHub/voice-agent
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Create `.env`:

```zsh
cp .env.example .env
# then edit .env and set OPENAI_API_KEY
```

## Run

```zsh
cd /Users/alexandrebredillot/Documents/GitHub/voice-agent
source .venv/bin/activate
python run.py
```

Controls:
- Press `ENTER` to start recording
- Press `ENTER` again to stop recording
- `Ctrl+C` to quit

## Notes
- Do not paste API keys into chat; rotate keys if leaked.
- If you get mic permission errors, enable Terminal/VS Code microphone access in macOS Privacy settings.
