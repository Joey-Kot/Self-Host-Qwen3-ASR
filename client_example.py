from pathlib import Path
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="EMPTY")
audio_path = Path("test.wav")

with audio_path.open("rb") as f:
    result = client.audio.transcriptions.create(
        model="Qwen/Qwen3-ASR-1.7B",
        file=f,
        response_format="json",
        # Custom service parameters; both default to false when omitted.
        extra_body={
            "enable_lid": True,
            "enable_itn": True,
        },
    )

print(result)
