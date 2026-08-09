from __future__ import annotations

import io
import unittest
from types import SimpleNamespace

import httpx
from openai import OpenAI

import server


def _multipart(*fields: tuple[str, str]):
    return [("file", ("sample.wav", b"fake-audio", "audio/wav"))] + [
        (name, (None, value)) for name, value in fields
    ]


class ServerFeatureTests(unittest.IsolatedAsyncioTestCase):
    def test_cjk_punctuation_is_preserved_in_subtitle_segments(self) -> None:
        timestamps = [
            SimpleNamespace(text="你", start_time=0.0, end_time=0.2),
            SimpleNamespace(text="好", start_time=0.2, end_time=0.4),
            SimpleNamespace(text="世", start_time=0.5, end_time=0.7),
            SimpleNamespace(text="界", start_time=0.7, end_time=0.9),
            SimpleNamespace(text="再", start_time=1.2, end_time=1.4),
            SimpleNamespace(text="见", start_time=1.4, end_time=1.7),
        ]

        segments = server._build_timed_segments("你好，世界。再见！", timestamps)

        self.assertEqual([segment["text"] for segment in segments], ["你好，世界。", "再见！"])

    async def asyncSetUp(self) -> None:
        self.original_transcribe = server._transcribe_sync
        self.fake_result = SimpleNamespace(
            text="Hello, world. Next sentence!",
            language="English",
            time_stamps=[
                SimpleNamespace(text="Hello", start_time=0.0, end_time=0.4),
                SimpleNamespace(text="world", start_time=0.45, end_time=0.9),
                SimpleNamespace(text="Next", start_time=1.2, end_time=1.6),
                SimpleNamespace(text="sentence", start_time=1.65, end_time=2.1),
            ],
        )
        server._transcribe_sync = lambda *args: self.fake_result
        self.client = httpx.AsyncClient(
            transport=httpx.ASGITransport(app=server.app),
            base_url="http://testserver",
        )

    async def asyncTearDown(self) -> None:
        server._transcribe_sync = self.original_transcribe
        await self.client.aclose()

    async def test_verbose_json_accepts_openai_bracket_timestamp_fields(self) -> None:
        response = await self.client.post(
            "/v1/audio/transcriptions",
            files=_multipart(
                ("response_format", "verbose_json"),
                ("timestamp_granularities[]", "segment"),
                ("timestamp_granularities[]", "word"),
            ),
        )

        self.assertEqual(response.status_code, 200)
        payload = response.json()
        self.assertEqual(payload["language"], "en")
        self.assertEqual(payload["duration"], 2.1)
        self.assertEqual([segment["text"] for segment in payload["segments"]], ["Hello, world.", "Next sentence!"])
        self.assertEqual([word["word"] for word in payload["words"]], ["Hello,", "world.", "Next", "sentence!"])

    async def test_srt_is_rendered_from_aligned_segments(self) -> None:
        response = await self.client.post(
            "/v1/audio/transcriptions",
            files=_multipart(("response_format", "srt")),
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("application/x-subrip"))
        self.assertEqual(
            response.text,
            "1\n00:00:00,000 --> 00:00:00,900\nHello, world.\n\n"
            "2\n00:00:01,200 --> 00:00:02,100\nNext sentence!\n",
        )

    async def test_stream_returns_openai_transcription_sse_events(self) -> None:
        response = await self.client.post(
            "/v1/audio/transcriptions",
            files=_multipart(("stream", "true")),
        )

        self.assertEqual(response.status_code, 200)
        self.assertTrue(response.headers["content-type"].startswith("text/event-stream"))
        self.assertEqual(response.headers["x-asr-stream-mode"], "pseudo")
        self.assertIn('"type":"transcript.text.delta"', response.text)
        self.assertIn('"type":"transcript.text.done"', response.text)
        self.assertTrue(response.text.endswith("data: [DONE]\n\n"))

    def test_sse_frames_are_parsed_by_the_openai_python_sdk(self) -> None:
        body = "".join(
            [
                server._sse_frame(
                    "transcript.text.delta",
                    {"type": "transcript.text.delta", "delta": "Hello "},
                ),
                server._sse_frame(
                    "transcript.text.delta",
                    {"type": "transcript.text.delta", "delta": "world"},
                ),
                server._sse_frame(
                    "transcript.text.done",
                    {"type": "transcript.text.done", "text": "Hello world"},
                ),
                "data: [DONE]\n\n",
            ]
        ).encode("utf-8")

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(
                200,
                headers={"content-type": "text/event-stream"},
                content=body,
                request=request,
            )

        http_client = httpx.Client(transport=httpx.MockTransport(handler))
        client = OpenAI(api_key="EMPTY", base_url="http://test/v1", http_client=http_client)
        try:
            events = list(
                client.audio.transcriptions.create(
                    model="Qwen/Qwen3-ASR-1.7B",
                    file=("sample.wav", io.BytesIO(b"fake-audio")),
                    stream=True,
                )
            )
        finally:
            http_client.close()

        self.assertEqual(
            [(event.type, getattr(event, "delta", None), getattr(event, "text", None)) for event in events],
            [
                ("transcript.text.delta", "Hello ", None),
                ("transcript.text.delta", "world", None),
                ("transcript.text.done", None, "Hello world"),
            ],
        )


if __name__ == "__main__":
    unittest.main()
