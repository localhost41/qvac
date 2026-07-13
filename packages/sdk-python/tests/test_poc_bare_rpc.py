"""QVAC-21806 PoC: same real-worker rigor as test_poc_smoke.py/test_poc_progress.py,
but the wire encode/decode is bare_rpc.RPC instead of poc_heartbeat.QvacWorker's
hand-rolled framing. Validates responses against the same generated pydantic
models either way -- proving bare_rpc is a drop-in replacement at the framing
layer without changing anything above it.
"""

from __future__ import annotations

import os

import pytest
import pytest_asyncio

from poc_bare_rpc_transport import BARE_RPC_AVAILABLE, WORKER
from qvac.models import QWEN3_600M_INST_Q4, TTS_EN_SUPERTONIC_Q4_0
from qvac.schemas import (
    CompletionStreamRequest,
    HeartbeatRequest,
    LoadModelRequest,
    ModelType,
    TextToSpeechStreamRequest,
)
from qvac.methods import completion_stream, heartbeat, load_model, text_to_speech_stream

pytestmark = [
    pytest.mark.asyncio,
    pytest.mark.skipif(
        not BARE_RPC_AVAILABLE,
        reason="bare_rpc not installed -- install the 'bare-rpc' extra "
        "(`pip install -e '.[bare-rpc]'`) to run these PoC tests",
    ),
    pytest.mark.skipif(
        not os.path.exists(WORKER),
        reason=f"no built SDK worker found at {WORKER!r} -- run `bun run build` in packages/sdk, or set QVAC_POC_SDK_DIR",
    ),
]


@pytest_asyncio.fixture
async def worker():
    from poc_bare_rpc_transport import BareRpcWorker

    async with BareRpcWorker() as w:
        yield w


async def test_heartbeat_unary_via_bare_rpc(worker) -> None:
    """BareRpcWorker.call/.call_stream already match qvac._transport.Transport's
    async shape exactly, so it plugs straight into the real generated stubs --
    no PocTransport-style adapter needed, unlike poc_heartbeat.QvacWorker."""
    response = await heartbeat(worker, HeartbeatRequest(type="heartbeat"))
    assert response.type == "heartbeat"
    assert isinstance(response.number, float)


async def test_load_model_and_completion_stream_via_bare_rpc(worker) -> None:
    load_request = LoadModelRequest.model_validate(
        {
            "type": "loadModel",
            "modelSrc": QWEN3_600M_INST_Q4.src,
            "modelType": "llamacpp-completion",
            "modelConfig": {},
        }
    )
    load_response = await load_model(worker, load_request)
    assert load_response.success, load_response.error
    model_id = load_response.model_id

    completion_request = CompletionStreamRequest.model_validate(
        {
            "type": "completionStream",
            "modelId": model_id,
            "history": [{"role": "user", "content": "Say hello in five words."}],
            "stream": True,
        }
    )

    text = ""
    async for chunk in completion_stream(worker, completion_request):
        for event in chunk.events:
            if event.type == "contentDelta":
                text += event.text
    assert text.strip(), "expected real completion text via the bare_rpc server-stream"


async def _as_async_iter(items):
    for item in items:
        yield item


async def test_load_model_and_tts_stream_duplex_via_bare_rpc(worker) -> None:
    """Exercises call_duplex end to end: text goes up the request stream while
    synthesized audio comes down the response stream, concurrently."""
    load_request = LoadModelRequest.model_validate(
        {
            "type": "loadModel",
            "modelSrc": TTS_EN_SUPERTONIC_Q4_0.src,
            "modelType": ModelType.tts_ggml,
            "modelConfig": {"ttsEngine": "supertonic", "language": "en"},
        }
    )
    load_response = await load_model(worker, load_request)
    assert load_response.success, load_response.error
    model_id = load_response.model_id

    tts_request = TextToSpeechStreamRequest.model_validate(
        {"type": "textToSpeechStream", "modelId": model_id}
    )
    text = b"Hello from QVAC. This is streaming text to speech."

    samples = []
    saw_done = False
    async for chunk in text_to_speech_stream(
        worker, tts_request, _as_async_iter([text])
    ):
        samples.extend(chunk.buffer)
        saw_done = saw_done or chunk.done
    assert samples, "expected real synthesized audio via the bare_rpc duplex stream"
    assert saw_done, "expected a terminal done=True event on the response stream"
