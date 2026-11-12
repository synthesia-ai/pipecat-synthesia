"""The shipped examples actually work: they import, and their pipeline runs against a fake worker.

Marker: ``integration_extended``. This needs a ``livekit-server``, and it is about the examples rather
than the product, so it belongs in the nightly tier and not on the PR gate.

**What this covers that the join-mode suites do not.** ``test_relayed_join.py`` and
``test_direct_join.py`` already prove the product's wiring exhaustively. What they never touch is the
*example files*: a sibling import that only resolves when run as a script, a vendor import that needs an
extra nobody installed, a brain helper whose fake path raises. Those break the first thing a new
developer runs, and they break silently: the examples are not imported by anything else in the suite.

So there are two halves here:

1. **Import every example.** Cheap, and it catches the whole class of "the quickstart is broken".
2. **Run the examples' own brain through a real pipeline against ``FakeAvatarWorker``.** Hermetic: the
   scripted brain needs no vendor keys and no network, which is the reason ``build_brain("fake")``
   exists.
"""

from __future__ import annotations

import asyncio
import importlib
import sys
from pathlib import Path

import pytest
from pipecat.frames.frames import (
    OutputTransportReadyFrame,
    TranscriptionFrame,
    TTSAudioRawFrame,
)
from pipecat.pipeline.pipeline import Pipeline
from pipecat.processors.frame_processor import FrameDirection
from pipecat.utils.time import time_now_iso8601

from tests.harness import bidirectional_pipeline, make_service
from tests.integration._join_modes import AVATAR_IDS, WAIT, private_room
from tests.integration._launcher import Launcher
from tests.support.events import EventRecorder
from tests.support.livekit_dev import LiveKitDev
from tests.support.waiting import until

pytestmark = pytest.mark.integration_extended

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"

#: Every example a developer is told to run, plus the helpers they import.
EXAMPLE_MODULES = (
    "_brain",
    "_fake_brain",
    "_presence",
    "_shutdown",
    "bot_direct_join",
    "bot_relayed_join",
    "mint_tokens",
)


@pytest.fixture(scope="module", autouse=True)
def _examples_on_path():
    """`examples/` on sys.path, exactly as running one as a script would put it there.

    The examples import each other as siblings (`from _brain import ...`) precisely *because*
    they run as scripts. Importing them as `examples.foo` here would test an arrangement that
    never happens in practice and would hide a broken sibling import.
    """
    sys.path.insert(0, str(EXAMPLES))
    try:
        yield
    finally:
        if str(EXAMPLES) in sys.path:
            sys.path.remove(str(EXAMPLES))
        for name in EXAMPLE_MODULES:
            sys.modules.pop(name, None)


@pytest.mark.parametrize("name", EXAMPLE_MODULES)
def test_example_imports(name: str) -> None:
    """Import the module. No livekit-server needed; this is the cheap half.

    Importing an example runs its module scope: the imports, the transport-params dict, the
    argument parser. That is where the breakages live -- a missing extra, a sibling import that
    assumed a package, a lazy import that was not lazy after all.
    """
    module = importlib.import_module(name)
    assert module is not None


def test_every_example_file_is_covered() -> None:
    """A sabotage check: a new example added to `examples/` must be added to EXAMPLE_MODULES.

    Without this, the parametrised test above silently keeps passing while the new file -- the
    one most likely to be broken -- is never imported.
    """
    on_disk = {
        path.stem
        for path in EXAMPLES.glob("*.py")
        if not path.name.startswith("__") and "__pycache__" not in path.parts
    }
    assert on_disk == set(EXAMPLE_MODULES), (
        f"examples/ contains {sorted(on_disk)} but EXAMPLE_MODULES lists "
        f"{sorted(EXAMPLE_MODULES)}; add the new example so it is imported by this suite"
    )


async def test_fake_brain_pipeline_runs_against_the_fake_worker(
    livekit_server: LiveKitDev, room_name: str, launcher: Launcher
) -> None:
    """The examples' own `--fake-brain` path, end to end, with no vendor keys and no network.

    This is the configuration `test_examples_smoke` exists to protect: a developer with no
    Deepgram/Cartesia/OpenAI account runs an example, and it has to work. The brain comes from
    the examples' helper rather than being rebuilt here -- rebuilding it would test this file.
    """
    from _brain import build_brain

    brain = build_brain("fake")
    assert brain.pre_avatar, "the fake brain produced no processors"

    api = await launcher.api()
    service = make_service(
        private_room(livekit_server, room_name=room_name),
        avatar_ids=AVATAR_IDS,
        api_url=api.base_url,
    )
    recorder = EventRecorder(service)

    # The brain has to be *in* the pipeline, not merely constructed: `bidirectional_pipeline`
    # wraps the one processor it is given, so passing the bare service would leave the
    # TranscriptionFrame below with nothing to act on it and no audio would ever be produced.
    # A Pipeline is itself a FrameProcessor, so the example's own chain nests cleanly.
    async with bidirectional_pipeline(Pipeline([*brain.pre_avatar, service])) as h:
        await h.queue(OutputTransportReadyFrame(), FrameDirection.UPSTREAM)
        (info,) = await recorder.wait("on_avatar_connected", timeout=WAIT)
        assert info.session_id, launcher.debug_dump()

        # One utterance, through the example's own brain: a TranscriptionFrame is what the fake
        # STT would emit, the fake LLM answers it, and the fake TTS speaks the answer.
        await h.queue(TranscriptionFrame("hello avatar", "", time_now_iso8601()))

        # The audio the brain produced must reach the worker, asserted against the *counterparty's*
        # recorder rather than anything on our side. Presence, never timing.
        await until(
            lambda: bool(launcher.fakes) and any(s["bytes"] for s in launcher.fakes[0].streams),
            timeout=WAIT,
            what="the fake worker to receive audio produced by the fake brain",
        )

        await h.finish()

    assert recorder.count("on_session_ended") == 1, recorder.names()
    assert recorder.count("on_metrics") == 1, recorder.names()


def test_fake_tts_produces_audible_audio() -> None:
    """The fake TTS must emit real samples, not silence.

    A stand-in that emitted zeroes would satisfy every "did audio flow" assertion in this file
    while producing an avatar that never moves its mouth.
    """
    from _fake_brain import FakeTTS

    tts = FakeTTS()
    assert tts._fallback_sample_rate > 0

    async def collect() -> list[TTSAudioRawFrame]:
        return [f async for f in tts.run_tts("hello there", "c1") if isinstance(f, TTSAudioRawFrame)]

    frames = asyncio.run(collect())
    total = b"".join(f.audio for f in frames)
    assert total, "the fake TTS produced no audio at all"
    assert total.strip(b"\x00"), "the fake TTS produced digital silence, which moves no lips"
