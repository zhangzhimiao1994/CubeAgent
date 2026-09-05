from __future__ import annotations

import asyncio
import contextlib
import uuid

from robot_runtime.audio import (
    AudioCapture,
    AudioPlayback,
    NullAudioCapture,
    NullAudioPlayback,
    PassthroughEchoCanceller,
)
from robot_runtime.bridge import Envelope, LoggingBridgeClient, WebsocketBridgeClient
from robot_runtime.config import RuntimeConfig
from robot_runtime.router import EdgeCloudRouter
from robot_runtime.session import InteractionLoop, LocalSession, LoopState
from robot_runtime.vad import EnergyTurnTaking, SimpleBargeIn


class RobotRuntime:
    """Pi frontend wiring: audio + bridge only. AI stays on the agent backend."""

    def __init__(
        self,
        config: RuntimeConfig | None = None,
        *,
        capture: AudioCapture | None = None,
        playback: AudioPlayback | None = None,
    ) -> None:
        self.config = config or RuntimeConfig()
        self.capture = capture or NullAudioCapture()
        self.playback = playback or NullAudioPlayback()
        self.aec = PassthroughEchoCanceller()
        self.turn_taking = EnergyTurnTaking()
        self.barge_in = SimpleBargeIn()
        self.bridge = (
            WebsocketBridgeClient(self.config)
            if self.config.device_token
            else LoggingBridgeClient(self.config)
        )
        self.router = EdgeCloudRouter()
        self.session = LocalSession()
        self.loop = InteractionLoop()
        self.bridge.on_message(self._on_cloud)

    async def run_forever(self) -> None:
        delay = 1.0
        while True:
            try:
                await self.start()
                print(
                    f"robot_runtime device={self.config.device_id} "
                    f"state={self.loop.state.value} cloud={self.config.cloud_ws_url}"
                )
                wait_closed = getattr(self.bridge, "wait_closed", None)
                if callable(wait_closed):
                    await wait_closed()
                delay = 1.0
            except asyncio.CancelledError:
                raise
            except Exception as error:
                print(f"robot_runtime reconnecting after error: {error}")
            finally:
                with contextlib.suppress(Exception):
                    await self.stop()
            await asyncio.sleep(delay)
            delay = min(delay * 2.0, 30.0)

    async def start(self) -> None:
        await self.capture.start()
        await self.playback.start()
        await self.bridge.connect()
        await self.bridge.send(
            Envelope(
                type="hello",
                payload={
                    "device_id": self.config.device_id,
                    "role": "frontend",
                    "capabilities": ["vad", "barge_in", "pcm16"],
                },
            )
        )

    async def stop(self) -> None:
        await self.bridge.close()
        await self.playback.stop()
        await self.capture.stop()

    async def on_frame(self, pcm: bytes, *, energy: float | None = None) -> None:
        cleaned = self.aec.process(pcm)
        decision = self.turn_taking.observe(cleaned, energy=energy)
        if decision.speech_active and self.loop.state is LoopState.IDLE:
            self.loop.on_speech_start()
            self.session.turn_id = str(uuid.uuid4())
            await self.bridge.send(
                Envelope(type="utterance.start", turn_id=self.session.turn_id)
            )
        if self.loop.state is LoopState.CAPTURING and self.session.turn_id:
            await self.bridge.send(
                Envelope(
                    type="utterance.audio",
                    turn_id=self.session.turn_id,
                    payload={"bytes": len(cleaned)},
                )
            )
        if (
            self.config.enable_barge_in
            and self.barge_in.should_interrupt(
                speech_active=decision.speech_active,
                assistant_playing=self.playback.is_playing,
            )
        ):
            await self.playback.clear()
            self.loop.on_barge_in()
            await self.bridge.send(
                Envelope(type="barge_in", turn_id=self.session.turn_id)
            )
        if decision.turn_ended and self.session.turn_id:
            self.loop.on_turn_end()
            transcript = self.session.pending_transcript.strip()
            await self.bridge.send(
                Envelope(
                    type="final_transcript" if transcript else "utterance.end",
                    turn_id=self.session.turn_id,
                    payload={
                        "route": self.router.choose().value,
                        "text": transcript,
                        "transcript": transcript,
                    },
                )
            )
            self.session.pending_transcript = ""

    async def _on_cloud(self, envelope: Envelope) -> None:
        if envelope.type in {"text_delta", "assistant.text", "final"}:
            self.loop.on_assistant_start()
        if envelope.type in {"final", "cancelled", "error", "assistant.end"}:
            self.loop.on_assistant_end()
            await self.playback.clear()
