"""Tests for ``vlmrun gw`` System One over the websocket transport.

Driven against a stub server that speaks the real protocol — ``session.create``
/ ``session.created``, ``state``, ``decide`` / ``decision``, ``session.close`` /
``session.stats`` — so the assertions cover this client's own work: the
handshake, id correlation, the in-flight ceiling and teardown.
"""

from __future__ import annotations

import asyncio
import base64
import json
import threading

import pytest

from vlmrun.client.exceptions import APIError, InputError
from vlmrun.client.systemone import SystemOne, WebSocketStream

pytest.importorskip("websockets")
import websockets  # noqa: E402

PNG_BYTES = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAYAAAAfFcSJAAAADUlEQVR42mP8z8BQDwAEhQGAhKmMIQAAAABJRU5ErkJggg=="
)
PNG_DATA_URL = f"data:image/png;base64,{base64.b64encode(PNG_BYTES).decode()}"
MODEL = "google/diffusiongemma-26b-a4b-it"


class StubServer:
    """A websocket server that speaks the System One session protocol.

    Records everything the client sent, and can answer out of order or with an
    error, which is where a pipelined client is easiest to get wrong.
    """

    def __init__(self, *, max_inflight=2, reverse=False, fail_on=None, noul=0.5):
        self.max_inflight = max_inflight
        self.reverse = reverse
        self.fail_on = fail_on
        self.noul = noul
        self.received: list = []
        self.granted = max_inflight
        self.peak_inflight = 0
        self._live = 0
        self._loop = None
        self._server = None
        self._thread = None
        self.port = None

    # -- protocol ------------------------------------------------------
    async def _handler(self, ws):
        held: list = []
        async for raw in ws:
            message = json.loads(raw)
            self.received.append(message)
            kind = message.get("type")
            if kind == "session.create":
                # Like the real route: grant what was asked, capped by our own.
                asked = message.get("max_inflight") or self.max_inflight
                self.granted = min(int(asked), self.max_inflight)
                await ws.send(
                    json.dumps(
                        {
                            "type": "session.created",
                            "session_id": "stub",
                            "model": message.get("model") or MODEL,
                            "questions": list(message.get("questions") or {}),
                            "max_inflight": self.granted,
                            "max_frame_bytes": 2097152,
                            "expires_in": 600.0,
                        }
                    )
                )
            elif kind == "decide":
                self._live += 1
                self.peak_inflight = max(self.peak_inflight, self._live)
                if self.fail_on is not None and len(held) == self.fail_on:
                    await ws.send(
                        json.dumps(
                            {
                                "type": "error",
                                "error_type": "invalid_request_error",
                                "message": "stub refuses this read",
                            }
                        )
                    )
                    self._live -= 1
                    continue
                held.append(message.get("id"))
                # Answer in batches so `reverse` actually reorders replies.
                if len(held) >= (self.max_inflight if self.reverse else 1):
                    for request_id in reversed(held) if self.reverse else held:
                        await ws.send(json.dumps(self._decision(request_id)))
                        self._live -= 1
                    held.clear()
            elif kind == "session.close":
                await ws.send(
                    json.dumps(
                        {
                            "type": "session.stats",
                            "session_id": "stub",
                            "decisions": sum(
                                1 for m in self.received if m.get("type") == "decide"
                            ),
                            "reads": 1,
                            "cost": 0.0001,
                        }
                    )
                )
                return

    def _decision(self, request_id):
        return {
            "type": "decision",
            "frame": 0,
            "id": request_id,
            # The id is folded into the answer so a test can prove each reply
            # reached the read that asked for it.
            "answers": {
                "a": {
                    "type": "noul",
                    "noul": round(self.noul + int(request_id) / 1000, 4),
                }
            },
            "usage": {"input_tokens": 10, "output_tokens": 0},
            "latency_ms": 5.0,
        }

    # -- lifetime ------------------------------------------------------
    def __enter__(self):
        ready = threading.Event()

        async def start():
            self._server = await websockets.serve(self._handler, "127.0.0.1", 0)
            self.port = self._server.sockets[0].getsockname()[1]

        def run():
            self._loop = asyncio.new_event_loop()
            asyncio.set_event_loop(self._loop)
            # Start, then hand the loop over to run_forever, so stopping it is
            # clean rather than interrupting a pending run_until_complete.
            self._loop.run_until_complete(start())
            ready.set()
            self._loop.run_forever()

        self._thread = threading.Thread(target=run, daemon=True)
        self._thread.start()
        assert ready.wait(10), "stub server did not start"
        return self

    def __exit__(self, *exc):
        if self._loop is not None:
            self._loop.call_soon_threadsafe(self._loop.stop)
        if self._thread is not None:
            self._thread.join(timeout=5)

    @property
    def decides(self):
        return [m for m in self.received if m.get("type") == "decide"]

    @property
    def states(self):
        return [m for m in self.received if m.get("type") == "state"]

    @property
    def creates(self):
        return [m for m in self.received if m.get("type") == "session.create"]


class FakeClient:
    def __init__(self, api_key="test-key"):
        self.api_key = api_key
        self.base_url = "https://api.vlm.run/v1"
        self.timeout = 120.0
        self.max_retries = 5


def stream_to(server, **kwargs):
    """A WebSocketStream pointed at the stub."""
    resource = SystemOne(
        FakeClient(), base_url=f"http://127.0.0.1:{server.port}", model=MODEL
    )
    return resource.stream([{"id": "a", "type": "noul"}], transport="ws", **kwargs)


class TestWebSocketHandshake:
    def test_url_is_derived_from_the_typesafe_root(self):
        resource = SystemOne(FakeClient(), base_url="https://gw.test/typesafe")
        assert resource.stream([{"id": "a", "type": "noul"}], transport="ws").url == (
            "wss://gw.test/typesafe/ws"
        )

    def test_plain_http_root_gives_a_ws_url(self):
        resource = SystemOne(FakeClient(), base_url="http://localhost:9/typesafe")
        assert (
            resource.stream([{"id": "a", "type": "noul"}], transport="ws").url
            == "ws://localhost:9/typesafe/ws"
        )

    def test_questions_are_sent_once_not_per_read(self):
        """The whole point of a session: the spec rides the handshake."""
        with StubServer() as server:
            with stream_to(server) as stream:
                list(stream.map([PNG_DATA_URL] * 4))
        assert len(server.creates) == 1
        assert server.creates[0]["questions"] == {"a": {"type": "noul"}}
        assert len(server.decides) == 4
        assert all("questions" not in d for d in server.decides)

    def test_the_ack_limits_are_exposed(self):
        with StubServer(max_inflight=3) as server:
            with stream_to(server) as stream:
                assert stream.limits["max_inflight"] == 3
                assert stream.limits["session_id"] == "stub"

    def test_state_is_sent_once_at_open(self):
        with StubServer() as server:
            with stream_to(server, state="the state") as stream:
                list(stream.map([PNG_DATA_URL] * 3))
        assert [m["state"] for m in server.states] == ["the state"]

    def test_no_state_message_when_there_is_no_state(self):
        with StubServer() as server:
            with stream_to(server) as stream:
                list(stream.map([PNG_DATA_URL]))
        assert server.states == []

    def test_a_refused_session_raises(self):
        class Refusing(StubServer):
            async def _handler(self, ws):
                await ws.recv()
                await ws.send(json.dumps({"type": "error", "message": "no"}))

        with Refusing() as server:
            with pytest.raises(InputError):
                with stream_to(server):
                    pass


class TestWebSocketReads:
    def test_images_ride_the_decide_as_content(self):
        with StubServer() as server:
            with stream_to(server) as stream:
                list(stream.map([PNG_DATA_URL]))
        content = server.decides[0]["content"]
        assert content[0]["type"] == "image_url"
        assert content[0]["image_url"]["url"] == PNG_DATA_URL

    def test_a_decision_decodes_into_the_shared_response_model(self):
        """Callers must not be able to tell which transport answered."""
        with StubServer() as server:
            with stream_to(server) as stream:
                decision = next(iter(stream.map([PNG_DATA_URL])))
        assert decision.response.model == MODEL
        assert decision.response.nouls["a"].noul == pytest.approx(0.5, abs=0.01)
        assert decision.response.usage.input_tokens == 10

    def test_out_of_order_replies_reach_the_right_read(self):
        """Correlation is by id, not arrival — the reason ids exist."""
        with StubServer(reverse=True, max_inflight=2) as server:
            with stream_to(server, concurrency=2) as stream:
                decisions = list(stream.map([PNG_DATA_URL] * 6))
        # The stub folds each reply's id into its answer, so a mis-paired reply
        # would show up as an answer out of sequence.
        assert [d.index for d in decisions] == [0, 1, 2, 3, 4, 5]
        assert [round(d.response.nouls["a"].noul, 4) for d in decisions] == [
            0.5,
            0.501,
            0.502,
            0.503,
            0.504,
            0.505,
        ]

    def test_frame_identity_survives_the_socket(self):
        from vlmrun.common.video import SampledFrame

        numpy = pytest.importorskip("numpy")
        frames = [
            SampledFrame(
                index=i * 10,
                timestamp_s=i / 2,
                frame=numpy.full((4, 4, 3), 10 * i, dtype=numpy.uint8),
            )
            for i in range(3)
        ]
        with StubServer() as server:
            with stream_to(server) as stream:
                decisions = list(stream.map(frames))
        assert [d.index for d in decisions] == [0, 10, 20]
        assert [d.timestamp_s for d in decisions] == [0.0, 0.5, 1.0]

    def test_send_reads_one_input(self):
        with StubServer() as server:
            with stream_to(server) as stream:
                response = stream.send(PNG_DATA_URL)
                assert stream.count == 1
        assert response.nouls["a"].noul == pytest.approx(0.5, abs=0.01)

    def test_a_per_read_state_change_is_a_barrier(self):
        with StubServer() as server:
            with stream_to(server, state="first") as stream:
                stream.send(PNG_DATA_URL)
                stream.send(PNG_DATA_URL, state="second")
                stream.send(PNG_DATA_URL)
        # The update is sent once and then sticks: it belongs to the session.
        assert [m["state"] for m in server.states] == ["first", "second"]

    def test_the_servers_inflight_ceiling_is_respected(self):
        """concurrency is an ask; max_inflight from the ack is the limit."""
        with StubServer(max_inflight=2) as server:
            with stream_to(server, concurrency=16) as stream:
                list(stream.map([PNG_DATA_URL] * 8))
        assert server.peak_inflight <= 2

    def test_a_server_error_fails_the_read(self):
        with StubServer(fail_on=0) as server:
            with pytest.raises(APIError):
                with stream_to(server) as stream:
                    list(stream.map([PNG_DATA_URL] * 2))


class TestWebSocketLifetime:
    def test_open_only_inside_its_block(self):
        with StubServer() as server:
            stream = stream_to(server)
            assert isinstance(stream, WebSocketStream)
            assert not stream.is_open
            with stream:
                assert stream.is_open
            assert not stream.is_open
            with pytest.raises(InputError):
                list(stream.map([PNG_DATA_URL]))

    def test_closing_collects_the_session_stats(self):
        with StubServer() as server:
            with stream_to(server) as stream:
                list(stream.map([PNG_DATA_URL] * 3))
                assert stream.stats == {}, "stats only exist once the session closes"
            assert stream.stats["decisions"] == 3
        assert any(m.get("type") == "session.close" for m in server.received)

    def test_leaving_on_an_error_still_closes(self):
        with StubServer() as server:
            with pytest.raises(RuntimeError):
                with stream_to(server) as stream:
                    stream.send(PNG_DATA_URL)
                    raise RuntimeError("boom")
            assert not stream.is_open

    def test_stopping_early_does_not_read_the_rest(self):
        with StubServer() as server:
            with stream_to(server, concurrency=2) as stream:
                for decision in stream.map([PNG_DATA_URL] * 40):
                    if decision.index == 1:
                        break
        assert len(server.decides) < 12, f"issued {len(server.decides)} of 40"

    def test_no_thread_is_left_running(self):
        before = threading.active_count()
        with StubServer() as server:
            with stream_to(server) as stream:
                list(stream.map([PNG_DATA_URL]))
        assert threading.active_count() <= before + 1


class TestTransportSelection:
    def test_http_is_the_default(self):
        from vlmrun.client.systemone import DecisionStream

        stream = SystemOne(FakeClient()).stream([{"id": "a", "type": "noul"}])
        assert type(stream) is DecisionStream

    def test_ws_selects_the_socket_transport(self):
        stream = SystemOne(FakeClient()).stream(
            [{"id": "a", "type": "noul"}], transport="ws"
        )
        assert isinstance(stream, WebSocketStream)

    def test_an_unknown_transport_is_rejected(self):
        with pytest.raises(InputError):
            SystemOne(FakeClient()).stream(
                [{"id": "a", "type": "noul"}], transport="grpc"
            )


class TestWebSocketUsage:
    """The gateway's cost and cached-token split must survive the socket.

    ``typesafe_sdk.Usage`` models only input/output tokens, and the HTTP path
    recovers the rest from the raw body — which a websocket read does not have.
    """

    def test_cost_and_cached_tokens_are_reported(self):
        from vlmrun.client.systemone import usage_of

        class Detailed(StubServer):
            def _decision(self, request_id):
                decision = super()._decision(request_id)
                decision["usage"] = {
                    "input_tokens": 88,
                    "output_tokens": 0,
                    "input_tokens_details": {"cached_tokens": 64, "image_tokens": 20},
                    "reads": 1,
                    "cost": 0.000012,
                }
                return decision

        with Detailed() as server:
            with stream_to(server) as stream:
                decision = next(iter(stream.map([PNG_DATA_URL])))
        usage = usage_of(decision.response)
        assert usage["input_tokens"] == 88
        assert usage["cost"] == pytest.approx(0.000012)
        assert usage["input_tokens_details"]["cached_tokens"] == 64


class TestInflightNegotiation:
    """The route defaults to fewer reads in flight than this SDK does.

    A session that does not ask is throttled to the server's default, so the
    request is part of the handshake rather than something left implicit.
    """

    def test_the_handshake_asks_for_the_requested_concurrency(self):
        with StubServer(max_inflight=8) as server:
            with stream_to(server, concurrency=4) as stream:
                assert stream.limits["max_inflight"] == 4
        assert server.creates[0]["max_inflight"] == 4

    def test_the_default_asks_for_four(self):
        from vlmrun.client.systemone import DEFAULT_STREAM_CONCURRENCY

        with StubServer(max_inflight=8) as server:
            with stream_to(server):
                pass
        assert server.creates[0]["max_inflight"] == DEFAULT_STREAM_CONCURRENCY == 4

    def test_the_request_is_clamped_to_the_protocol_ceiling(self):
        """Asking past the route's limit is a validation error, not a clamp."""
        from vlmrun.client.systemone import WEBSOCKET_MAX_INFLIGHT

        with StubServer(max_inflight=8) as server:
            with stream_to(server, concurrency=64):
                pass
        assert server.creates[0]["max_inflight"] == WEBSOCKET_MAX_INFLIGHT == 8

    def test_a_server_granting_less_than_asked_still_wins(self):
        with StubServer(max_inflight=2) as server:
            with stream_to(server, concurrency=8) as stream:
                assert stream.limits["max_inflight"] == 2
                list(stream.map([PNG_DATA_URL] * 6))
        assert server.creates[0]["max_inflight"] == 8
        assert server.peak_inflight <= 2
