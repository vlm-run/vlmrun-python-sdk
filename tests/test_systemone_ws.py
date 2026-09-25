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
import re
import threading
from pathlib import Path

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


def answer_for(message, base=0.5):
    """A value unique to the image a decide carried.

    Lets a test assert that each answer came back to the read whose image
    produced it, without depending on how the client numbers its requests.
    """
    import zlib

    if not message:
        return base
    content = message.get("content") or []
    if not content:
        return base
    url = content[0]["image_url"]["url"]
    return round(base + (zlib.crc32(url.encode()) % 100_000) / 1_000_000, 6)


def distinct_images(count):
    """`count` distinct JPEG data URLs, so replies can be told apart."""
    import numpy

    from vlmrun.common.image import encode_frame

    return [
        encode_frame(numpy.full((8, 8, 3), 10 + i * 3, dtype=numpy.uint8))
        for i in range(count)
    ]


class StubServer:
    """A websocket server that speaks the System One session protocol.

    Records everything the client sent, and can answer out of order or with an
    error, which is where a pipelined client is easiest to get wrong.
    """

    def __init__(
        self, *, max_inflight=2, reverse=False, shuffle=None, fail_on=None, noul=0.5
    ):
        self.max_inflight = max_inflight
        self.reverse = reverse
        self.shuffle = shuffle  # a random.Random, to permute each batch
        self.reply_order: list = []
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
                held.append((message.get("id"), message))
                # Answer in batches so a reorder has something to reorder.
                batched = self.reverse or self.shuffle is not None
                if len(held) >= (self.max_inflight if batched else 1):
                    order = list(held)
                    if self.shuffle is not None:
                        self.shuffle.shuffle(order)
                    elif self.reverse:
                        order.reverse()
                    for request_id, decide in order:
                        self.reply_order.append(request_id)
                        await ws.send(json.dumps(self._decision(request_id, decide)))
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

    def _decision(self, request_id, message=None):
        return {
            "type": "decision",
            "frame": 0,
            "id": request_id,
            # Derived from the image this decide actually carried, so a reply
            # delivered to the wrong read is visible. Keying off the request id
            # instead would only test that ids are handed out in order, which is
            # the client's private business.
            "answers": {"a": {"type": "noul", "noul": answer_for(message, self.noul)}},
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
        assert 0.0 <= decision.response.nouls["a"].noul <= 1.0
        assert decision.response.usage.input_tokens == 10

    def test_out_of_order_replies_reach_the_right_read(self):
        """Correlation is by id, not arrival — the reason ids exist."""
        pytest.importorskip("numpy")
        images = distinct_images(6)
        expected = [
            answer_for({"content": [{"image_url": {"url": image}}]}) for image in images
        ]
        with StubServer(reverse=True, max_inflight=2) as server:
            with stream_to(server, concurrency=2) as stream:
                decisions = list(stream.map(images))
        assert [d.index for d in decisions] == [0, 1, 2, 3, 4, 5]
        # Each answer is derived from the image that produced it, so a reply
        # handed to the wrong read shows up here.
        assert [d.response.nouls["a"].noul for d in decisions] == expected

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
        assert 0.0 <= response.nouls["a"].noul <= 1.0

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
            def _decision(self, request_id, message=None):
                decision = super()._decision(request_id, message)
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


class TestOrderingIsIntact:
    """Responses must come back paired to the input that produced them.

    One socket carries every read, so arrival order is whatever the server
    chose. Each reply's answer is derived from the image that produced it, so a
    mispairing is visible rather than merely possible — keying off the request
    id would only assert that ids are handed out in order, which is the client's
    private business and not the property under test.
    """

    @pytest.mark.parametrize("seed", range(5))
    def test_a_shuffled_server_still_pairs_correctly(self, seed):
        import random

        pytest.importorskip("numpy")
        images = distinct_images(16)
        expected = [
            answer_for({"content": [{"image_url": {"url": image}}]}) for image in images
        ]
        assert len(set(expected)) == len(expected), "the oracle must be injective"
        with StubServer(max_inflight=4, shuffle=random.Random(seed)) as server:
            with stream_to(server, concurrency=4) as stream:
                decisions = list(stream.map(images))
        assert [d.index for d in decisions] == list(range(16))
        assert [d.response.nouls["a"].noul for d in decisions] == expected

    def test_frame_timestamps_stay_with_their_answers(self):
        """The timeline's x-axis and y-axis must not come apart."""
        import random

        from vlmrun.common.video import SampledFrame

        numpy = pytest.importorskip("numpy")
        frames = [
            SampledFrame(
                index=i,
                timestamp_s=i * 0.5,
                frame=numpy.full((4, 4, 3), i, dtype=numpy.uint8),
            )
            for i in range(12)
        ]
        from vlmrun.common.image import encode_frame

        expected = [
            answer_for({"content": [{"image_url": {"url": encode_frame(f.frame)}}]})
            for f in frames
        ]
        assert len(set(expected)) == len(expected), "the oracle must be injective"
        with StubServer(max_inflight=4, shuffle=random.Random(7)) as server:
            with stream_to(server, concurrency=4) as stream:
                decisions = list(stream.map(frames))
        assert [d.index for d in decisions] == list(range(12))
        assert [d.timestamp_s for d in decisions] == [i * 0.5 for i in range(12)]
        # The answer, its frame index and its timestamp all have to belong together.
        assert [d.response.nouls["a"].noul for d in decisions] == expected


class TestAnswerTypesOverTheSocket:
    """Every answer type has to decode, not just the one the other tests use.

    A score's legend and probabilities are keyed by level, which JSON carries as
    strings. The HTTP path is decoded by the TypeSafe SDK, which converts them;
    a socket frame validated directly does not, and the model wants ints.
    """

    @staticmethod
    def _server_answering(answers):
        class Fixed(StubServer):
            def _decision(self, request_id, message=None):
                return {
                    "type": "decision",
                    "frame": 0,
                    "id": request_id,
                    "answers": answers,
                    "usage": {"input_tokens": 10, "output_tokens": 0},
                    "latency_ms": 5.0,
                }

        return Fixed()

    def test_a_score_answer_decodes(self):
        answers = {
            "mood": {
                "type": "score",
                "score": 1.4,
                # String keys, as JSON delivers them.
                "legend": {"0": "calm", "1": "cross", "2": "furious"},
                "probabilities": {"0": 0.1, "1": 0.5, "2": 0.4},
                "confidence": 0.33,
            }
        }
        with self._server_answering(answers) as server:
            with stream_to(server) as stream:
                decision = next(iter(stream.map([PNG_DATA_URL])))
        score = decision.response.scores["mood"]
        assert score.score == pytest.approx(1.4)
        assert score.legend[1] == "cross", "legend must be keyed by level"
        assert score.probabilities[2] == pytest.approx(0.4)

    def test_a_choice_answer_decodes(self):
        answers = {
            "dept": {
                "type": "choice",
                "choice": "billing",
                "probabilities": {"billing": 0.9, "sales": 0.1},
                "confidence": 0.7,
            }
        }
        with self._server_answering(answers) as server:
            with stream_to(server) as stream:
                decision = next(iter(stream.map([PNG_DATA_URL])))
        choice = decision.response.choices["dept"]
        assert choice.choice == "billing"
        assert choice.probabilities["billing"] == pytest.approx(0.9)

    def test_all_three_types_in_one_read(self):
        answers = {
            "urgent": {"type": "noul", "noul": 0.9},
            "dept": {
                "type": "choice",
                "choice": "billing",
                "probabilities": {"billing": 1.0},
                "confidence": 1.0,
            },
            "mood": {
                "type": "score",
                "score": 0.5,
                "legend": {"0": "calm", "1": "cross"},
                "probabilities": {"0": 0.5, "1": 0.5},
                "confidence": 0.0,
            },
        }
        with self._server_answering(answers) as server:
            with stream_to(server) as stream:
                decision = next(iter(stream.map([PNG_DATA_URL])))
        assert decision.response.nouls["urgent"].noul == pytest.approx(0.9)
        assert decision.response.choices["dept"].choice == "billing"
        assert decision.response.scores["mood"].legend[0] == "calm"


class TestReviewFindings:
    """Regressions for the faults found in review of this transport."""

    def test_read_options_ride_the_handshake(self):
        """`decide` rejects these outright, so losing them is silent and costly.

        `samples` especially: the server would bill its own default number of
        draws while the caller believed it had asked for more.
        """
        with StubServer() as server:
            with stream_to(server, steps=3, samples=4) as stream:
                list(stream.map([PNG_DATA_URL]))
        create = server.creates[0]
        assert create["steps"] == 3
        assert create["samples"] == 4
        # And not on the decide, which the route refuses.
        assert all("samples" not in d and "steps" not in d for d in server.decides)

    def test_reasoning_effort_rides_the_handshake(self):
        with StubServer() as server:
            with stream_to(server, reasoning_effort="low") as stream:
                list(stream.map([PNG_DATA_URL]))
        assert server.creates[0]["reasoning_effort"] == "low"

    def test_options_left_unset_are_not_sent(self):
        with StubServer() as server:
            with stream_to(server):
                pass
        create = server.creates[0]
        assert "steps" not in create and "samples" not in create
        assert "reasoning_effort" not in create

    def test_extra_body_is_refused_rather_than_dropped(self):
        """The route rejects unknown fields, so there is nowhere to put these."""
        with StubServer() as server:
            with pytest.raises(InputError) as caught:
                stream_to(server, extra_body={"anything": 1})
        assert "extra_body" in caught.value.message

    def test_concurrent_state_changes_do_not_deadlock(self):
        """Two barriers each holding part of the window would wait forever."""
        import threading

        with StubServer(max_inflight=4) as server:
            with stream_to(server, state="start", concurrency=4) as stream:
                errors: list = []

                def change(value):
                    try:
                        stream.send(PNG_DATA_URL, state=value)
                    except Exception as e:  # noqa: BLE001 - reported below
                        errors.append(e)

                threads = [
                    threading.Thread(target=change, args=(f"state-{i}",))
                    for i in range(4)
                ]
                for thread in threads:
                    thread.start()
                for thread in threads:
                    thread.join(timeout=15)
                assert not any(t.is_alive() for t in threads), "a state change hung"
                assert errors == []
        # Every transition was applied, one at a time.
        assert len(server.states) == 5  # the initial state, then four changes

    def test_a_read_times_out_instead_of_hanging(self):
        """A server that holds the socket open must not block a read forever."""
        from vlmrun.client.exceptions import RequestTimeoutError

        class Silent(StubServer):
            async def _handler(self, ws):
                async for raw in ws:
                    message = json.loads(raw)
                    self.received.append(message)
                    if message.get("type") == "session.create":
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "session.created",
                                    "session_id": "stub",
                                    "model": MODEL,
                                    "questions": ["a"],
                                    "max_inflight": 2,
                                }
                            )
                        )
                    # Never answers a decide.

        with Silent() as server:
            with stream_to(server, timeout=0.5) as stream:
                with pytest.raises(RequestTimeoutError):
                    stream.send(PNG_DATA_URL)

    def test_a_timed_out_read_leaves_nothing_pending(self):
        """A late reply must not be handed to a read that already failed."""
        from vlmrun.client.exceptions import RequestTimeoutError

        class Slow(StubServer):
            async def _handler(self, ws):
                async for raw in ws:
                    message = json.loads(raw)
                    self.received.append(message)
                    if message.get("type") == "session.create":
                        await ws.send(
                            json.dumps(
                                {
                                    "type": "session.created",
                                    "session_id": "stub",
                                    "model": MODEL,
                                    "questions": ["a"],
                                    "max_inflight": 2,
                                }
                            )
                        )
                    elif message.get("type") == "decide":
                        await asyncio.sleep(0.4)
                        await ws.send(
                            json.dumps(self._decision(message.get("id"), message))
                        )

        with Slow() as server:
            with stream_to(server, timeout=0.1) as stream:
                with pytest.raises(RequestTimeoutError):
                    stream.send(PNG_DATA_URL)
                assert stream._pending == {}
                # The session still works once given enough time.
                stream._options["timeout"] = 5.0
                assert stream.send(PNG_DATA_URL) is not None


class TestDeclaredWebsocketsRange:
    """The declared range has to match the API this transport actually calls.

    The floor is 14.0 rather than 13.x because ``websockets.connect`` only became
    the asyncio client there. On 13.x the same name resolves to the legacy
    client, which takes ``extra_headers`` — so a 13.x install would raise
    TypeError on the first session, and declaring it supported would be a lie.
    """

    @staticmethod
    def _declared_specs():
        """Every `websockets` requirement in pyproject, in file order.

        Read as text rather than parsed: `tomllib` is stdlib only from 3.11 and
        this project supports 3.10, and pulling in a TOML backport to assert on
        two lines would be a poor trade.
        """
        root = Path(__file__).resolve().parents[1] / "pyproject.toml"
        return re.findall(
            r'^\s*"(websockets[^"]*)"', root.read_text(), flags=re.MULTILINE
        )

    def test_the_ws_extra_has_both_bounds(self):
        from packaging.requirements import Requirement

        specs = self._declared_specs()
        assert specs, "no websockets requirement found in pyproject.toml"
        operators = {s.operator for s in Requirement(specs[0]).specifier}
        assert ">=" in operators, "a floor is needed: 13.x cannot run this code"
        assert any(op in operators for op in ("<", "<=")), "an upper bound is needed"

    def test_every_extra_declares_the_same_range(self):
        """The `ws` extra and `all` both carry it; they must not drift apart."""
        specs = self._declared_specs()
        assert len(specs) == 2, specs
        assert len(set(specs)) == 1, specs

    def test_the_installed_version_satisfies_what_is_declared(self):
        from packaging.requirements import Requirement

        spec = Requirement(self._declared_specs()[0]).specifier
        assert websockets.version.version in spec

    def test_the_api_the_session_uses_exists(self):
        """If a release drops one of these, the upper bound should have caught it."""
        import inspect

        parameters = set(inspect.signature(websockets.connect).parameters)
        assert {"additional_headers", "max_size", "open_timeout"} <= parameters
        assert websockets.connect.__module__ == "websockets.asyncio.client"
