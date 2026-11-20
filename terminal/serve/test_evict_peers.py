"""Pressure eviction (evict_on_pressure): when the load gate finds too little VRAM free for this model, the other
strata servers are asked to give their GPU back - in idle or priority order, skipping busy, protected and just-
evicted-us peers - and the asked side remembers who asked (asked_by) for the grace window."""
import os
import time
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import (ByteTokenizer, GpuBusy, MockEngine, Service, StrataEngine,
                          discover_serve_servers, parse_evict_peers)


class QuietEngine(StrataEngine):
    """A fake engine that starts and stops on demand and never speaks, like the lifecycle tests' ResidentEngine."""

    def __init__(self, tok):
        self.reply = MockEngine(tok, "Hello.", max_context=4096)
        self.max_context, self.info, self.last = 4096, {}, {}
        self.loaded, self.starts, self.unloaded = False, 0, True

    def alive(self):
        return self.loaded

    def exit_code(self):
        return None

    def restart(self):
        self.loaded, self.unloaded = True, False
        self.starts += 1

    def unload(self):
        self.loaded, self.unloaded = False, True

    def generate(self, *args, **kwargs):
        yield from self.reply.generate(*args, **kwargs)


PEER = "http://127.0.0.1:9001"
OTHER = "http://127.0.0.1:9002"


class Evict(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.svc = Service(QuietEngine(self.tok), self.tok,
                           ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.svc._release_orphan_encoder = mock.Mock(return_value=False)
        self.svc.min_free_vram_mib = 10_000
        self.svc.load_wait_s = 0.3
        self.svc.evict_on_pressure = True
        self.svc.evict_peers = [{"url": PEER}]
        self.svc.evict_self_url = "http://127.0.0.1:8001"

    def fake_http(self, statuses, unloads):
        """A _evict_http answering from canned dicts; POSTs are recorded as (url, asked_by)."""
        asked = []

        def http(peer, path, post=None):
            if path == "/v1/status":
                return statuses.get(peer["url"])
            asked.append((peer["url"], (post or {}).get("asked_by")))
            return unloads.get(peer["url"], {"status": "unloaded"})
        return http, asked

    def loaded_peer(self, idle_s):
        return {"loaded": True, "activity": {"last_request_at": time.time() - idle_s}}

    def test_off_by_default_refuses_without_asking(self):
        self.svc.evict_on_pressure = False
        self.svc.free_vram_mib = mock.Mock(return_value=500)
        self.svc._evict_http = mock.Mock()
        with self.assertRaises(GpuBusy):
            self.svc.ensure_loaded()
        self.svc._evict_http.assert_not_called()
        self.assertEqual(self.svc.engine.starts, 0)

    def test_enough_free_vram_never_asks(self):
        self.svc.free_vram_mib = mock.Mock(return_value=20_000)
        self.svc._evict_http = mock.Mock()
        self.svc.ensure_loaded()
        self.svc._evict_http.assert_not_called()
        self.assertEqual(self.svc.engine.starts, 1)

    def test_asks_the_peer_and_loads(self):
        self.svc.free_vram_mib = mock.Mock(return_value=500)
        http, asked = self.fake_http({PEER: self.loaded_peer(3600)}, {})
        self.svc._evict_http = http
        self.svc._evict_settle = mock.Mock(return_value=20_000)
        self.svc.ensure_loaded()
        self.assertEqual(self.svc.engine.starts, 1)
        self.assertEqual(asked, [(PEER, "http://127.0.0.1:8001")])   # the ask carries our URL (asked_by)

    def test_busy_peer_keeps_its_model_and_the_load_still_refuses(self):
        self.svc.free_vram_mib = mock.Mock(return_value=500)
        http, asked = self.fake_http({PEER: self.loaded_peer(10)}, {PEER: {"status": "busy"}})
        self.svc._evict_http = http
        with self.assertRaises(GpuBusy):
            self.svc.ensure_loaded()
        self.assertEqual(len(asked), 1)                              # the ask was tried, the peer refused
        self.assertEqual(self.svc.engine.starts, 0)

    def test_peer_whose_model_is_already_unloaded_is_not_asked(self):
        self.svc.free_vram_mib = mock.Mock(return_value=500)
        http, asked = self.fake_http({PEER: {"loaded": False}}, {})
        self.svc._evict_http = http
        with self.assertRaises(GpuBusy):
            self.svc.ensure_loaded()
        self.assertEqual(asked, [])
        self.assertEqual(self.svc.engine.starts, 0)

    def test_unreachable_peer_is_not_asked(self):
        self.svc.free_vram_mib = mock.Mock(return_value=500)
        self.svc._evict_http = mock.Mock(return_value=None)
        with self.assertRaises(GpuBusy):
            self.svc.ensure_loaded()
        self.assertEqual(self.svc._evict_http.call_count, 1)         # one /v1/status try, no POST
        self.assertEqual(self.svc.engine.starts, 0)

    def test_grace_skips_the_peer_that_just_evicted_us(self):
        self.svc.unloaded_by[PEER] = time.time()                     # it took our model a moment ago
        self.svc.free_vram_mib = mock.Mock(return_value=500)
        self.svc._evict_http = mock.Mock()
        with self.assertRaises(GpuBusy):
            self.svc.ensure_loaded()
        self.svc._evict_http.assert_not_called()                     # the ping-pong stops here, on our side
        self.svc.unloaded_by[PEER] = time.time() - 61                # outside the 60 s grace
        http, asked = self.fake_http({PEER: self.loaded_peer(10)}, {})
        self.svc._evict_http = http
        self.svc._evict_settle = mock.Mock(return_value=20_000)
        self.svc.ensure_loaded()
        self.assertEqual(len(asked), 1)

    def test_protected_peer_is_skipped(self):
        self.svc.evict_peers = [{"url": PEER, "protect": True}]
        self.svc.free_vram_mib = mock.Mock(return_value=500)
        self.svc._evict_http = mock.Mock()
        with self.assertRaises(GpuBusy):
            self.svc.ensure_loaded()
        self.svc._evict_http.assert_not_called()

    def test_idle_order_asks_the_longest_idle_first_and_stops_when_enough(self):
        self.svc.min_free_vram_mib = 25_000
        self.svc.evict_peers = [{"url": PEER}, {"url": OTHER}]
        self.svc.free_vram_mib = mock.Mock(return_value=500)
        st = {PEER: self.loaded_peer(100), OTHER: self.loaded_peer(3000)}
        http, asked = self.fake_http(st, {})
        self.svc._evict_http = http
        self.svc._evict_settle = mock.Mock(side_effect=[15_000, 26_000])   # the first peer is not enough
        self.svc.ensure_loaded()
        self.assertEqual([u for u, _ in asked], [OTHER, PEER])             # idle 3000 s before idle 100 s
        self.assertEqual(self.svc.engine.starts, 1)

    def test_priority_order_asks_the_lowest_number_first(self):
        self.svc.min_free_vram_mib = 25_000
        self.svc.evict_order = "priority"
        self.svc.evict_peers = [{"url": PEER, "priority": 5}, {"url": OTHER, "priority": 1}]
        self.svc.free_vram_mib = mock.Mock(return_value=500)
        http, asked = self.fake_http({}, {})
        self.svc._evict_http = http
        self.svc._evict_settle = mock.Mock(side_effect=[15_000, 26_000])
        self.svc.ensure_loaded()
        self.assertEqual([u for u, _ in asked], [OTHER, PEER])
        self.assertEqual(self.svc.engine.starts, 1)

    def test_asked_side_remembers_the_asker(self):
        self.svc.engine.restart()                                   # loaded, so the unload really happens
        self.assertEqual(self.svc.unload(asked_by=PEER + "/"), "unloaded")
        self.assertIn(PEER, self.svc.unloaded_by)                   # the trailing / is normalized away
        self.assertEqual(self.svc.engine.starts, 1)                 # and the next request loads it again


class Parsing(unittest.TestCase):
    def test_strings_and_objects_and_bad_entries(self):
        out = parse_evict_peers(["http://a:1/",
                                 {"url": " http://b:2/ ", "api_key": "k", "priority": 3, "protect": True},
                                 {"no_url": 1}, 5])
        self.assertEqual(out, [{"url": "http://a:1"},
                               {"url": "http://b:2", "api_key": "k", "priority": 3, "protect": True}])
        self.assertEqual(parse_evict_peers("http://c:3"), [{"url": "http://c:3"}])
        self.assertEqual(parse_evict_peers(None), [])

    def test_discover_reads_the_process_table(self):
        with TemporaryDirectory() as d:
            def add(pid, *parts):
                Path(d, str(pid)).mkdir()
                Path(d, str(pid), "cmdline").write_bytes(b"\0".join(p.encode() for p in parts) + b"\0")
            add(111, "/app/.venv/bin/python", "serve/server.py", "--engine", "strata", "--port", "8001")
            add(222, "python3", "/srv/strata/terminal/serve/server.py", "--port=8002")
            add(333, "python3", "server.py", "--port", "9000")          # not under serve/: not strata serve
            add(444, "python3", "serve/server.py")                      # no --port: nothing to ask
            add(555, "nginx", "-g", "daemon off;")
            self.assertEqual(discover_serve_servers(d, me=111), ["http://127.0.0.1:8002"])
        self.assertEqual(discover_serve_servers(os.devnull), [])        # no process table: an empty list


if __name__ == "__main__":
    unittest.main()
