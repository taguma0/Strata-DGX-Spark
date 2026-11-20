"""The observed footprint behind vram_need "auto": a start measures what it really took off the card - the
free VRAM as the vision encoder and engine find it vs. after the engine runs - keeps it beside the engine's log,
and the next start's auto line prices that.  The formula alone is a floor, not a footprint, when the expert
cache fills eagerly."""
import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import (ByteTokenizer, GpuBusy, MockEngine, Service, StrataEngine,
                          observed_vram_read, observed_vram_write)


class QuietEngine(StrataEngine):
    """A fake engine that starts on demand and never speaks, like the lifecycle tests' ResidentEngine."""

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

    def generate(self, *args, **kwargs):
        yield from self.reply.generate(*args, **kwargs)


class Sidecar(unittest.TestCase):
    def test_round_trip_and_corruption(self):
        with TemporaryDirectory() as d:
            p = str(Path(d) / "strata-vram-observed.json")
            self.assertEqual(observed_vram_read(p, "k"), 0)               # never measured
            observed_vram_write(p, "k1", 55264)
            observed_vram_write(p, "k2", 48000)
            self.assertEqual(observed_vram_read(p, "k1"), 55264)
            self.assertEqual(observed_vram_read(p, "k2"), 48000)          # one file, one entry per engine-args key
            observed_vram_write(p, "k1", 56000)                            # the last start wins, self-correcting
            self.assertEqual(observed_vram_read(p, "k1"), 56000)
            Path(p).write_text("not json", encoding="utf-8")
            self.assertEqual(observed_vram_read(p, "k1"), 0)               # a corrupt file: the floor decides alone


class MeasuredAuto(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.svc = Service(QuietEngine(self.tok), self.tok,
                           ChatTemplate(Path(__file__).parent / "chat_template.jinja"))
        self.svc._release_orphan_encoder = mock.Mock(return_value=False)
        self.svc.vram_need = "auto"
        self.svc.vram_need_auto_mib = 8000        # vram_fit's floor, as the wiring prices it
        self.dir = TemporaryDirectory()
        self.svc.observed_vram_path = str(Path(self.dir.name) / "strata-vram-observed.json")
        self.svc.observed_vram_key = "k"

    def tearDown(self):
        self.dir.cleanup()

    def test_the_start_prices_what_the_card_gave_and_auto_follows(self):
        # gate's read, the start's own read, the card after the engine runs: 55 GB really went away
        self.svc.free_vram_mib = mock.Mock(side_effect=[60_000, 60_000, 5_000])
        self.svc.ensure_loaded()
        self.assertEqual(json.loads(Path(self.svc.observed_vram_path).read_text(encoding="utf-8")), {"k": 55_000})
        self.assertEqual(self.svc.vram_need_auto_mib, 55_000)
        self.assertEqual(self.svc.vram_need_mib(), (55_000, "vram_need auto"))   # the next load prices the footprint

    def test_a_start_that_only_regains_memory_is_not_priced(self):
        self.svc.free_vram_mib = mock.Mock(side_effect=[60_000, 60_000, 59_800])   # 200 MiB: a settle's noise
        self.svc.ensure_loaded()
        self.assertFalse(Path(self.svc.observed_vram_path).exists())
        self.assertEqual(self.svc.vram_need_auto_mib, 8000)                        # the floor stands
        self.svc.free_vram_mib = mock.Mock(side_effect=[60_000, 60_000, 61_000])   # the card gave memory back
        self.svc.engine.loaded = False
        self.svc.ensure_loaded()
        self.assertEqual(self.svc.vram_need_auto_mib, 8000)

    def test_a_previous_run_s_measurement_raises_the_line_before_any_start(self):
        observed_vram_write(self.svc.observed_vram_path, "k", 52_000)
        # what the startup wiring does: the observed footprint is priced into the auto line before any gate runs
        self.svc.vram_need_auto_mib = max(self.svc.vram_need_auto_mib,
                                          observed_vram_read(self.svc.observed_vram_path, self.svc.observed_vram_key))
        self.svc.free_vram_mib = mock.Mock(return_value=40_000)                    # less than the observed footprint
        self.svc.load_wait_s = 0.3
        with self.assertRaises(GpuBusy):                                            # evict/wait first, as any line would
            self.svc.ensure_loaded()
        self.assertEqual(self.svc.engine.starts, 0)


if __name__ == "__main__":
    unittest.main()
