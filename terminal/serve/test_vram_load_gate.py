"""The load-time VRAM gate: an unloaded model waits for the card to come back (vram_need / load_wait_s)
instead of answering 503 at once, and the gate's line comes from the model's own estimate when "auto"."""
import time
import unittest
from pathlib import Path
from unittest import mock

from serve.frontend import ChatTemplate
from serve.server import ByteTokenizer, GpuBusy, MockEngine, Service, StrataEngine, vram_fit_start_need


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


def make_service(tok):
    return Service(QuietEngine(tok), tok, ChatTemplate(Path(__file__).parent / "chat_template.jinja"))


class VramGate(unittest.TestCase):
    def setUp(self):
        self.tok = ByteTokenizer()
        self.svc = make_service(self.tok)
        self.svc._release_orphan_encoder = mock.Mock(return_value=False)

    def frees(self, values):
        """A free_vram_mib() that walks the list, holding the last value."""
        seq = iter(values)
        last = [values[-1]]
        def free():
            try:
                v = next(seq)
            except StopIteration:
                v = last[0]
            last[0] = v
            return v
        return free

    def test_off_by_default_loads_immediately(self):
        # nothing configured: the gate is not even read, and the engine starts with 1 MiB "free"
        self.svc.free_vram_mib = mock.Mock(return_value=1)
        self.svc.ensure_loaded()
        self.svc.free_vram_mib.assert_not_called()
        self.assertEqual(self.svc.engine.starts, 1)

    def test_waits_and_loads_when_the_gpu_gives_memory_back(self):
        self.svc.min_free_vram_mib = 10_000
        self.svc.load_wait_s = 2.0
        self.svc.free_vram_mib = self.frees([500, 500, 20_000])
        started = time.monotonic()
        self.svc.ensure_loaded()
        self.assertEqual(self.svc.engine.starts, 1)
        self.assertGreater(time.monotonic() - started, 0.5)     # it really waited, it did not just re-read once

    def test_refuses_only_after_the_wait_budget_expired(self):
        self.svc.min_free_vram_mib = 10_000
        self.svc.load_wait_s = 0.5
        self.svc.free_vram_mib = mock.Mock(return_value=500)
        with self.assertRaises(GpuBusy) as caught:
            self.svc.ensure_loaded()
        self.assertIn("10000 (min_free_vram_mib)", str(caught.exception))
        self.assertIsNotNone(caught.exception.retry_after)
        self.assertEqual(self.svc.engine.starts, 0)

    def test_unknown_free_vram_never_refuses(self):
        self.svc.min_free_vram_mib = 10_000
        self.svc.load_wait_s = 1.0
        self.svc.free_vram_mib = mock.Mock(return_value=None)
        self.svc.ensure_loaded()                                  # None: nothing is claimed, nothing is refused
        self.assertEqual(self.svc.engine.starts, 1)

    def test_auto_line_wins_over_a_lower_set_floor(self):
        self.svc.min_free_vram_mib = 1000
        self.svc.vram_need = "auto"
        self.svc.vram_need_auto_mib = 9000
        self.assertEqual(self.svc.vram_need_mib(), (9000, "vram_need auto"))
        self.svc.load_wait_s = 0.3
        self.svc.free_vram_mib = mock.Mock(return_value=5000)
        with self.assertRaises(GpuBusy) as caught:
            self.svc.ensure_loaded()
        self.assertIn("9000 (vram_need auto)", str(caught.exception))

    def test_set_number_wins_and_lower_auto_changes_nothing(self):
        self.svc.vram_need = 15000
        self.svc.vram_need_auto_mib = 100                         # a low estimate never undercuts a set line
        self.assertEqual(self.svc.vram_need_mib(), (15000, "vram_need"))


class StartNeed(unittest.TestCase):
    ARGS = ["--serve", "--native", "/models/Qwen3.8-Flash-Next-UD-IQ3_XXS.gguf", "--max-context", "262144",
            "--kv", "int8", "--kv-resident", "32768", "--vram-reserve-mib", "2048", "--prefill", "auto:32768",
            "--draft", "dflash2", "--draft-fallback", "none"]

    def test_the_floor_is_the_exact_other_side_of_fits(self):
        # free == need  <=>  fits(): the gate's line and the start's own estimate never disagree
        from strata_control import vram_fit
        need = vram_fit_start_need({"backend": "cuda"}, self.ARGS, "cuda")
        self.assertTrue(need and need > 8000)
        self.assertAlmostEqual(vram_fit.slack_mib(
            "qwen3.8-flash-next", need / 1024.0, 262144, "int8", 2048, "auto", "cuda", 0, 32768, "dflash2"),
            vram_fit.needed_mib("qwen3.8-flash-next", 0, "cuda", "dflash2"), delta=1.0)
        self.assertFalse(vram_fit.fits("qwen3.8-flash-next", (need - 1024) / 1024.0, 262144, "int8", 2048, "auto",
                                       "cuda", 0, 32768, "dflash2"))
        self.assertTrue(vram_fit.fits("qwen3.8-flash-next", (need + 1024) / 1024.0, 262144, "int8", 2048, "auto",
                                      "cuda", 0, 32768, "dflash2"))

    def test_no_claim_without_a_gpu_backend(self):
        self.assertIsNone(vram_fit_start_need({}, self.ARGS, None))
        self.assertIsNone(vram_fit_start_need({"layer_split": True}, self.ARGS, "cuda"))


class CudaTelemetry(unittest.TestCase):
    """The GB10 fallback: when NVML has no frame buffer, _Nvidia merges the CUDA driver's figures."""

    def test_nvml_numbers_pass_through_untouched(self):
        from serve import telemetry
        nvml = mock.Mock()
        nvml.read.return_value = {"mem_used": 1, "mem_total": 2, "util": 5}
        cuda = mock.Mock()
        n = telemetry._Nvidia(0, cuda=cuda)
        n.nvml = nvml
        self.assertEqual(n.read(), {"mem_used": 1, "mem_total": 2, "util": 5})
        cuda.read.assert_not_called()

    def test_cuda_figures_merge_when_nvml_has_no_memory(self):
        from serve import telemetry
        nvml = mock.Mock()
        nvml.read.return_value = {"mem_used": None, "mem_total": None, "util": 5}
        cuda = mock.Mock()
        cuda.read.return_value = {"mem_used": 80 << 30, "mem_total": 121 << 30}
        n = telemetry._Nvidia(0, cuda=cuda)
        n.nvml = nvml
        out = n.read()
        self.assertEqual(out["mem_total"], 121 << 30)
        self.assertEqual(out["mem_used"], 80 << 30)
        self.assertEqual(out["util"], 5)                             # NVML's other readings survive the merge

    def test_a_dying_driver_is_cached_as_a_failure(self):
        from serve import telemetry
        lib = mock.Mock()
        lib.cuInit.return_value = 999                                 # no usable driver
        c = telemetry._CudaMem(0, lib=lib)
        self.assertEqual(c.read(), {})
        calls = lib.cuInit.call_count
        self.assertEqual(c.read(), {})                               # the second read does not hammer the driver
        self.assertEqual(lib.cuInit.call_count, calls)


if __name__ == "__main__":
    unittest.main()
