"""The unified-card VRAM reading (serve/telemetry.py): on GB10/Tegra the card's total is the machine's RAM and
the driver's free counts reclaimable page cache as used, so the gate's line comes up to MemAvailable - while a
discrete frame buffer's reading never moves."""
import os
import tempfile
import unittest

from serve.telemetry import meminfo_kb, unified_used

GB10_TOTAL = 124544 * 2 ** 20        # what cuMemGetInfo reports as total on the DGX Spark, bytes
GB10_MT = 124544 * 1024              # the same RAM seen from /proc/meminfo, MemTotal in kB (one rounding apart)
GB10_AVAIL = 116578 * 1024           # MemAvailable in kB: the engine unloaded, ~40 GiB of page cache warm
GB10_USED = 45410 * 2 ** 20          # what cuMemGetInfo reports as used with all that cache counted "used"
GB10_MEM = {"MemTotal": GB10_MT, "MemAvailable": GB10_AVAIL}


class UnifiedUsed(unittest.TestCase):
    def test_a_warm_cache_is_free_for_a_loader(self):
        self.assertEqual(unified_used(GB10_USED, GB10_TOTAL, GB10_MEM), GB10_TOTAL - GB10_AVAIL * 1024)

    def test_a_discrete_frame_buffer_is_left_alone(self):
        total = 24576 * 2 ** 20                          # a 24 GB card in a 128 GB machine
        self.assertIsNone(unified_used(12 * 2 ** 30, total,
                                       {"MemTotal": 128 * 1024 * 1024, "MemAvailable": 120 * 1024 * 1024}))

    def test_no_lowering_when_the_driver_is_already_roomier(self):
        self.assertIsNone(unified_used(GB10_TOTAL - GB10_AVAIL * 1024, GB10_TOTAL, GB10_MEM))

    def test_no_meminfo_no_claim(self):
        for mem in ({}, {"MemTotal": GB10_MT}, {"MemAvailable": GB10_AVAIL}, {"MemTotal": 0}):
            self.assertIsNone(unified_used(GB10_USED, GB10_TOTAL, mem))


class Meminfo(unittest.TestCase):
    def test_reads_the_two_figures(self):
        fd, path = tempfile.mkstemp(suffix="-meminfo")
        try:
            with os.fdopen(fd, "w", encoding="ascii") as f:
                f.write("MemTotal:  124544 kB\nMemFree: 1 kB\nBuffers: 2 kB\nMemAvailable: 99 kB\n")
            self.assertEqual(meminfo_kb(path), {"MemTotal": 124544, "MemAvailable": 99})
        finally:
            os.unlink(path)

    def test_no_proc_is_not_an_error(self):
        self.assertEqual(meminfo_kb("/nonexistent/proc-meminfo"), {})


if __name__ == "__main__":
    unittest.main()
