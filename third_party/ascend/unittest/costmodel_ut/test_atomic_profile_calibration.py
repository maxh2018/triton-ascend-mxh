import json
from pathlib import Path
import unittest


class AtomicProfileCalibrationTest(unittest.TestCase):

    def test_loaded_index_fadd_preserves_measured_endpoint_ordering(self):
        ascend = Path(__file__).resolve().parents[2]
        profile_path = ascend / "costmodel/profiles/simd_simt/david_v100_simd_simt_v1.json"
        profile = json.loads(profile_path.read_text())
        simd = profile["simd"]["stage_resources"]["atomic_memory"]["operations"]["fadd.f32"]
        simt = profile["simt"]["stage_resources"]["atomic_memory"]["operations"]["fadd.f32"]

        measured_unique_simd_us = 464.1674906015396
        measured_unique_simt_us = 239.29700255393982
        measured_hotspot_simd_us = 8503.260612487793
        measured_hotspot_simt_us = 13636.185646057129

        modeled_unique_cost_ratio = (
            simt["logical_elements_per_system_cycle"]
            / simd["logical_elements_per_system_cycle"]
        )
        measured_unique_throughput_ratio = measured_unique_simd_us / measured_unique_simt_us
        self.assertAlmostEqual(modeled_unique_cost_ratio, measured_unique_throughput_ratio, places=12)

        modeled_unknown_contention_cost_ratio = (
            simd["logical_elements_per_system_cycle"]
            / simt["logical_elements_per_system_cycle"]
            * simt["unknown_contention_multiplier"]
            / simd["unknown_contention_multiplier"]
        )
        measured_hotspot_cost_ratio = measured_hotspot_simt_us / measured_hotspot_simd_us
        self.assertAlmostEqual(
            modeled_unknown_contention_cost_ratio,
            measured_hotspot_cost_ratio,
            places=12,
        )
        self.assertLess(measured_unique_simt_us, measured_unique_simd_us)
        self.assertLess(measured_hotspot_simd_us, measured_hotspot_simt_us)


if __name__ == "__main__":
    unittest.main()
