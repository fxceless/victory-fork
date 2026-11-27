"""Regression cases for the offline skin draw comparison. All data is synthetic."""

import runpy
import unittest
from pathlib import Path

compare = runpy.run_path(str(Path(__file__).with_name("compare-draw-trace.py")))
changes = compare["changes"]
readiness = compare["readiness"]
completion = compare["completion"]


class DrawTraceTests(unittest.TestCase):
    def setUp(self):
        self.meta = dict(draw_fence="9", completed_before="7", event_order="3",
                         signal_ok="true", execute_us="90")
        self.row = dict(title="title-texture view_format=28 view_levels=0+1", submit_slot="8",
                        record_state="resource", record_desc="d3:16x16x1:f28:m1:s1",
                        record_object="0xaa", submit_object="0xaa", submit_state="resource",
                        submit_desc="d3:16x16x1:f28:m1:s1", upload_fence="7", upload_order="1",
                        upload_us="40", upload_sample="0000000000000042")

    def test_different_runtime_allocations_can_show_the_same_staged_content(self):
        relocated = dict(self.row, record_object="0xdead", submit_object="0xdead", submit_slot="567")
        self.assertEqual(changes("pixel/table/0/8/Srv", self.row, relocated, self.meta, self.meta), [])

    def test_rebinding_in_one_run_remains_visible_with_matching_upload_samples(self):
        rebound = dict(self.row, submit_object="0xbb")
        self.assertEqual(changes("pixel/table/0/8/Srv", self.row, rebound, self.meta, self.meta),
                         ["binding changed between recording and submission in one run"])

    def test_material_index_change_is_distinct_from_a_sample_change(self):
        moved = dict(self.row, submit_slot="12")
        self.assertEqual(changes("word/M/0x8", self.row, moved, self.meta, self.meta),
                         ["material word selected a different heap index"])
        updated = dict(self.row, upload_sample="0000000000000043")
        self.assertEqual(changes("word/M/0x8", self.row, updated, self.meta, self.meta),
                         ["sampled upload texels changed"])

    def test_unbound_or_unobserved_is_not_a_sample_of_zero(self):
        empty = dict(self.row, submit_state="no-retained", submit_desc="-",
                     upload_fence="-", upload_sample="-")
        diff = changes("word/M/0x8", self.row, empty, self.meta, self.meta)
        self.assertIn("host resource state or description changed", diff)
        self.assertNotIn("sampled upload texels changed", diff)
        self.assertEqual(readiness(empty, self.meta), "no tracked staging copy")

    def test_fence_order_and_completion_upper_bound(self):
        t = ([4, 8, 10], [45, 73, 104], {})
        self.assertIn("complete by 73 us", completion(self.row, self.meta, t))
        self.assertEqual(readiness(dict(self.row, upload_fence="8"), self.meta),
                         "earlier submission in flight at enqueue")
        same = dict(self.row, upload_fence="9", upload_order="2")
        self.assertEqual(readiness(same, self.meta), "copy precedes draw in the same submission")
        self.assertEqual(readiness(dict(same, upload_order="4"), self.meta), "inconsistent queue order")
        self.assertEqual(readiness(self.row, dict(self.meta, completed_before=str(2**64 - 1))),
                         "fence unavailable")

    def test_changed_mip_range_even_when_the_uploaded_bytes_match(self):
        other = dict(self.row, title="title-texture view_format=28 view_levels=2+1")
        self.assertEqual(changes("word/M/0x8", self.row, other, self.meta, self.meta),
                         ["title view_levels changed"])


if __name__ == "__main__":
    unittest.main()
