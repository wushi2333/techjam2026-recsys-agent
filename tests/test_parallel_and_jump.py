from __future__ import annotations

import time
import unittest

from agent.config import load_settings
from agent.recsys.multitask import spec_from
from agent.search.parallel import map_trials, planned_workers
from agent.search.uct import uct_score, not_enabled
from agent.memory.journal import Node
from agent.types import Metrics
from agent.recsys.modules import require_ready


class ReservedHooksTest(unittest.TestCase):
    def test_parallel_default_three(self):
        settings = load_settings()
        self.assertEqual(planned_workers(settings), 3)
        self.assertLessEqual(settings.max_workers, 4)

    def test_llm_n_workers_can_raise_1k_default(self):
        from dataclasses import replace

        settings = replace(load_settings(), data_scale="1k", parallel_enabled=True, max_workers=4)
        self.assertEqual(planned_workers(settings), 1)
        self.assertEqual(planned_workers(settings, requested=3), 3)
        self.assertEqual(planned_workers(settings, requested=9), 4)

    def test_map_trials_overlaps(self):
        def slow(x):
            time.sleep(0.25)
            return x

        t0 = time.time()
        out = map_trials(slow, [1, 2, 3], 3)
        elapsed = time.time() - t0
        self.assertEqual(out, [1, 2, 3])
        self.assertLess(elapsed, 0.6)

    def test_uct_reserved(self):
        n = Node("0", None, "draft", "d", "h", "", Metrics(0.5, 0.5, 0.5), False)
        self.assertTrue(not_enabled())
        self.assertGreater(uct_score(n, 10, 0), 0)

    def test_multitask_main_is_long_view(self):
        spec = spec_from(load_settings())
        self.assertEqual(spec.main_task, "long_view")
        self.assertIn("is_click", spec.auxiliary)
        self.assertFalse(spec.enabled)

    def test_deepfm_ready(self):
        mod = require_ready(load_settings(), "deepfm")
        self.assertEqual(mod.status, "ready")
        self.assertEqual(require_ready(load_settings(), "dcnv2").status, "ready")
