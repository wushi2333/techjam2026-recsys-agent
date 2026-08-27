from __future__ import annotations

import unittest

from agent.config import load_settings
from agent.recsys.multitask import spec_from
from agent.search.parallel import planned_workers
from agent.search.uct import uct_score, not_enabled
from agent.memory.journal import Node
from agent.types import Metrics
from agent.recsys.modules import ReservedModuleError, require_ready


class ReservedHooksTest(unittest.TestCase):
    def test_parallel_default_one(self):
        settings = load_settings()
        self.assertEqual(planned_workers(settings), 1)
        self.assertLessEqual(settings.max_workers, 4)

    def test_uct_reserved(self):
        n = Node("0", None, "draft", "d", "h", "", Metrics(0.5, 0.5, 0.5), False)
        self.assertTrue(not_enabled())
        self.assertGreater(uct_score(n, 10, 0), 0)

    def test_multitask_main_is_long_view(self):
        spec = spec_from(load_settings())
        self.assertEqual(spec.main_task, "long_view")
        self.assertIn("is_click", spec.auxiliary)
        self.assertFalse(spec.enabled)

    def test_deepfm_reserved(self):
        with self.assertRaises(ReservedModuleError):
            require_ready(load_settings(), "deepfm")
