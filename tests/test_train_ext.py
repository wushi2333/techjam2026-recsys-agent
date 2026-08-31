from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "templates"))

from fm import FM  # noqa: E402
from sampling import iter_user_batches  # noqa: E402
from train import prepare_stop_split, should_eval, train_limits  # noqa: E402


class TrainExtTest(unittest.TestCase):
    def test_train_limits_screen_not_smoke(self):
        cfg = {
            "epochs": 40,
            "patience": 4,
            "budget_epochs": 6,
            "budget_patience": 2,
        }
        self.assertEqual(train_limits(cfg), (6, 2))

    def test_train_limits_ablate_full(self):
        self.assertEqual(train_limits({"epochs": 40, "patience": 4}), (40, 4))

    def test_train_limits_smoke_wins(self):
        cfg = {
            "smoke": True,
            "epochs": 1,
            "patience": 4,
            "budget_epochs": 6,
            "budget_patience": 2,
        }
        self.assertEqual(train_limits(cfg), (1, 2))

    def test_listwise_runs(self):
        rng = np.random.default_rng(0)
        x = rng.integers(0, 20, size=(16, 5), dtype=np.int32)
        y = np.array([1, 0, 1, 0] * 4, dtype=np.float32)
        users = ["a"] * 8 + ["b"] * 8
        m = FM(20, k=4, lr=0.01, seed=0)
        loss = m.step_listwise(x, y, users=users)
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(m.listwise_gain, "uniform")

    def test_listwise_ndcg_runs(self):
        rng = np.random.default_rng(0)
        x = rng.integers(0, 20, size=(16, 5), dtype=np.int32)
        y = np.array([1, 0, 1, 0] * 4, dtype=np.float32)
        users = ["a"] * 8 + ["b"] * 8
        m = FM(20, k=4, lr=0.01, seed=0, listwise_gain="ndcg")
        loss = m.step_listwise(x, y, users=users)
        self.assertTrue(np.isfinite(loss))

    def test_gbm_default_params_are_safe(self):
        from gbm import cat_columns, gbm_params

        rng = np.random.default_rng(0)
        x = np.stack(
            [
                np.arange(2500),
                rng.integers(0, 8, size=2500),
            ],
            axis=1,
        )
        self.assertEqual(cat_columns(x, "none"), [])
        self.assertEqual(cat_columns(x, "all"), [0, 1])
        self.assertEqual(cat_columns(x, "lowcard"), [1])
        p = gbm_params({"seed": 0}, smoke=False)
        self.assertTrue(p["deterministic"])
        self.assertEqual(p["min_data_in_bin"], 3)
        self.assertEqual(p["_cat"], "lowcard")

    def test_gbm_drops_singleton_users(self):
        from gbm import _drop_singletons, _group_order

        users = ["a", "a", "b", "c", "c", "c"]
        order, groups = _group_order(users)
        keep, kept_g = _drop_singletons(order, groups)
        self.assertEqual(kept_g, [2, 3])
        kept_users = [users[int(i)] for i in order[keep]]
        self.assertNotIn("b", kept_users)

    def test_user_batches_never_split_a_user(self):
        from collections import Counter

        users = ["a"] * 3 + ["b"] * 5 + ["c"] * 2
        total = Counter(users)
        rng = np.random.default_rng(0)
        seen = []
        for sl in iter_user_batches(users, batch_rows=4, rng=rng):
            names = [users[int(i)] for i in sl]
            for u, c in Counter(names).items():
                self.assertEqual(c, total[u])
                self.assertNotIn(u, seen)
            seen.extend(set(names))
        self.assertEqual(set(seen), {"a", "b", "c"})

    def test_din_step_and_predict(self):
        rng = np.random.default_rng(1)
        x = rng.integers(0, 12, size=(8, 5), dtype=np.int32)
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float32)
        h = rng.integers(0, 12, size=(8, 4), dtype=np.int32)
        mask = np.ones((8, 4), dtype=np.float32)
        mask[:, 0] = 0
        m = FM(12, k=4, lr=0.01, seed=0, seq_len=4, seq_mode="din")
        loss = m.step_logloss(x, y, h, mask)
        scores = m.predict(x, h, mask)
        self.assertTrue(np.isfinite(loss))
        self.assertEqual(len(scores), 8)
        self.assertTrue(np.isfinite(scores).all())

    def test_bpr_within_user_not_cross_user(self):
        x = np.arange(4, dtype=np.int32).reshape(4, 1).repeat(5, axis=1)
        y = np.array([1, 1, 0, 0], dtype=np.float32)
        users = ["a", "a", "a", "b"]
        m_w = FM(8, k=4, lr=0.05, seed=0, l2=0.0)
        m_g = FM(8, k=4, lr=0.05, seed=0, l2=0.0)
        v_w0 = m_w.V[3].copy()
        v_g0 = m_g.V[3].copy()
        loss_w = m_w.step_bpr(x, y, users=users)
        loss_g = m_g.step_bpr_global(x, y, users=users)
        self.assertTrue(np.isfinite(loss_w))
        self.assertTrue(np.isfinite(loss_g))
        self.assertTrue(np.allclose(m_w.V[3], v_w0))
        self.assertFalse(np.allclose(m_g.V[3], v_g0))
        mixed = ["a", "b", "a", "b"]
        y_mixed = np.array([1, 0, 1, 0], dtype=np.float32)
        m_a = FM(8, k=4, lr=0.05, seed=0, l2=0.0)
        m_b = FM(8, k=4, lr=0.05, seed=0, l2=0.0)
        loss_fb = m_a.step_bpr(x, y_mixed, users=mixed)
        loss_ll = m_b.step_logloss(x, y_mixed)
        self.assertAlmostEqual(loss_fb, loss_ll)

    def test_prepare_stop_split_default_off(self):
        dates = np.array([20220408, 20220418, 20220419, 20220421] * 30, dtype=np.int32)
        n = len(dates)
        enc = {
            "train": (
                np.zeros((n, 2), dtype=np.int32),
                np.zeros(n, dtype=np.float32),
                ["u"] * n,
            )
        }

        class Rows:
            date = dates

        splits = {"train": Rows()}
        prepare_stop_split(enc, splits, {})
        self.assertNotIn("stop", enc)
        self.assertEqual(len(enc["train"][1]), n)
        prepare_stop_split(enc, splits, {"train_tail_stop": True})
        self.assertIn("stop", enc)
        self.assertEqual(len(enc["train"][1]), int((dates < 20220419).sum()))
        self.assertEqual(len(enc["stop"][1]), int((dates >= 20220419).sum()))

    def test_cwm_independent_does_not_read_z(self):
        x = np.arange(4, dtype=np.int32).reshape(4, 1).repeat(5, axis=1)
        play = np.array([500.0, 500.0, 2000.0, 100.0], dtype=np.float32)
        dur = np.array([2000.0, 2000.0, 2000.0, 2000.0], dtype=np.float32)
        aux_ind = {"play": play, "dur": dur, "w_cwm": 1.0, "cwm_head": "independent"}
        aux_sh = {"play": play, "dur": dur, "w_cwm": 1.0, "cwm_head": "shared"}
        m_a = FM(8, k=4, lr=0.05, seed=0)
        m_b = FM(8, k=4, lr=0.05, seed=0)
        E = m_a.V[x]
        z0 = np.zeros(4, dtype=np.float32)
        z1 = np.ones(4, dtype=np.float32) * 5.0
        g = np.zeros(4, dtype=np.float32)
        loss_a, g_cwm = m_a._cwm_aux(z0, g, aux_ind, E=E, X=x)
        loss_b, _ = m_b._cwm_aux(z1, np.zeros(4, dtype=np.float32), aux_ind, E=E, X=x)
        self.assertTrue(np.allclose(g, 0.0))
        self.assertIsNone(g_cwm)
        self.assertAlmostEqual(loss_a, loss_b, places=6)
        m2 = FM(8, k=4, lr=0.05, seed=0)
        g2 = np.zeros(4, dtype=np.float32)
        loss_sh0, _ = m2._cwm_aux(z0, g2, aux_sh, E=E, X=x)
        self.assertFalse(np.allclose(g2, 0.0))
        m3 = FM(8, k=4, lr=0.05, seed=0)
        loss_sh1, _ = m3._cwm_aux(z1, np.zeros(4, dtype=np.float32), aux_sh, E=E, X=x)
        self.assertNotAlmostEqual(loss_sh0, loss_sh1, places=3)
        m1 = FM(8, k=4, lr=0.05, seed=0)
        before_w = m1.W_cwm.copy()
        m1.step_logloss(x, np.array([1, 0, 1, 0], dtype=np.float32), aux=aux_ind)
        self.assertFalse(np.allclose(m1.W_cwm, before_w))

    def test_deepfm_and_dcnv2_train(self):
        rng = np.random.default_rng(0)
        x = rng.integers(0, 12, size=(8, 5), dtype=np.int32)
        y = np.array([1, 0, 1, 0, 1, 0, 1, 0], dtype=np.float32)
        fm = FM(12, k=4, lr=0.05, seed=0, arch="fm")
        deep = FM(12, k=4, lr=0.05, seed=0, arch="deepfm")
        dcn = FM(12, k=4, lr=0.05, seed=0, arch="dcnv2")
        z_fm = fm.predict(x)
        z_deep = deep.predict(x)
        z_dcn = dcn.predict(x)
        self.assertFalse(np.allclose(z_fm, z_deep))
        self.assertFalse(np.allclose(z_fm, z_dcn))
        before = deep.arch.W1.copy()
        deep.step_logloss(x, y)
        self.assertFalse(np.allclose(deep.arch.W1, before))
        before_c = dcn.arch.Wc.copy()
        dcn.step_logloss(x, y)
        self.assertFalse(np.allclose(dcn.arch.Wc, before_c))

    def test_should_eval_last_epoch(self):
        self.assertFalse(should_eval(1, 6, 2))
        self.assertTrue(should_eval(2, 6, 2))
        self.assertTrue(should_eval(6, 6, 2))
        self.assertTrue(should_eval(1, 1, 2))

    def test_user_batches_keep_groups(self):
        users = ["u0"] * 3 + ["u1"] * 5 + ["u2"] * 2
        rng = np.random.default_rng(0)
        seen = []
        for sl in iter_user_batches(users, batch_rows=6, rng=rng):
            batch_users = [users[int(i)] for i in sl]
            seen.extend(batch_users)
            for u in set(batch_users):
                self.assertEqual(batch_users.count(u), users.count(u))
        self.assertEqual(sorted(seen), sorted(users))
