"""L0: scoring contract pinned by the starter kit."""

from __future__ import annotations

LABEL = "long_view"
PRIMARY = "primary"
METRICS = ("GAUC", "nDCG@5", PRIMARY)
TASK = "within-user ranking over logged impressions"

TRAIN_DATES = (20220408, 20220421)
VALID_DATES = (20220422, 20220428)
TEST_DATES = (20220429, 20220508)

FM_VALID_GAUC = 0.6674
FM_VALID_NDCG5 = 0.5357
FM_VALID_PRIMARY = 0.6016
FM_TEST_PRIMARY = 0.5946
RANDOM_PRIMARY = 0.4753
EPSILON = 0.002
PATIENCE_N = 3

FORBIDDEN_NAMES = ("evaluate.py",)
SUBMISSION_HEADER = ("row_id", "user_id", "video_id", "score")

ORGANIZER_DEAD_ENDS = (
    "static_features",
    "embedding_capacity",
)

HEADROOM = (
    "pairwise_or_listwise_loss",
    "user_history_sequence",
    "multitask_aux_for_long_view",
    "watch_time_cwm",
    "deepfm_dcn_after_loss",
)
