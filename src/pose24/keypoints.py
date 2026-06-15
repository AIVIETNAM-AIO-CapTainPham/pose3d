"""POSE-24 keypoint definitions derived from MHR70 (SAM-3D-Body).

24 body keypoints = standard 17 COCO body + 7 additional anatomical landmarks
(olecranon, cubital fossa, acromion, neck).
"""

from __future__ import annotations

# ── Names ─────────────────────────────────────────────────────────────────
POSE24_KEYPOINT_NAMES: tuple[str, ...] = (
    "nose",  # 0
    "left_eye",  # 1
    "right_eye",  # 2
    "left_ear",  # 3
    "right_ear",  # 4
    "left_shoulder",  # 5
    "right_shoulder",  # 6
    "left_elbow",  # 7
    "right_elbow",  # 8
    "left_wrist",  # 9
    "right_wrist",  # 10
    "left_hip",  # 11   ← root joint (with right_hip)
    "right_hip",  # 12   ← root joint (with left_hip)
    "left_knee",  # 13
    "right_knee",  # 14
    "left_ankle",  # 15
    "right_ankle",  # 16
    "left_olecranon",  # 17
    "right_olecranon",  # 18
    "left_cubital_fossa",  # 19
    "right_cubital_fossa",  # 20
    "left_acromion",  # 21
    "right_acromion",  # 22
    "neck",  # 23
)
NUM_KEYPOINTS: int = len(POSE24_KEYPOINT_NAMES)  # 24

# ── MHR70 source indices for each POSE-24 keypoint ────────────────────────
# Maps POSE-24 index → MHR70 index.
MHR70_INDICES: tuple[int, ...] = (
    0,  # nose
    1,  # left_eye
    2,  # right_eye
    3,  # left_ear
    4,  # right_ear
    5,  # left_shoulder
    6,  # right_shoulder
    7,  # left_elbow
    8,  # right_elbow
    62,  # left_wrist
    41,  # right_wrist
    9,  # left_hip
    10,  # right_hip
    11,  # left_knee
    12,  # right_knee
    13,  # left_ankle
    14,  # right_ankle
    63,  # left_olecranon
    64,  # right_olecranon
    65,  # left_cubital_fossa
    66,  # right_cubital_fossa
    67,  # left_acromion
    68,  # right_acromion
    69,  # neck
)

# Root joints used for Z-axis normalisation in SimCC3D.
ROOT_INDICES: tuple[int, int] = (11, 12)  # left_hip, right_hip

# ── Flip mapping (left ↔ right) ───────────────────────────────────────────
# FLIP_INDICES[i] = j means keypoint i maps to keypoint j under horizontal flip.
FLIP_INDICES: tuple[int, ...] = (
    0,  # nose → nose
    2,  # left_eye → right_eye
    1,  # right_eye → left_eye
    4,  # left_ear → right_ear
    3,  # right_ear → left_ear
    6,  # left_shoulder → right_shoulder
    5,  # right_shoulder → left_shoulder
    8,  # left_elbow → right_elbow
    7,  # right_elbow → left_elbow
    10,  # left_wrist → right_wrist
    9,  # right_wrist → left_wrist
    12,  # left_hip → right_hip
    11,  # right_hip → left_hip
    14,  # left_knee → right_knee
    13,  # right_knee → left_knee
    16,  # left_ankle → right_ankle
    15,  # right_ankle → left_ankle
    18,  # left_olecranon → right_olecranon
    17,  # right_olecranon → left_olecranon
    20,  # left_cubital_fossa → right_cubital_fossa
    19,  # right_cubital_fossa → left_cubital_fossa
    22,  # left_acromion → right_acromion
    21,  # right_acromion → left_acromion
    23,  # neck → neck
)

# ── Skeleton (bone edges) ─────────────────────────────────────────────────
SKELETON_EDGES: tuple[tuple[int, int], ...] = (
    (15, 13),
    (13, 11),  # left leg
    (16, 14),
    (14, 12),  # right leg
    (11, 12),  # hips
    (5, 11),
    (6, 12),  # torso sides
    (5, 6),  # shoulders
    (5, 7),
    (7, 9),  # left arm
    (6, 8),
    (8, 10),  # right arm
    (0, 1),
    (0, 2),  # nose–eyes
    (1, 3),
    (2, 4),  # eyes–ears
    (3, 5),
    (4, 6),  # ears–shoulders
    (23, 5),
    (23, 6),  # neck–shoulders
    (7, 17),
    (8, 18),  # elbow–olecranon
    (5, 21),
    (6, 22),  # shoulder–acromion
)

# ── Bone-loss parent tree ─────────────────────────────────────────────────
# JOINT_PARENTS[i] = parent index of keypoint i in the kinematic tree.
# Root joints point to themselves (left_hip = 11).
JOINT_PARENTS: tuple[int, ...] = (
    23,  # 0  nose       → neck
    0,  # 1  left_eye   → nose
    0,  # 2  right_eye  → nose
    1,  # 3  left_ear   → left_eye
    2,  # 4  right_ear  → right_eye
    23,  # 5  left_shoulder  → neck
    23,  # 6  right_shoulder → neck
    5,  # 7  left_elbow → left_shoulder
    6,  # 8  right_elbow→ right_shoulder
    7,  # 9  left_wrist → left_elbow
    8,  # 10 right_wrist→ right_elbow
    11,  # 11 left_hip   → self (root)
    11,  # 12 right_hip  → left_hip
    11,  # 13 left_knee  → left_hip
    12,  # 14 right_knee → right_hip
    13,  # 15 left_ankle → left_knee
    14,  # 16 right_ankle→ right_knee
    7,  # 17 left_olecranon  → left_elbow
    8,  # 18 right_olecranon → right_elbow
    7,  # 19 left_cubital_fossa  → left_elbow
    8,  # 20 right_cubital_fossa → right_elbow
    5,  # 21 left_acromion  → left_shoulder
    6,  # 22 right_acromion → right_shoulder
    11,  # 23 neck        → left_hip (spine root)
)
