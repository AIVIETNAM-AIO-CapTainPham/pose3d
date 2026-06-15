"""pose24 — RTMPose3D finetune package for MHR70 body-only 24-keypoint subset."""

from importlib.metadata import version, PackageNotFoundError

try:
    __version__ = version("pose24")
except PackageNotFoundError:
    __version__ = "0.0.0"

# Side-effect: registers SimCC3DLabel with KEYPOINT_CODECS.
from .codecs import SimCC3DLabel  # noqa: F401

# Side-effect: registers GTJsonDataset with mmpose's DATASETS registry and
# Flip3DKeypoints with the TRANSFORMS registry.
from .datasets import Flip3DKeypoints, GTJsonDataset  # noqa: F401

# Side-effect: registers the 3D estimator / head / loss with MODELS.
from .models import (  # noqa: F401
    KLDiscretLossWithWeight,
    Pose3DStructureLoss,
    RTMW3DHead,
    TopdownPoseEstimator3D,
)

# Side-effect: registers the visualization hook with mmpose's HOOKS registry.
from .engine import Pose3DVisualizationHook  # noqa: F401

__all__ = [
    "SimCC3DLabel",
    "GTJsonDataset",
    "Flip3DKeypoints",
    "KLDiscretLossWithWeight",
    "Pose3DStructureLoss",
    "RTMW3DHead",
    "TopdownPoseEstimator3D",
    "Pose3DVisualizationHook",
]
