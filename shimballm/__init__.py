"""ShimbaLLM: compute-efficient small language model training.

Implements the CROP recipe (Curriculum-guided Reduction Of Pretraining):

  Phase 0   offline reference scoring (excess loss, Rho-1 style)
  Stage A   document-level gating, full loss, short context   (WSD warmup -> stable)
  Stage B   selective token replay, context ramp 2K -> 4K     (stable LR)
  Stage C   LR annealing on the top-quality slice             (WSD decay)
  Recycle   re-anneal branches from the Stage-B stable checkpoint

Reference: "Smaller, Faster, Smarter: Loss-Gated Data Staging with Selective
Token Replay for Compute-Efficient Small Language Model Training" (2025).
"""

__version__ = "0.1.0"

from .config import RunConfig, ModelConfig, RefConfig, DataConfig, CROPSettings, ScheduleConfig, EngineConfig, HubConfig
from .models.tiny_gpt import TinyGPT
from .models import build_model, build_reference_model
from .optim.lr import WSDLRSchedule
from .engine.flops import FLOPsMeter, project_crop, format_flops
from .engine.trainer import CROPTrainer
from .engine.recycling import AnnealBranchManager

__all__ = [
    "__version__",
    "RunConfig",
    "ModelConfig",
    "RefConfig",
    "DataConfig",
    "CROPSettings",
    "ScheduleConfig",
    "EngineConfig",
    "HubConfig",
    "TinyGPT",
    "build_model",
    "build_reference_model",
    "WSDLRSchedule",
    "FLOPsMeter",
    "project_crop",
    "format_flops",
    "CROPTrainer",
    "AnnealBranchManager",
]
