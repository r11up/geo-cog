"""Project paths and hyperparameters."""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Tuple


@dataclass
class PathConfig:
    eav_root:       Path = Path(
        __import__("os").environ.get("EAV_ROOT", "data/eav_dataset")
    )
    cache_dir:      Path = Path("data/cache")
    checkpoint_dir: Path = Path("checkpoints")
    output_dir:     Path = Path("outputs")
    seediv_pretrain_ckpt: Path = Path(
        __import__("os").environ.get("SEEDIV_CKPT", "checkpoints/eeg_conformer_seediv.pt")
    )


@dataclass
class PreprocessConfig:
    native_fs:        int   = 500
    target_fs:        int   = 100
    bp_low:           float = 3.0
    bp_high:          float = 50.0
    butter_order:     int   = 6
    asr_threshold:    float = 5.0
    n_ica_components: int   = 20
    ica_random_state: int   = 42
    epoch_sec:        float = 5.0
    n_channels:       int   = 30

    @property
    def epoch_samples(self) -> int:
        return int(self.epoch_sec * self.target_fs)


@dataclass
class EAVConfig:
    n_subjects:             int             = 42
    n_channels:             int             = 30
    speaking_label_indices: Tuple[int, ...] = (1, 3, 5, 7, 9)
    class_names:            Tuple[str, ...] = (
        "neutral", "sadness", "anger", "happiness", "relaxed"
    )
    n_classes:          int = 5
    trials_per_class:   int = 20
    epochs_per_trial:   int = 4
    epochs_per_subject: int = 400


@dataclass
class GeoConfig:
    grid_size:              int            = 10
    n_nodes:                int            = 100
    epsg_astana:            int            = 32641
    epsg_singapore:         int            = 3414
    node_feature_dim:       int            = 6
    epsilon:                float          = 1e-6
    high_exposure_quartile: float          = 0.75
    low_exposure_quartile:  float          = 0.25
    maup_resolutions:       Tuple[int,...] = (1, 5, 10)
    buffer_radius_m:        int            = 250
    gee_project:            str            = None
    openaq_api_key:         str            = None
    satellite_year:         int            = 2025


@dataclass
class ModelConfig:
    embed_dim:           int       = 256
    eeg_temporal_kernel: int       = 25
    conformer_heads:     int       = 8
    conformer_ff_dim:    int       = 512
    conformer_layers:    int       = 2
    conformer_dropout:   float     = 0.5
    gcn_intermediate:    int       = 64
    gcn_hidden:          int       = 128
    gcn_layers:          int       = 2
    gru_hidden:          int       = 128
    gru_layers:          int       = 1
    attn_heads:          int       = 8
    fused_dim:           int       = 512
    clf_hidden_dims:     List[int] = field(default_factory=lambda: [256])
    clf_dropout:         float     = 0.3
    lambda_align:        float     = 0.1


@dataclass
class TrainConfig:
    n_folds:       int   = 5
    random_seed:   int   = 42
    learning_rate: float = 1e-4
    weight_decay:  float = 1e-4
    batch_size:    int   = 64
    n_epochs:      int   = 100
    cosine_t_max:  int   = 100
    patience:      int   = 15
    device:        str   = "cuda"


@dataclass
class Config:
    paths:      PathConfig       = field(default_factory=PathConfig)
    preprocess: PreprocessConfig = field(default_factory=PreprocessConfig)
    eav:        EAVConfig        = field(default_factory=EAVConfig)
    geo:        GeoConfig        = field(default_factory=GeoConfig)
    model:      ModelConfig      = field(default_factory=ModelConfig)
    train:      TrainConfig      = field(default_factory=TrainConfig)

    def __post_init__(self):
        try:
            import torch
            if self.train.device == "cuda" and not torch.cuda.is_available():
                self.train.device = "cpu"
        except ImportError:
            self.train.device = "cpu"
        for attr in ("cache_dir", "checkpoint_dir", "output_dir"):
            getattr(self.paths, attr).mkdir(parents=True, exist_ok=True)


CFG = Config()
