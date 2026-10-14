"""Shared configuration: data locations, backbone registry, prompt templates.

Modules in ``src/`` are flat files (same convention as assignment_2) and are
run as ``python3 src/<script>.py`` from the assignment root, so ``import config``
resolves to this file.
"""
from __future__ import annotations

from pathlib import Path

# Folder name of the competition data inside this assignment directory.
DATA_SUBDIR = "match-the-dish-to-its-rizz"

# Files that must exist for the pipeline to run.
REQUIRED_FILES = (
    "train.csv",
    "train_queries.csv",
    "category_metadata.csv",
    "validation_gallery.csv",
    "validation_queries.csv",
    "validation_ground_truth.csv",
    "test_gallery.csv",
    "test_queries.csv",
)

# key -> (open_clip architecture name, pretrained tag)
#
# Ordered roughly by ImageNet zero-shot strength per unit of compute. All of
# these are *generic* pretrained models, which the competition rules allow; the
# report must name whichever ones were used.
BACKBONES = {
    # cheap smoke-test backbone (~150M params, seconds per 1k images)
    "clip-b32-laion": ("ViT-B-32", "laion2b_s34b_b79k"),
    # strong, cheap, 384px input: the workhorse default
    "siglip2-so400m-384": ("ViT-SO400M-16-SigLIP2-384", "webli"),
    # strongest ImageNet zero-shot of the SigLIP2 family
    "siglip2-gopt-384": ("ViT-gopt-16-SigLIP2-384", "webli"),
    # DFN-5B CLIP ViT-H/14 @378
    "dfn5b-h14-378": ("ViT-H-14-378", "dfn5b"),
    # LAION-2B CLIP ViT-H/14 (independent pretraining corpus -> good ensemble partner)
    "clip-h14-laion": ("ViT-H-14", "laion2b_s32b_b79k"),
    # OpenAI CLIP ViT-L/14 @336, useful as a "classic CLIP" reference point
    "clip-l14-336": ("ViT-L-14-336", "openai"),
    # MetaCLIP ViT-bigG/14 @448 - best zero-shot, needs a big GPU
    "pe-bigG-448": ("PE-Core-bigG-14-448", "meta"),
}

DEFAULT_BACKBONE = "siglip2-so400m-384"

# Prompt ensemble for category prototypes. Averaging the text embedding over
# several phrasings is worth ~1-2 points of zero-shot accuracy over a single
# "a photo of {}" template, which matters here because the whole image->category
# decision is driven by these prototypes.
PROMPT_TEMPLATES = (
    "a photo of {name}",
    "a photo of {name}, a type of food",
    "a photo of {name}, a type of dish",
    "a close-up photo of {name}",
    "a photo of a plate of {name}",
    "a food photograph of {name}",
    "an image of {name}",
    "{name}",
)

# Same idea, but injecting the dish description from category_metadata.csv.
# Ambiguous dishes (several pasta/cake/rice categories) get a prototype that
# states what the dish actually is, not just its name.
DESCRIPTION_TEMPLATES = (
    "a photo of {name}: {description}",
    "a photo of {name}, which is {description}",
    "{name}, {description}",
)

# Text strings used to map a *query* onto a category (used by textmap.py).
QUERY_CATEGORY_TEMPLATES = (
    "{name}",
    "{name}, a type of food",
    "{description}",
    "a photo of {name}",
)

# Test-time augmentation views built by dataset.build_eval_views().
# "center" = standard CLIP preprocessing, "full" = squash the whole photo
# (no crop: a centre crop often cuts the plate away), "zoom" = mild zoom-in.
# Flips are cheap extra views and food photos are left/right symmetric.
EVAL_VIEWS = ("center", "full", "zoom", "flip_center", "flip_full", "flip_zoom")
DEFAULT_VIEWS = ("center", "full", "flip_center", "flip_full")

DEFAULT_TOP_K = 20


def resolve_data_dir(path: str | Path | None = None) -> Path:
    """Locate the competition data directory.

    Checks (in order): explicit ``path``, ``assignment_3/<DATA_SUBDIR>``,
    ``../<DATA_SUBDIR>``, then any single sub-folder of ``/kaggle/input`` that
    contains ``train.csv`` (Kaggle mounts the data under a slug).
    """
    candidates: list[Path] = []
    if path:
        candidates.append(Path(path))
    else:
        candidates += [
            Path(DATA_SUBDIR),
            Path("..") / DATA_SUBDIR,
            repo_root() / DATA_SUBDIR,
        ]
        kaggle_input = Path("/kaggle/input")
        if kaggle_input.is_dir():
            candidates += sorted(p for p in kaggle_input.iterdir() if p.is_dir())

    for cand in candidates:
        if all((cand / name).exists() for name in REQUIRED_FILES):
            return cand.resolve()

    tried = "\n  ".join(str(c) for c in candidates[:6]) or "(none)"
    raise FileNotFoundError(
        "Could not find the competition data.\n"
        f"Looked for {REQUIRED_FILES[0]} and friends in:\n  {tried}\n"
        "Pass --data <dir> (Kaggle: /kaggle/input/<competition-slug>)."
    )


def repo_root() -> Path:
    """assignment_3/ (the parent of src/)."""
    return Path(__file__).resolve().parents[1]


def feature_cache_dir() -> Path:
    """Where embeddings are cached between runs (git-ignored)."""
    path = repo_root() / "features"
    path.mkdir(parents=True, exist_ok=True)
    return path


def backbone_spec(key: str) -> tuple[str, str]:
    """Return ``(architecture, pretrained_tag)`` for a registry key."""
    if key not in BACKBONES:
        raise KeyError(f"unknown backbone {key!r}; choose from {sorted(BACKBONES)}")
    return BACKBONES[key]


def cache_key(*parts: str) -> str:
    """Filesystem-safe key for a cached embedding matrix."""
    return "_".join(str(p).replace("/", "-").replace(" ", "-") for p in parts)