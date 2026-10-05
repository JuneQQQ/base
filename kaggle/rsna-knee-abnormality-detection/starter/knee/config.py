"""Shared constants. Training and inference MUST use identical values here."""
from dataclasses import dataclass

LABELS = [
    "ACL", "MCL", "Medial Meniscus", "Lateral Meniscus",
    "Medial OA", "Lateral OA", "PF OA",
    "Effusion", "Synovitis", "Baker's", "Contusion", "Fracture",
]
N_LABELS = len(LABELS)

# Kaggle mounts competition data at one of these roots depending on the image version.
COMP_SLUG = "rsna-knee-abnormality-detection"
KAGGLE_COMP_ROOTS = [
    f"/kaggle/input/competitions/{COMP_SLUG}",
    f"/kaggle/input/{COMP_SLUG}",
]

# Two training DICOMs have truncated PixelData (discussion 737163 / 738708).
KNOWN_BAD_STUDIES = {
    "1.2.826.0.1.3680043.8.498.37833587429731221455928642963031995680",
    "1.2.826.0.1.3680043.8.498.34685905030370793639196564723935583035",
}


@dataclass(frozen=True)
class PrepConfig:
    """How one study is turned into a fixed (S, H, W) uint8 stack.

    Each slot takes one series of a given plane, preferring a fluid-sensitive (1) or
    non-fluid-sensitive (0) series (-1 = no preference), and samples `n` physically
    ordered slices evenly across `span` of the stack. Slots that have no matching
    series stay zero and are masked out.
    """
    slots: tuple = (
        ("Sagittal", 1, 16),
        ("Sagittal", 0, 12),
        ("Coronal", 1, 12),
        ("Coronal", 0, 8),
        ("Axial", 1, 12),
    )
    span: tuple = (0.06, 0.94)       # outer slices hold collaterals / lateral meniscus; don't cut to 20-80%
    crop_mm: float = 140.0           # physical field of view of the square crop
    img_size: int = 256              # stored resolution; train/infer may resize down
    pct_lo: float = 1.0              # intensity window percentiles, per series
    pct_hi: float = 99.0
    normalize_laterality: bool = True  # mirror right knees so medial is always on the same side
    canonical_orientation: bool = True  # flip slices so rows/cols point the same way for every scanner

    @property
    def n_slices(self) -> int:
        return sum(s[2] for s in self.slots)

    def slot_bounds(self):
        """[(start, end)] slice index range of each slot inside the stack."""
        out, i = [], 0
        for _, _, n in self.slots:
            out.append((i, i + n))
            i += n
        return out


DEFAULT_PREP = PrepConfig()
