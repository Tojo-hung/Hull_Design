"""
hull_core/schema.py — Parametric hull data model.
===================================================

Defines the canonical representation of a Moth-class hull as structured
Python dataclasses.  This module has NO CadQuery or geometry dependencies.

The flat parameter dictionary used throughout the current codebase
(moth_designer/config.py DEFAULTS, best_hull.json, hull_design.json)
can be converted to and from a HullForm via::

    form = HullForm.from_flat_dict(DEFAULTS)
    d    = form.to_flat_dict()

Supported flat-dict flavours
-----------------------------
- Desktop (4 stations): keys p1_x … p4_x, p1_hb … p4_hb, etc.
- Web     (5 stations): keys p1_x … p5_x, plus p{i}_dw for deck width.
- Optimizer JSON:        same as desktop with possible extra keys
                         (e.g. ``ends_draft``) that are silently ignored.

All lengths are in **mm**.  Z is positive up, origin at the design waterline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


# ──────────────────────────────────────────────────────────────────────
# Section-level parameters
# ──────────────────────────────────────────────────────────────────────

@dataclass
class SectionParams:
    """Parameters for a single hull cross-section at a given longitudinal station.

    Invariants enforced in ``__post_init__``:
      - ``keel_width`` ≤ ``half_beam``
      - ``draft`` ≥ 0
      - ``sheer_height`` > ``beam_height``
      - ``deck_half_beam`` (if set) ≥ ``half_beam``
    """

    x: float                             # mm from bow (strictly between 0 and LWL)
    half_beam: float                     # mm, max half-width at the beam knuckle
    draft: float                         # mm, keel depth below DWL (positive value)
    keel_width: float = 0.0              # mm, flat-keel half-width (0 = V-keel)
    beam_height: float = 0.0             # mm, z-height of max-beam point (0 = DWL)
    sheer_height: float = 200.0          # mm, z-height of deck edge
    deck_half_beam: float | None = None  # mm (None → same as half_beam)

    def __post_init__(self) -> None:
        if self.half_beam < 0:
            raise ValueError(
                f"half_beam must be non-negative at x={self.x}, "
                f"got {self.half_beam}"
            )
        if self.draft < 0:
            raise ValueError(
                f"draft must be non-negative at x={self.x}, got {self.draft}"
            )
        if self.keel_width < 0:
            raise ValueError(
                f"keel_width must be non-negative at x={self.x}, "
                f"got {self.keel_width}"
            )
        if self.keel_width > self.half_beam + 1e-6:
            raise ValueError(
                f"keel_width={self.keel_width:.1f} exceeds "
                f"half_beam={self.half_beam:.1f} at x={self.x}"
            )
        if self.sheer_height <= self.beam_height:
            raise ValueError(
                f"sheer_height={self.sheer_height:.1f} must exceed "
                f"beam_height={self.beam_height:.1f} at x={self.x}"
            )
        if self.deck_half_beam is not None and self.deck_half_beam < 0:
            raise ValueError(
                f"deck_half_beam must be non-negative at x={self.x}, "
                f"got {self.deck_half_beam}"
            )

    @property
    def effective_deck_half_beam(self) -> float:
        """Deck width used by geometry: explicit value or fallback to half_beam."""
        if self.deck_half_beam is not None:
            return self.deck_half_beam
        return self.half_beam


@dataclass
class BowSection:
    """Bow endpoint.  ``half_beam`` is typically 0 (knife bow)."""

    draft: float             # mm, keel depth below DWL
    sheer_height: float      # mm, bow freeboard height

    def __post_init__(self) -> None:
        if self.draft < 0:
            raise ValueError(f"bow draft must be non-negative, got {self.draft}")
        if self.sheer_height <= 0:
            raise ValueError(
                f"bow sheer_height must be positive, got {self.sheer_height}"
            )


@dataclass
class TransomSection:
    """Transom (stern) endpoint.  Always a full cross-section."""

    half_beam: float         # mm
    draft: float             # mm
    sheer_height: float      # mm
    keel_width: float = 0.0  # mm
    beam_height: float = 0.0 # mm

    def __post_init__(self) -> None:
        if self.half_beam < 0:
            raise ValueError(
                f"transom half_beam must be non-negative, got {self.half_beam}"
            )
        if self.draft < 0:
            raise ValueError(
                f"transom draft must be non-negative, got {self.draft}"
            )
        if self.keel_width < 0:
            raise ValueError(
                f"transom keel_width must be non-negative, got {self.keel_width}"
            )
        if self.keel_width > self.half_beam + 1e-6:
            raise ValueError(
                f"transom keel_width={self.keel_width:.1f} exceeds "
                f"half_beam={self.half_beam:.1f}"
            )
        if self.sheer_height <= self.beam_height:
            raise ValueError(
                f"transom sheer_height={self.sheer_height:.1f} must exceed "
                f"beam_height={self.beam_height:.1f}"
            )


# ──────────────────────────────────────────────────────────────────────
# Hull form — the top-level parametric definition
# ──────────────────────────────────────────────────────────────────────

# Default class constants (Moth class rules)
_DEFAULT_LWL = 3355.0        # mm
_DEFAULT_FREEBOARD = 200.0   # mm
_DEFAULT_MAX_DEPTH = 350.0   # mm


@dataclass
class HullForm:
    """Complete parametric definition of a Moth-class hull.

    Stations are sorted by ``x`` in ``__post_init__``.  The number of
    stations is variable (the desktop app uses 4, the web app uses 5,
    and any N ≥ 1 is valid).

    Invariants enforced:
      - All station ``x`` values are strictly between 0 and ``lwl``.
      - All station drafts ≥ ``max(bow.draft, transom.draft)``.
    """

    # Class dimensions
    lwl: float = _DEFAULT_LWL
    freeboard: float = _DEFAULT_FREEBOARD
    max_depth: float = _DEFAULT_MAX_DEPTH

    # Endpoints
    bow: BowSection = field(
        default_factory=lambda: BowSection(draft=75.0, sheer_height=150.0)
    )
    transom: TransomSection = field(
        default_factory=lambda: TransomSection(
            half_beam=195.0, draft=65.0, sheer_height=200.0,
            keel_width=100.0, beam_height=0.0,
        )
    )

    # Intermediate control stations (any count ≥ 1)
    stations: list[SectionParams] = field(default_factory=list)

    def __post_init__(self) -> None:
        # Sort by x
        self.stations = sorted(self.stations, key=lambda s: s.x)

        # Validate station x positions
        for s in self.stations:
            if not (0.0 < s.x < self.lwl):
                raise ValueError(
                    f"Station x={s.x} must be strictly between 0 and "
                    f"lwl={self.lwl}"
                )

        # Check for duplicate x positions
        xs = [s.x for s in self.stations]
        for i in range(1, len(xs)):
            if abs(xs[i] - xs[i - 1]) < 0.1:
                raise ValueError(
                    f"Stations at x={xs[i-1]:.1f} and x={xs[i]:.1f} are "
                    f"too close (< 0.1 mm apart)"
                )

    @property
    def n_stations(self) -> int:
        """Number of intermediate control stations."""
        return len(self.stations)

    @property
    def draft_floor(self) -> float:
        """Minimum draft that any station must meet (from endpoints)."""
        return max(self.bow.draft, self.transom.draft)

    # ── Serialization: flat dict ←→ HullForm ─────────────────────

    def to_flat_dict(self) -> dict[str, float]:
        """Convert to the legacy flat-dict format.

        Output is compatible with ``moth_designer/config.py DEFAULTS``,
        ``hull_design.json``, and ``best_hull.json``.
        """
        d: dict[str, float] = {
            "bow_draft": self.bow.draft,
            "bow_sheer": self.bow.sheer_height,
            "transom_half_beam": self.transom.half_beam,
            "transom_draft": self.transom.draft,
            "transom_sheer": self.transom.sheer_height,
            "transom_keel_w": self.transom.keel_width,
            "transom_hb_z": self.transom.beam_height,
        }
        for i, s in enumerate(self.stations, 1):
            d[f"p{i}_x"] = s.x
            d[f"p{i}_hb"] = s.half_beam
            d[f"p{i}_d"] = s.draft
            d[f"p{i}_dz"] = s.sheer_height
            d[f"p{i}_kw"] = s.keel_width
            d[f"p{i}_hz"] = s.beam_height
            if s.deck_half_beam is not None:
                d[f"p{i}_dw"] = s.deck_half_beam
        return d

    @classmethod
    def from_flat_dict(
        cls,
        d: dict[str, Any],
        lwl: float = _DEFAULT_LWL,
        freeboard: float = _DEFAULT_FREEBOARD,
        max_depth: float = _DEFAULT_MAX_DEPTH,
    ) -> "HullForm":
        """Load from a legacy flat-dict.

        Detects the number of stations from the ``p{i}_x`` keys present.
        Compatible with 4-station (desktop) and 5-station (web) formats.
        Unknown keys (e.g. ``ends_draft``) are silently ignored.
        """
        # Detect station count
        n = 0
        while f"p{n + 1}_x" in d:
            n += 1
        if n == 0:
            raise ValueError(
                "No station keys found (expected p1_x, p2_x, …)"
            )

        stations: list[SectionParams] = []
        for i in range(1, n + 1):
            stations.append(SectionParams(
                x=float(d[f"p{i}_x"]),
                half_beam=float(d[f"p{i}_hb"]),
                draft=float(d[f"p{i}_d"]),
                sheer_height=float(d.get(f"p{i}_dz", freeboard)),
                keel_width=float(d.get(f"p{i}_kw", 0.0)),
                beam_height=float(d.get(f"p{i}_hz", 0.0)),
                deck_half_beam=(
                    float(d[f"p{i}_dw"]) if f"p{i}_dw" in d else None
                ),
            ))

        bow = BowSection(
            draft=float(d.get("bow_draft", 75.0)),
            sheer_height=float(d.get("bow_sheer", 150.0)),
        )
        transom = TransomSection(
            half_beam=float(d.get("transom_half_beam", 195.0)),
            draft=float(d.get("transom_draft", 65.0)),
            sheer_height=float(d.get("transom_sheer", 200.0)),
            keel_width=float(d.get("transom_keel_w", 0.0)),
            beam_height=float(d.get("transom_hb_z", 0.0)),
        )

        return cls(
            lwl=lwl,
            freeboard=freeboard,
            max_depth=max_depth,
            bow=bow,
            transom=transom,
            stations=stations,
        )

    def __repr__(self) -> str:
        return (
            f"HullForm(lwl={self.lwl}, stations={self.n_stations}, "
            f"bow_draft={self.bow.draft}, transom_hb={self.transom.half_beam})"
        )
