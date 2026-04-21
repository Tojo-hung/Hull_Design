"""
hull_core/constraints.py — Tiered constraint checking for hull designs.
========================================================================

Three evaluation tiers, ordered by cost:

Tier 0 — Parameter validation  (< 1 ms, no geometry)
    Pure checks on the HullForm fields.  Catches obviously invalid
    designs before any interpolation or CAD call.

Tier 1 — Approximate geometric checks  (< 50 ms, PCHIP, no CAD)
    Uses fast Python interpolation to pre-screen designs.
    Checks rocker, beam monotonicity, and draft continuity.

Tier 2 — Solid-level checks  (placeholder, requires CAD solid)
    Exact displacement, Cp, and watertight validation from an OCCT solid.
    NOT implemented here — hull_cad will call these once the solid exists.

This module has NO dependency on CadQuery or any geometry code.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    pass  # future: HullSolid for Tier 2 type hints

from hull_core.schema import HullForm


# ──────────────────────────────────────────────────────────────────────
# Primitives
# ──────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class Range:
    """Inclusive numeric range [lo, hi]."""

    lo: float
    hi: float

    def __post_init__(self) -> None:
        if self.lo > self.hi:
            raise ValueError(f"Range lo={self.lo} > hi={self.hi}")

    def contains(self, v: float) -> bool:
        return self.lo <= v <= self.hi

    def clamp(self, v: float) -> float:
        return max(self.lo, min(self.hi, v))

    def __repr__(self) -> str:
        return f"[{self.lo}, {self.hi}]"


@dataclass
class ConstraintResult:
    """Outcome of a constraint check.  Falsy when any violation exists."""

    passed: bool
    violations: list[str] = field(default_factory=list)

    def __bool__(self) -> bool:
        return self.passed

    def __str__(self) -> str:
        if self.passed:
            return "ok"
        return "; ".join(self.violations)

    def merge(self, other: "ConstraintResult") -> "ConstraintResult":
        """Combine two results (AND logic)."""
        return ConstraintResult(
            passed=self.passed and other.passed,
            violations=self.violations + other.violations,
        )


# ──────────────────────────────────────────────────────────────────────
# Constraint set
# ──────────────────────────────────────────────────────────────────────

@dataclass
class HullConstraints:
    """Declarative constraint set for a Moth-class hull.

    Default values match the bounds in ``hull_optimizer.py``.
    """

    # ── Tier 0 / 1: Form coefficient ranges ──────────────────────
    prismatic_coeff: Range = field(
        default_factory=lambda: Range(0.55, 0.85)
    )

    # ── Tier 0: Dimensional limits ───────────────────────────────
    max_canoe_draft: float = 350.0     # mm — Moth class max
    min_freeboard: float = 100.0       # mm — minimum sheer height

    # ── Tier 1: Keel geometry ────────────────────────────────────
    rocker: Range = field(
        default_factory=lambda: Range(60.0, 130.0)  # mm
    )

    # ──────────────────────────────────────────────────────────────
    # Tier 0: Pure parameter validation
    # ──────────────────────────────────────────────────────────────

    def check_parameters(self, form: HullForm) -> ConstraintResult:
        """Tier 0: instant checks on the HullForm fields.

        No interpolation, no CAD.  Runs in < 1 ms.
        """
        violations: list[str] = []

        # Draft floor: every station must be at least as deep as the
        # shallower endpoint (same logic as hull_optimizer.check_constraints).
        draft_floor = form.draft_floor
        for s in form.stations:
            if s.draft < draft_floor - 1e-6:
                violations.append(
                    f"Station x={s.x:.0f}: draft={s.draft:.1f} mm < "
                    f"floor={draft_floor:.1f} mm (max of bow/transom draft)"
                )

        # Max canoe body draft
        all_drafts = (
            [s.draft for s in form.stations]
            + [form.bow.draft, form.transom.draft]
        )
        for d in all_drafts:
            if d > self.max_canoe_draft + 1e-6:
                violations.append(
                    f"draft={d:.1f} mm exceeds class max={self.max_canoe_draft:.1f} mm"
                )

        # Freeboard: sheer heights must meet minimum
        sheer_values = (
            [(f"bow", form.bow.sheer_height)]
            + [(f"x={s.x:.0f}", s.sheer_height) for s in form.stations]
            + [(f"transom", form.transom.sheer_height)]
        )
        for label, sh in sheer_values:
            if sh < self.min_freeboard - 1e-6:
                violations.append(
                    f"{label}: sheer_height={sh:.1f} mm < "
                    f"min_freeboard={self.min_freeboard:.1f} mm"
                )

        # Station x ordering is enforced by HullForm.__post_init__,
        # but check that stations are within hull bounds.
        for s in form.stations:
            if s.x <= 0.0 or s.x >= form.lwl:
                violations.append(
                    f"Station x={s.x} is outside hull (0, {form.lwl})"
                )

        # Half-beam sanity: must be non-negative (already checked in
        # SectionParams, but we catch it here for the transom too).
        if form.transom.half_beam < 0:
            violations.append(
                f"transom half_beam={form.transom.half_beam:.1f} is negative"
            )

        return ConstraintResult(
            passed=len(violations) == 0, violations=violations
        )

    # ──────────────────────────────────────────────────────────────
    # Tier 1: Approximate geometric checks (PCHIP-based)
    # ──────────────────────────────────────────────────────────────

    def check_form(self, form: HullForm) -> ConstraintResult:
        """Tier 1: geometric pre-screening using Python PCHIP.

        ~10–50 ms.  Catches bad designs before calling CadQuery.
        Requires scipy (already a project dependency).
        """
        import numpy as np
        from scipy.interpolate import PchipInterpolator

        violations: list[str] = []

        # Build x / draft arrays including endpoints
        xs = (
            [0.0]
            + [s.x for s in form.stations]
            + [form.lwl]
        )
        drafts = (
            [form.bow.draft]
            + [s.draft for s in form.stations]
            + [form.transom.draft]
        )

        # ── Rocker ───────────────────────────────────────────────
        # Rocker = max rise of the keel above the straight chord
        # joining bow keel to transom keel.
        # (Same definition as hull_optimizer.check_constraints)
        x_arr = np.linspace(0.0, form.lwl, 300)
        d_interp = PchipInterpolator(xs, drafts)(x_arr)
        chord = drafts[0] + (drafts[-1] - drafts[0]) * (x_arr / form.lwl)
        rocker_mm = float(np.max(d_interp - chord))

        if not self.rocker.contains(rocker_mm):
            violations.append(
                f"rocker={rocker_mm:.1f} mm not in {self.rocker}"
            )

        # ── Half-beam continuity ─────────────────────────────────
        # Flag severe local beam reversals (non-monotonic sections
        # between adjacent stations that might cause loft problems).
        hbs = [s.half_beam for s in form.stations]
        if len(hbs) >= 3:
            for i in range(1, len(hbs) - 1):
                if hbs[i] < hbs[i - 1] and hbs[i] < hbs[i + 1]:
                    violations.append(
                        f"Beam reversal at station x="
                        f"{form.stations[i].x:.0f}: "
                        f"hb={hbs[i]:.1f} is a local minimum "
                        f"between {hbs[i-1]:.1f} and {hbs[i+1]:.1f}"
                    )

        return ConstraintResult(
            passed=len(violations) == 0, violations=violations
        )

    # ──────────────────────────────────────────────────────────────
    # Tier 2: CAD-based solid checks (placeholder)
    # ──────────────────────────────────────────────────────────────

    def check_solid(
        self,
        solid: object,
        form: HullForm,
    ) -> ConstraintResult:
        """Tier 2: exact solid checks via OCCT properties.

        **Not yet implemented.**  Returns a passing result unconditionally.
        This will be wired up in hull_cad after the CAD loft is validated.

        Future checks:
          - Displaced volume at design waterline matches target ± tolerance
          - Prismatic coefficient (Cp) from actual CAD solid properties
          - BRepCheck_Analyzer validity
          - No open edges (watertight)
        """
        return ConstraintResult(passed=True, violations=[])
