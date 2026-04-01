# ─────────────────────────────────────────────────────────────
# config.py  —  Fixed constants, defaults, and plot colours
# ─────────────────────────────────────────────────────────────

# ─── Fixed constants ──────────────────────────────────────────
LWL       = 3355   # mm — Moth class rule
MAX_DEPTH = 350    # mm — canoe body draft
FREEBOARD = 100    # mm — sheer height

TARGET_DISP_L = 130  # litres — displacement waterline target

# ─── Plot colours ─────────────────────────────────────────────
WL_COLORS = ['#00eeff', '#33ccff', '#6699ff', '#9966ff', '#cc33ff', '#ff00ee']
N_WL      = len(WL_COLORS)

SECTION_COLORS = [
    '#00e5ff', '#00ccee', '#00aaff', '#1e90ff', '#4477ff',
    '#6655ff', '#8844ee', '#aa33cc', '#cc2299', '#ee1166',
]
N_SEC = len(SECTION_COLORS)

# ================================================================
# Paste this block into config.py to restore a saved design.
# Replace lines  _PT = [  ...  through  DEFAULTS[f'p{_i}_kw'] = ...
# ================================================================
#
# Based on design recommendations:
#   - Max half-beam 240 mm (48 cm) at 76 cm fwd of transom (x=2595)
#   - Max draft at 103 cm fwd of transom (x=2325), pronounced aft rocker
#   - Elliptical/V sections forward, full U-shaped at max section
#   - V-shaped stern (~30% transom area vs main section)
#   - Deck height > 250 mm at midship
#
_PT = [
    (  335,   90,  110),   # Pt 1
    ( 1000,  205,  155),   # Pt 2
    ( 1700,  220,  175),   # Pt 3
    ( 2500,  240,  160),   # Pt 4
]
DEFAULTS = dict(transom_half_beam=200.0, transom_draft=75.0,
                transom_sheer=150.0, transom_keel_w=100.0,
                bow_draft=75.0, bow_sheer=150.0)
for _i, (_px, _pb, _pd) in enumerate(_PT, 1):
    DEFAULTS[f'p{_i}_x']  = float(_px)
    DEFAULTS[f'p{_i}_hb'] = float(_pb)
    DEFAULTS[f'p{_i}_d']  = float(_pd)
    DEFAULTS[f'p{_i}_dz'] = 150.0   # fixed — sheer height not optimised
    DEFAULTS[f'p{_i}_kw'] = [5, 80, 110, 140][_i-1]
    DEFAULTS[f'p{_i}_hz'] = [0, 0, 0, 0][_i-1]
# ================================================================