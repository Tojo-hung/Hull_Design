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
_PT = [
    (  335,   80,   20),   # Pt 1
    ( 1000,  200,   32),   # Pt 2
    ( 1800,  200,   38),   # Pt 3
    ( 2281,  200,   37),   # Pt 4
    ( 2851,  200,   34),   # Pt 5
]
DEFAULTS = dict(transom_half_beam=180.0, transom_draft=25.0,
                transom_sheer=100.0, transom_keel_w=75.0,
                bow_draft=0.0, bow_sheer=100.0)
for _i, (_px, _pb, _pd) in enumerate(_PT, 1):
    DEFAULTS[f'p{_i}_x']  = float(_px)
    DEFAULTS[f'p{_i}_hb'] = float(_pb)
    DEFAULTS[f'p{_i}_d']  = float(_pd)
    DEFAULTS[f'p{_i}_dw'] = [50, 200, 200, 200, 200][_i-1]
    DEFAULTS[f'p{_i}_dz'] = [100, 100, 100, 100, 100][_i-1]
    DEFAULTS[f'p{_i}_kw'] = [100, 100, 100, 100, 100][_i-1]
# ================================================================