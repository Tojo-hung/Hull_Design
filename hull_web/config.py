# config.py — standalone (no package imports)

LWL       = 3355   # mm — Moth class rule
MAX_DEPTH = 350    # mm — canoe body draft
FREEBOARD = 100    # mm — sheer height above DWL

TARGET_DISP_L = 130  # litres

WL_COLORS = ['#00eeff', '#33ccff', '#6699ff', '#9966ff', '#cc33ff', '#ff00ee']
N_WL      = len(WL_COLORS)

SECTION_COLORS = [
    '#00e5ff', '#00ccee', '#00aaff', '#1e90ff', '#4477ff',
    '#6655ff', '#8844ee', '#aa33cc', '#cc2299', '#ee1166',
]

_PT = [
    (  335,   90,  110),
    ( 1100,  205,  155),
    ( 1700,  250,  175),
    ( 2300,  280,  160),
    ( 2850,  285,  120),
]
DEFAULTS = dict(transom_half_beam=195.0, transom_draft=65.0,
                transom_sheer=200.0, transom_keel_w=100.0,
                bow_draft=70.0, bow_sheer=150.0)
for _i, (_px, _pb, _pd) in enumerate(_PT, 1):
    DEFAULTS[f'p{_i}_x']  = float(_px)
    DEFAULTS[f'p{_i}_hb'] = float(_pb)
    DEFAULTS[f'p{_i}_d']  = float(_pd)
    DEFAULTS[f'p{_i}_dw'] = [90, 205, 255, 280, 290][_i-1]
    DEFAULTS[f'p{_i}_dz'] = [150, 150, 150, 150, 150][_i-1]
    DEFAULTS[f'p{_i}_kw'] = [1, 80, 110, 140, 175][_i-1]
    DEFAULTS[f'p{_i}_hz'] = [120, 120, 0, 0, 0][_i-1]
