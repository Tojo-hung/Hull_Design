"""
Moth Hull Designer — Streamlit Web App
=======================================
Run locally:  streamlit run streamlit_app.py
Deploy:       push to GitHub, connect to Streamlit Community Cloud
"""

import sys, os, json, hashlib
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import pandas as pd
import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import (LWL, MAX_DEPTH, FREEBOARD, TARGET_DISP_L,
                    WL_COLORS, N_WL, SECTION_COLORS, DEFAULTS)
from geometry import (lagrange, cross_section, build_ctrl, beam_eval,
                      displaced_volume, find_disp_waterline,
                      hydrostatic_coefficients, build_3d_mesh,
                      export_stl_bytes)

# ─── Page config ──────────────────────────────────────────────
st.set_page_config(
    page_title='Moth Hull Designer',
    page_icon='⛵',
    layout='wide',
    initial_sidebar_state='expanded',
)

st.title('⛵ Moth Hull Designer')

# ─── Config upload (processed before widgets so session_state is set first) ──
if 'cfg_hash' not in st.session_state:
    st.session_state['cfg_hash']       = None
    st.session_state['editor_version'] = 0

st.sidebar.caption('**Load saved config**')
uploaded_cfg = st.sidebar.file_uploader('Load config (JSON)', type=['json'],
                                         label_visibility='collapsed',
                                         help='Upload a previously saved hull_config.json')
st.sidebar.divider()
if uploaded_cfg is not None:
    raw       = uploaded_cfg.read()
    file_hash = hashlib.md5(raw).hexdigest()
    if file_hash != st.session_state['cfg_hash']:
        loaded = json.loads(raw)
        _KEY_MAP = {
            'transom_half_beam': 't_hb',
            'transom_draft':     't_dft',
            'transom_sheer':     't_shr',
            'transom_keel_w':    't_kw',
        }
        for k, v in loaded.items():
            try:
                sk = _KEY_MAP.get(k, k)
                st.session_state[sk] = float(v)
            except (TypeError, ValueError):
                pass
        st.session_state['cfg_hash']        = file_hash
        st.session_state['editor_version'] += 1
        st.rerun()

# ─── Sidebar — bow & transom ──────────────────────────────────
st.sidebar.header('Bow')
bow_draft = st.sidebar.number_input('Bow draft (mm)',  0.0, float(MAX_DEPTH),
                                     float(DEFAULTS['bow_draft']),  1.0, key='bow_draft')
bow_sheer = st.sidebar.number_input('Bow sheer (mm)',  0.0, 400.0,
                                     float(DEFAULTS['bow_sheer']),  1.0, key='bow_sheer')

st.sidebar.header('Transom')
t_hb  = st.sidebar.number_input('Half-beam (mm)',  0.0,  500.0, float(DEFAULTS['transom_half_beam']), 1.0, key='t_hb')
t_dft = st.sidebar.number_input('Draft (mm)',      0.0,  float(MAX_DEPTH), float(DEFAULTS['transom_draft']),     1.0, key='t_dft')
t_shr = st.sidebar.number_input('Sheer (mm)',      0.0,  400.0, float(DEFAULTS['transom_sheer']),  1.0, key='t_shr')
t_kw  = st.sidebar.number_input('Keel width (mm)', 0.0,  300.0, float(DEFAULTS['transom_keel_w']), 1.0, key='t_kw')

st.sidebar.divider()
st.sidebar.header('Hull Length')
lwl = st.sidebar.number_input(
    'LWL — waterline length (mm)',
    min_value=500.0, max_value=6000.0,
    value=float(st.session_state.get('lwl', LWL)),
    step=10.0, key='lwl',
    help='Moth class rule maximum is 3355 mm')

import geometry as _geom
_geom.LWL = lwl

st.caption(f'LWL = {lwl:.0f} mm  |  Target displacement = {TARGET_DISP_L} L')
st.sidebar.divider()
st.sidebar.caption('Adjust control points in the table above the plots.')

# ─── Control-point table ──────────────────────────────────────
st.subheader('Control Points')

col_cfg = {
    'X (mm)':          st.column_config.NumberColumn('X (mm)',          min_value=10,           max_value=int(lwl)-10,  step=1),
    'Half-beam (mm)':  st.column_config.NumberColumn('Half-beam (mm)',  min_value=0,            max_value=500,          step=1),
    'Draft (mm)':      st.column_config.NumberColumn('Draft (mm)',      min_value=0,            max_value=MAX_DEPTH,    step=1),
    'Deck HB (mm)':    st.column_config.NumberColumn('Deck HB (mm)',    min_value=0,            max_value=500,          step=1),
    'Deck Z (mm)':     st.column_config.NumberColumn('Deck Z (mm)',     min_value=0,            max_value=400,          step=1),
    'Keel W (mm)':     st.column_config.NumberColumn('Keel W (mm)',     min_value=0,            max_value=300,          step=1),
    'Beam Z (mm)':     st.column_config.NumberColumn('Beam Z (mm)',     min_value=-MAX_DEPTH,   max_value=200,          step=1),
}

def _src(key):
    """Return session_state value if present (from upload), else DEFAULTS."""
    return float(st.session_state.get(key, DEFAULTS[key]))

default_rows = []
for i in range(1, 6):
    default_rows.append({
        'Point':          f'Pt {i}',
        'X (mm)':         int(_src(f'p{i}_x')),
        'Half-beam (mm)': int(_src(f'p{i}_hb')),
        'Draft (mm)':     int(_src(f'p{i}_d')),
        'Deck HB (mm)':   int(_src(f'p{i}_dw')),
        'Deck Z (mm)':    int(_src(f'p{i}_dz')),
        'Keel W (mm)':    int(_src(f'p{i}_kw')),
        'Beam Z (mm)':    int(_src(f'p{i}_hz')),
    })

default_df = pd.DataFrame(default_rows).set_index('Point')

edited_df = st.data_editor(
    default_df,
    column_config=col_cfg,
    use_container_width=True,
    num_rows='fixed',
    key=f'ctrl_pts_{st.session_state["editor_version"]}',
)

# ─── Build params dict from table + sidebar ───────────────────
params = dict(DEFAULTS)
params['bow_draft']         = bow_draft
params['bow_sheer']         = bow_sheer
params['transom_half_beam'] = t_hb
params['transom_draft']     = t_dft
params['transom_sheer']     = t_shr
params['transom_keel_w']    = t_kw

for i, row in enumerate(edited_df.itertuples(), 1):
    params[f'p{i}_x']  = float(row._1)   # X (mm)
    params[f'p{i}_hb'] = float(row._2)   # Half-beam
    params[f'p{i}_d']  = float(row._3)   # Draft
    params[f'p{i}_dw'] = float(row._4)   # Deck HB
    params[f'p{i}_dz'] = float(row._5)   # Deck Z
    params[f'p{i}_kw'] = float(row._6)   # Keel W
    params[f'p{i}_hz'] = float(row._7)   # Beam Z

# ─── Geometry ─────────────────────────────────────────────────
ctrl_x, beam_hb, keel_d, deck_hb_c, sheer_z_c, keel_w_c, hb_z_c, _, _, _ = build_ctrl(params)

x_dense  = np.linspace(0.0, float(lwl), 300)
keel_z   = -lagrange(x_dense, ctrl_x, keel_d, clip_min=0.0, clip_max=float(MAX_DEPTH))
sheer_z  = lagrange(x_dense, ctrl_x, sheer_z_c, clip_min=0.0)
beam_arr = beam_eval(x_dense, ctrl_x, beam_hb)

deck_kw  = dict(deck_x=ctrl_x, deck_hb_arr=deck_hb_c,
                sheer_z_arr=sheer_z_c, keel_w_arr=keel_w_c, hb_z_arr=hb_z_c)
total_vol = displaced_volume(float(np.max(sheer_z_c)), ctrl_x, beam_hb,
                              ctrl_x, keel_d, **deck_kw)
dz_wl    = find_disp_waterline(TARGET_DISP_L, ctrl_x, beam_hb,
                                ctrl_x, keel_d, **deck_kw)

# ─── Metrics row ──────────────────────────────────────────────
m1, m2, m3, m4 = st.columns(4)
m1.metric('Max beam',        f'{2*float(np.max(beam_arr)):.0f} mm')
m2.metric('Max depth',       f'{float(np.max(-keel_z)):.0f} mm')
m3.metric('Total hull vol',  f'{total_vol:.1f} L')
m4.metric(f'{TARGET_DISP_L} L waterline z', f'{dz_wl:.1f} mm')

# ─── Plots ────────────────────────────────────────────────────
BG   = '#0d1525'
GRID = 'rgba(30,45,70,0.6)'

ctrl_pts_x = [float(params[f'p{i}_x']) for i in range(1, 6)]

fig = make_subplots(
    rows=2, cols=2,
    subplot_titles=('Profile (side view)', 'Body Plan (cross-sections)',
                    'Plan View (waterlines)', ''),
    column_widths=[0.6, 0.4],
    row_heights=[0.5, 0.5],
    horizontal_spacing=0.08,
    vertical_spacing=0.12,
)

# ── Profile ───────────────────────────────────────────────────
fig.add_trace(go.Scatter(
    x=np.concatenate([x_dense, x_dense[::-1]]),
    y=np.concatenate([keel_z, sheer_z[::-1]]),
    fill='toself', fillcolor='rgba(30,144,255,0.10)',
    line=dict(width=0), showlegend=False,
), row=1, col=1)
fig.add_trace(go.Scatter(x=x_dense, y=keel_z,
    line=dict(color='#ff6b6b', width=2), name='Keel',  showlegend=True), row=1, col=1)
fig.add_trace(go.Scatter(x=x_dense, y=sheer_z,
    line=dict(color='#00d4aa', width=2), name='Sheer', showlegend=True), row=1, col=1)
fig.add_hline(y=0,     line=dict(color='#4488cc', width=1, dash='dash'), row=1, col=1)
fig.add_hline(y=dz_wl, line=dict(color='#ffd700', width=1.5, dash='dot'), row=1, col=1)
# Transom line
fig.add_trace(go.Scatter(
    x=[float(lwl), float(lwl)], y=[-float(t_dft), float(t_shr)],
    line=dict(color='#ffaa33', width=2.5), name='Transom', showlegend=False,
), row=1, col=1)
# Bow stem: vertical line at x=0 from bow_draft (keel) to bow_sheer
fig.add_trace(go.Scatter(
    x=[0.0, 0.0], y=[-float(bow_draft), float(bow_sheer)],
    line=dict(color='#00d4aa', width=2), name='Bow stem', showlegend=False,
), row=1, col=1)
ctrl_kz = [-lagrange([xi], ctrl_x, keel_d, clip_min=0.0)[0] for xi in ctrl_pts_x]
fig.add_trace(go.Scatter(
    x=ctrl_pts_x, y=ctrl_kz,
    mode='markers+text',
    marker=dict(color='#ffaa33', size=9),
    text=[f'P{i}' for i in range(1, 6)],
    textposition='bottom center',
    textfont=dict(color='#ffaa33', size=9),
    name='Control pts', showlegend=False,
), row=1, col=1)

# ── Body plan ─────────────────────────────────────────────────
_n_body   = 5
_body_cols = [SECTION_COLORS[int(i * (len(SECTION_COLORS) - 1) / (_n_body - 1))]
               for i in range(_n_body)]
for idx, (xi, col) in enumerate(zip(ctrl_pts_x, _body_cols)):
    hb_    = float(beam_eval([xi], ctrl_x, beam_hb)[0])
    depth_ = float(lagrange([xi], ctrl_x, keel_d, clip_min=0.0, clip_max=float(MAX_DEPTH))[0])
    dhb_   = float(lagrange([xi], ctrl_x, deck_hb_c,  clip_min=0.0)[0])
    dz_    = float(lagrange([xi], ctrl_x, sheer_z_c,  clip_min=0.0)[0])
    kw_    = float(lagrange([xi], ctrl_x, keel_w_c,   clip_min=0.0)[0])
    hz_    = float(lagrange([xi], ctrl_x, hb_z_c)[0])
    ys, zs = cross_section(hb_, depth_, dhb_, dz_, kw_, 60, hb_z=hz_)
    y_full = np.concatenate([-ys[::-1], ys[1:]])
    z_full = np.concatenate([zs[::-1],  zs[1:]])
    side = 1 if xi <= float(LWL) / 2 else -1
    fig.add_trace(go.Scatter(
        x=side * y_full, y=z_full,
        line=dict(color=col, width=1.8),
        name=f'X={xi:.0f} mm', showlegend=True,
    ), row=1, col=2)

fig.add_vline(x=0,     line=dict(color='#334466', width=1),              row=1, col=2)
fig.add_hline(y=0,     line=dict(color='#4488cc', width=1, dash='dash'), row=1, col=2)
fig.add_hline(y=dz_wl, line=dict(color='#ffd700', width=1.5, dash='dot'), row=1, col=2)

ys_t, zs_t = cross_section(float(t_hb), float(t_dft), float(t_hb), float(t_shr), float(t_kw), 40)
fig.add_trace(go.Scatter(
    x=np.concatenate([-ys_t[::-1], ys_t[1:]]),
    y=np.concatenate([zs_t[::-1],  zs_t[1:]]),
    line=dict(color='#ffaa33', width=2, dash='dot'),
    name='Transom', showlegend=False,
), row=1, col=2)

# ── Plan view ─────────────────────────────────────────────────
x_sample = np.linspace(0.0, float(LWL), 150)
hbs_s    = beam_eval(x_sample, ctrl_x, beam_hb)
depths_s = lagrange(x_sample, ctrl_x, keel_d, clip_min=0.0, clip_max=float(MAX_DEPTH))
dhbs_s   = lagrange(x_sample, ctrl_x, deck_hb_c, clip_min=0.0)
sheers_s = lagrange(x_sample, ctrl_x, sheer_z_c, clip_min=0.0)
kws_s    = lagrange(x_sample, ctrl_x, keel_w_c,  clip_min=0.0)
hzs_s    = lagrange(x_sample, ctrl_x, hb_z_c)

fig.add_trace(go.Scatter(x=x_sample, y= hbs_s,
    line=dict(color='#00d4aa', width=2), showlegend=False), row=2, col=1)
fig.add_trace(go.Scatter(x=x_sample, y=-hbs_s,
    line=dict(color='#00d4aa', width=2), showlegend=False), row=2, col=1)
# Transom line across plan view
fig.add_trace(go.Scatter(
    x=[float(LWL), float(LWL)], y=[-float(t_hb), float(t_hb)],
    line=dict(color='#ffaa33', width=2.5), showlegend=False,
), row=2, col=1)
# Bow point
fig.add_trace(go.Scatter(
    x=[0.0], y=[0.0],
    mode='markers', marker=dict(color='#00d4aa', size=8),
    showlegend=False,
), row=2, col=1)

z_levels = np.linspace(-float(MAX_DEPTH) * 0.9, -float(MAX_DEPTH) * 0.05, N_WL)
for j, z_lev in enumerate(z_levels):
    stbd = []
    for k in range(len(x_sample)):
        ys, zs = cross_section(hbs_s[k], depths_s[k], dhbs_s[k],
                                sheers_s[k], kws_s[k], 50, hb_z=hzs_s[k])
        for m in range(len(zs) - 1):
            if (zs[m] - z_lev) * (zs[m + 1] - z_lev) <= 0:
                f = (z_lev - zs[m]) / (zs[m + 1] - zs[m] + 1e-12)
                stbd.append((float(x_sample[k]), float(ys[m] + f * (ys[m + 1] - ys[m]))))
                break
    if not stbd:
        continue
    xs_w, ys_w = zip(*stbd)
    fig.add_trace(go.Scatter(x=xs_w, y= list(ys_w),
        line=dict(color=WL_COLORS[j], width=1), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=xs_w, y=[-y for y in ys_w],
        line=dict(color=WL_COLORS[j], width=1), showlegend=False), row=2, col=1)

aw_cols   = ['#ff9955', '#ff6622', '#ff2200']
max_shr   = float(np.max(sheer_z_c))
aw_levels = np.linspace(max_shr * 0.2, max_shr * 0.9, 3)
for j, z_lev in enumerate(aw_levels):
    stbd = []
    for k in range(len(x_sample)):
        ys, zs = cross_section(hbs_s[k], depths_s[k], dhbs_s[k],
                                sheers_s[k], kws_s[k], 50, hb_z=hzs_s[k])
        for m in range(len(zs) - 1):
            if (zs[m] - z_lev) * (zs[m + 1] - z_lev) <= 0:
                f = (z_lev - zs[m]) / (zs[m + 1] - zs[m] + 1e-12)
                stbd.append((float(x_sample[k]), float(ys[m] + f * (ys[m + 1] - ys[m]))))
                break
    if not stbd:
        continue
    xs_w, ys_w = zip(*stbd)
    fig.add_trace(go.Scatter(x=xs_w, y= list(ys_w),
        line=dict(color=aw_cols[j], width=1), showlegend=False), row=2, col=1)
    fig.add_trace(go.Scatter(x=xs_w, y=[-y for y in ys_w],
        line=dict(color=aw_cols[j], width=1), showlegend=False), row=2, col=1)

fig.add_trace(go.Scatter(
    x=ctrl_pts_x,
    y=[float(params[f'p{i}_hb']) for i in range(1, 6)],
    mode='markers+text',
    marker=dict(color='#ffaa33', size=9),
    text=[f'P{i}' for i in range(1, 6)],
    textposition='top center',
    textfont=dict(color='#ffaa33', size=10),
    showlegend=False,
), row=2, col=1)

# ── Style ─────────────────────────────────────────────────────
fig.update_layout(
    height=700,
    paper_bgcolor=BG,
    plot_bgcolor=BG,
    font=dict(color='#c8d6e5'),
    margin=dict(l=10, r=10, t=40, b=10),
    legend=dict(bgcolor='rgba(13,21,37,0.8)', bordercolor='#1e2d45',
                borderwidth=1, font=dict(size=10)),
)
axis_style = dict(gridcolor=GRID, zerolinecolor=GRID, showgrid=True)
fig.update_xaxes(**axis_style)
fig.update_yaxes(**axis_style)
fig.update_yaxes(scaleanchor='x',  scaleratio=1, row=1, col=1)
fig.update_yaxes(scaleanchor='x2', scaleratio=1, row=1, col=2)
fig.update_yaxes(scaleanchor='x3', scaleratio=1, row=2, col=1)

fig.update_xaxes(title_text='X from bow (mm)', row=1, col=1)
fig.update_yaxes(title_text='Z (mm)',           row=1, col=1)
fig.update_xaxes(title_text='Y (mm)',           row=1, col=2)
fig.update_yaxes(title_text='Z (mm)',           row=1, col=2)
fig.update_xaxes(title_text='X from bow (mm)', row=2, col=1)
fig.update_yaxes(title_text='Y (mm)',           row=2, col=1)

st.plotly_chart(fig, use_container_width=True)

# ─── 3D View ──────────────────────────────────────────────────
st.divider()
with st.expander('3D Hull View', expanded=False):
    with st.spinner('Building 3D mesh...'):
        verts, faces = build_3d_mesh(params, N_X=60, N_T=32)

    x_v = verts[:, 0].tolist()
    y_v = verts[:, 1].tolist()
    z_v = verts[:, 2].tolist()
    i_f = faces[:, 0].tolist()
    j_f = faces[:, 1].tolist()
    k_f = faces[:, 2].tolist()

    fig3d = go.Figure(go.Mesh3d(
        x=x_v, y=y_v, z=z_v,
        i=i_f, j=j_f, k=k_f,
        color='#1e90ff',
        opacity=0.85,
        flatshading=False,
        lighting=dict(ambient=0.4, diffuse=0.8, specular=0.3,
                      roughness=0.5, fresnel=0.2),
        lightposition=dict(x=1000, y=500, z=2000),
    ))
    fig3d.update_layout(
        height=550,
        paper_bgcolor='#0d1525',
        scene=dict(
            bgcolor='#0d1525',
            xaxis=dict(title='X — bow→stern (mm)', color='#7a8fa8',
                       gridcolor='#1e2d45', zerolinecolor='#1e2d45'),
            yaxis=dict(title='Y — port/stbd (mm)',  color='#7a8fa8',
                       gridcolor='#1e2d45', zerolinecolor='#1e2d45'),
            zaxis=dict(title='Z — vertical (mm)',   color='#7a8fa8',
                       gridcolor='#1e2d45', zerolinecolor='#1e2d45'),
            aspectmode='data',
            camera=dict(eye=dict(x=0.8, y=1.6, z=0.6)),
        ),
        margin=dict(l=0, r=0, t=10, b=0),
    )
    st.plotly_chart(fig3d, use_container_width=True)
    st.caption('Drag to rotate  •  Scroll to zoom  •  Shift+drag to pan')

# ─── Export ───────────────────────────────────────────────────
st.divider()
st.subheader('Export')
exp_col1, exp_col2, exp_col3 = st.columns([1, 1, 4])

# Config JSON download (always available — reflects current params)
cfg_json = json.dumps({k: round(float(v), 4) for k, v in params.items()}, indent=2)
exp_col1.download_button(
    label='⬇ Save config (JSON)',
    data=cfg_json,
    file_name='hull_config.json',
    mime='application/json',
)

# STL — generate on demand, then download
with exp_col2:
    if st.button('Generate STL'):
        with st.spinner('Building mesh…'):
            stl_data = export_stl_bytes(params)
        st.session_state['stl_bytes'] = stl_data

if 'stl_bytes' in st.session_state:
    exp_col3.download_button(
        label='⬇ Download hull.stl',
        data=st.session_state['stl_bytes'],
        file_name='moth_hull.stl',
        mime='application/octet-stream',
    )

# ─── Hydrostatic coefficients ─────────────────────────────────
st.divider()
if st.button('Calculate Hull Form Coefficients', type='primary'):
    with st.spinner('Calculating...'):
        h = hydrostatic_coefficients(
            TARGET_DISP_L,
            ctrl_x, beam_hb, ctrl_x, keel_d,
            deck_x=ctrl_x, deck_hb_arr=deck_hb_c,
            sheer_z_arr=sheer_z_c, keel_w_arr=keel_w_c, hb_z_arr=hb_z_c,
        )
    c1, c2, c3, c4 = st.columns(4)
    c1.metric('Block Coeff  Cb',      f"{h['Cb']:.4f}", help='V / (Lwl × Bwl × T)')
    c2.metric('Prismatic Coeff  Cp',  f"{h['Cp']:.4f}", help='V / (Am × Lwl)')
    c3.metric('Midship Coeff  Cm',    f"{h['Cm']:.4f}", help='Am / (Bm × Tm)')
    c4.metric('Waterplane Coeff  Cw', f"{h['Cw']:.4f}", help='Aw / (Lwl × Bwl)')
    st.caption(
        f"Waterline z = {h['z_wl']:.1f} mm  |  "
        f"Draft T = {h['T']:.1f} mm  |  "
        f"Bwl = {h['B_wl']:.1f} mm  |  "
        f"Midship area = {h['A_m']/1e4:.1f} cm²  |  "
        f"Waterplane area = {h['A_wp']/1e6:.4f} m²"
    )
