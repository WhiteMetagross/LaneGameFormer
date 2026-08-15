"""
Generate publication-quality visualizations for the ProjectPrayagBEV Dataset
research paper. Five consolidated multi-panel figures — PNG only, 300 DPI.
Updates: Uses pre-calibrated metric columns (meters for size, km/h for speed),
supports the new 51-chunk contiguous dataset, and outputs for both 30Hz and 10Hz variants.
"""

import json, os, glob, sys
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import matplotlib.gridspec as gridspec
from matplotlib.patches import FancyBboxPatch
from matplotlib.colors import LinearSegmentedColormap
from numpy.polynomial import polynomial as P
import warnings
warnings.filterwarnings('ignore')

# ── Global RC params ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size': 9,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'axes.labelsize': 10,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8,
    'figure.dpi': 300,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.12,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': False,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
})

# ── Palette ───────────────────────────────────────────────────────────────────
C_SPLIT = {'train': '#2196F3', 'val': '#FF9800', 'test': '#E91E63'}
C_TOD   = {'morning': '#42A5F5', 'afternoon': '#FF7043', 'evening': '#AB47BC'}
C_DENS  = {'low': '#66BB6A', 'medium': '#FFA726', 'high': '#EF5350'}
C_CLASS = {'HPE': '#FF8A80', 'SVE': '#42A5F5', 'LVE': '#66BB6A'}

BASE_DIR = Path(__file__).parent.resolve() if '__file__' in locals() else Path(os.getcwd())
OUTPUT_DIR = BASE_DIR / "dataset_paper_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

BASE_30 = BASE_DIR / "ChunkedProjectPrayagBEVDataset"
BASE_10 = BASE_DIR / "ChunkedProjectPrayagBEVDataset10Hz"
BASE_ORIG = BASE_DIR / "ProjectPrayagTopDownDataset"

SCENES_ORDERED = []
SCENE_SHORT = {}
C_SCENE = {}

DEFAULT_SCENE_COLORS = [
    '#7E57C2', '#26A69A', '#EC407A', '#5C6BC0', '#FFA726', '#8D6E63',
    '#42A5F5', '#66BB6A', '#EF5350', '#AB47BC', '#26C6DA', '#9CCC65',
]

CLASS_ORDER = ['HPE', 'SVE', 'LVE']
CLASS_FULL  = {'HPE': 'Human-Pedestrian Entity',
               'SVE': 'Small Vehicle Entity',
               'LVE': 'Large Vehicle Entity'}

def _scene_sort_key(scene_id):
    """Sort scene IDs by numeric suffix, then lexicographically."""
    try:
        return (0, int(scene_id.split('_')[-1]))
    except Exception:
        return (1, str(scene_id))

def configure_scene_palette(scene_ids):
    """Configure scene ordering, short labels, and colours from available scene IDs."""
    global SCENES_ORDERED, SCENE_SHORT, C_SCENE
    SCENES_ORDERED = sorted(set(scene_ids), key=_scene_sort_key)
    SCENE_SHORT = {}
    for s in SCENES_ORDERED:
        sid = s.split('_')[-1]
        if sid.startswith('0'):
            sid = sid[1:]
        SCENE_SHORT[s] = f'S{sid}'
    C_SCENE = {s: DEFAULT_SCENE_COLORS[i % len(DEFAULT_SCENE_COLORS)]
               for i, s in enumerate(SCENES_ORDERED)}

# ── helpers ───────────────────────────────────────────────────────────────────

def _save(fig, name, output_suffix=''):
    try:
        fig.tight_layout()
    except Exception:
        pass
    save_path = OUTPUT_DIR / f'{name}{output_suffix}.png'
    fig.savefig(save_path)
    plt.close(fig)
    print(f'  [+] Saved: {save_path.name}')

def _comma(x, _=None):
    return f'{int(x):,}'

def load_manifest(path):
    with open(os.path.join(path, 'chunk_manifest.json'), 'r') as f:
        return json.load(f)

def flatten_chunks(manifest):
    rows = []
    for sp in ('train', 'val', 'test'):
        for c in manifest['splits'][sp]:
            c['split'] = sp
            rows.append(c)
    return pd.DataFrame(rows)

def load_all_tracks(base_path):
    """Load tracking CSVs — one representative chunk per scene."""
    frames = []
    loaded_scenes = set()
    for split in ('train', 'val', 'test'):
        adir = os.path.join(base_path, split, 'annotations')
        if not os.path.isdir(adir):
            continue
        for csv_path in sorted(glob.glob(os.path.join(adir, '*_tracks.csv'))):
            name = os.path.basename(csv_path)
            scene = '_'.join(name.split('_')[:2])
            if scene in loaded_scenes:
                continue
            loaded_scenes.add(scene)
            try:
                df = pd.read_csv(csv_path)
                df['_scene'] = scene
                frames.append(df)
            except Exception as e:
                print(f"Error loading {csv_path}: {e}")
                continue
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def load_original_tracks_with_metrics(is_10hz=False):
    """
    Load all tracks from ProjectPrayagTopDownDataset/unified_tracking_data.csv,
    downsample if is_10hz is True, and dynamically project pixel coordinates to physical meters.
    """
    csv_path = os.path.join(BASE_ORIG, 'unified_tracking_data.csv')
    if not os.path.exists(csv_path):
        print(f"  [!] {csv_path} not found!")
        return pd.DataFrame()
        
    print(f"  Loading original tracking data from {csv_path} ...")
    usecols = [
        'scene_id', 'frame_id', 'track_id', 'class_name',
        'center_x', 'center_y',
        'obb_corner1_x', 'obb_corner1_y',
        'obb_corner2_x', 'obb_corner2_y',
        'obb_corner3_x', 'obb_corner3_y',
        'obb_corner4_x', 'obb_corner4_y'
    ]
    df = pd.read_csv(csv_path, usecols=usecols)
    
    if is_10hz:
        df = df[(df['frame_id'] - 1) % 3 == 0].copy()
        df['frame_id'] = (df['frame_id'] - 1) // 3 + 1
        
    metadata_dir = os.path.join(BASE_ORIG, 'CIRAerialDroneIndianIntersectionsVideoes')
    
    CX, CY = 960.0, 540.0
    SENSOR_WIDTH = 13.2
    IMAGE_WIDTH = 1920.0
    CALIBRATION_FACTOR = 6.55
    REF_LAT, REF_LON = 25.436562, 81.841327
    R_EARTH = 6378137.0
    
    scenes = df['scene_id'].unique()
    frames = []
    
    for scene in scenes:
        sdf = df[df['scene_id'] == scene].copy()
        meta_path = os.path.join(metadata_dir, f"{scene}_metadata.csv")
        
        meta_dict = {}
        if os.path.exists(meta_path):
            df_meta = pd.read_csv(meta_path)
            for _, row in df_meta.iterrows():
                f_id = int(row['frame_id'])
                meta_dict[f_id] = {
                    "lat": float(row['latitude']),
                    "lon": float(row['longitude']),
                    "alt": float(row['altitude']),
                    "focal": float(row['focal_length']) / 10.0
                }
        
        meta_df_data = []
        for f_id_0based, m in meta_dict.items():
            meta_df_data.append({
                'frame_id_0based': f_id_0based,
                'alt': m['alt'],
                'focal': m['focal'],
                'lat': m['lat'],
                'lon': m['lon']
            })
            
        if meta_df_data:
            df_meta_lookup = pd.DataFrame(meta_df_data)
        else:
            df_meta_lookup = pd.DataFrame(columns=['frame_id_0based', 'alt', 'focal', 'lat', 'lon'])
            
        sdf['frame_id_0based'] = sdf['frame_id'] - 1
        if is_10hz:
            sdf['frame_id_0based'] = (sdf['frame_id'] - 1) * 3
            
        sdf_merged = pd.merge(sdf, df_meta_lookup, on='frame_id_0based', how='left')
        
        sdf_merged['alt'] = sdf_merged['alt'].fillna(80.0)
        sdf_merged['focal'] = sdf_merged['focal'].fillna(31.9)
        sdf_merged['lat'] = sdf_merged['lat'].fillna(REF_LAT)
        sdf_merged['lon'] = sdf_merged['lon'].fillna(REF_LON)
        
        sdf_merged['scale'] = np.where(
            sdf_merged['focal'] > 0,
            (sdf_merged['alt'] * SENSOR_WIDTH) / (sdf_merged['focal'] * IMAGE_WIDTH),
            0.05
        ) * CALIBRATION_FACTOR
        scale = sdf_merged['scale']
        
        px_dx = sdf_merged['center_x'] - CX
        px_dy = CY - sdf_merged['center_y']
        
        sdf_merged['local_center_x_m'] = px_dx * scale
        sdf_merged['local_center_y_m'] = px_dy * scale
        
        lat_rad = np.radians(sdf_merged['lat'])
        lon_rad = np.radians(sdf_merged['lon'])
        ref_lat_rad = np.radians(REF_LAT)
        ref_lon_rad = np.radians(REF_LON)
        
        drone_dy = R_EARTH * (lat_rad - ref_lat_rad)
        drone_dx = R_EARTH * (lon_rad - ref_lon_rad) * np.cos(ref_lat_rad)
        
        sdf_merged['world_center_x_m'] = drone_dx + sdf_merged['local_center_x_m']
        sdf_merged['world_center_y_m'] = drone_dy + sdf_merged['local_center_y_m']
        
        for i in range(1, 5):
            pt_dx = sdf_merged[f'obb_corner{i}_x'] - CX
            pt_dy = CY - sdf_merged[f'obb_corner{i}_y']
            
            pt_local_x = pt_dx * scale
            pt_local_y = pt_dy * scale
            
            sdf_merged[f'local_obb_corner{i}_x_m'] = pt_local_x
            sdf_merged[f'local_obb_corner{i}_y_m'] = pt_local_y
            sdf_merged[f'world_obb_corner{i}_x_m'] = drone_dx + pt_local_x
            sdf_merged[f'world_obb_corner{i}_y_m'] = drone_dy + pt_local_y
            
        sdf_merged['_scene'] = scene
        frames.append(sdf_merged)
        
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

def load_original_class_stats():
    """Load class distribution from the original non-split dataset."""
    csv_path = os.path.join(BASE_ORIG, 'unified_tracking_data.csv')
    if not os.path.exists(csv_path):
        return None
    header_cols = pd.read_csv(csv_path, nrows=0).columns.tolist()

    base_cols = ['scene_id', 'track_id', 'frame_id']
    if not all(c in header_cols for c in base_cols):
        return None

    if 'class_name' in header_cols:
        return pd.read_csv(csv_path, usecols=['scene_id', 'class_name', 'track_id', 'frame_id'])

    if 'class' in header_cols:
        df = pd.read_csv(csv_path, usecols=['scene_id', 'class', 'track_id', 'frame_id'])
        class_map = {0: 'HPE', 1: 'LVE', 2: 'SVE'}
        df['class_name'] = df['class'].map(class_map).fillna('SVE')
        return df[['scene_id', 'class_name', 'track_id', 'frame_id']]

    return None

def _add_bbox_cols(tdf):
    """Add bb_length_m, bb_width_m, bb_area_m using pre-calibrated OBB corners in meters."""
    # Check if metric columns exist, else fallback to standard pixel computation (divided by typical scale for plotting)
    if 'local_obb_corner1_x_m' in tdf.columns:
        tdf['bb_w_m'] = np.sqrt((tdf['local_obb_corner2_x_m'] - tdf['local_obb_corner1_x_m'])**2 +
                               (tdf['local_obb_corner2_y_m'] - tdf['local_obb_corner1_y_m'])**2)
        tdf['bb_h_m'] = np.sqrt((tdf['local_obb_corner3_x_m'] - tdf['local_obb_corner2_x_m'])**2 +
                               (tdf['local_obb_corner3_y_m'] - tdf['local_obb_corner2_y_m'])**2)
        tdf['bb_length_m'] = tdf[['bb_w_m', 'bb_h_m']].max(axis=1)
        tdf['bb_width_m']  = tdf[['bb_w_m', 'bb_h_m']].min(axis=1)
    else:
        # Fallback to dynamic approximation (pixels * 0.113m/px)
        tdf['bb_w'] = np.sqrt((tdf['obb_corner2_x'] - tdf['obb_corner1_x'])**2 +
                               (tdf['obb_corner2_y'] - tdf['obb_corner1_y'])**2)
        tdf['bb_h'] = np.sqrt((tdf['obb_corner3_x'] - tdf['obb_corner2_x'])**2 +
                               (tdf['obb_corner3_y'] - tdf['obb_corner2_y'])**2)
        tdf['bb_length_m'] = tdf[['bb_w', 'bb_h']].max(axis=1) * 0.113
        tdf['bb_width_m']  = tdf[['bb_w', 'bb_h']].min(axis=1) * 0.113
    
    tdf['bb_area_m']   = tdf['bb_length_m'] * tdf['bb_width_m']
    return tdf

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — Dataset Structure & Splitting Strategy
# ══════════════════════════════════════════════════════════════════════════════
def figure1(df30, output_suffix=''):
    fig = plt.figure(figsize=(20, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.52, wspace=0.35)
    scenes = SCENES_ORDERED

    # ── (a) Density heatmap (scene × chunk) ──────────────────────────────
    ax = fig.add_subplot(gs[0, :])  # full top row
    max_ci = int(df30['chunk_index'].max()) + 1
    heat = np.full((len(scenes), max_ci), np.nan)
    for _, r in df30.iterrows():
        if r['original_scene_id'] not in scenes:
            continue
        si = scenes.index(r['original_scene_id'])
        heat[si, int(r['chunk_index'])] = r['avg_vehicles_per_frame']

    cmap = LinearSegmentedColormap.from_list(
        'dens', ['#E3F2FD', '#42A5F5', '#7E57C2', '#FF7043', '#E91E63'], N=256)
    im = ax.imshow(heat, cmap=cmap, aspect='auto', interpolation='nearest')
    cbar = fig.colorbar(im, ax=ax, shrink=0.65, pad=0.015)
    cbar.set_label('Average vehicles / frame', fontsize=9)

    for i in range(heat.shape[0]):
        for j in range(heat.shape[1]):
            v = heat[i, j]
            if not np.isnan(v):
                tc = 'white' if v > 55 else 'black'
                ax.text(j, i, f'{v:.0f}', ha='center', va='center',
                    fontsize=6.5, fontweight='bold', color=tc)

    split_border = {'train': '#00E676', 'val': '#FFD600', 'test': '#7B1FA2'}
    for _, r in df30.iterrows():
        if r['original_scene_id'] not in scenes:
            continue
        si = scenes.index(r['original_scene_id'])
        ci = int(r['chunk_index'])
        rect = plt.Rectangle((ci - .5, si - .5), 1, 1, fill=False,
                              edgecolor=split_border[r['split']], linewidth=2.5)
        ax.add_patch(rect)

    ax.set_yticks(range(len(scenes)))
    ax.set_yticklabels([SCENE_SHORT[s] for s in scenes])
    ax.set_xlabel('Chunk Index')
    ax.set_title('(a) Vehicle Density Heatmap: All Chunks (border colour = split)', pad=10)
    leg = [mpatches.Patch(fc='none', ec=split_border[s], lw=2.5, label=s.capitalize())
           for s in ('train', 'val', 'test')]
    ax.legend(handles=leg, loc='lower right', framealpha=.9, fontsize=8)

    # ── (b) Split distribution — grouped bars (chunks / frames / tracks) ─
    ax = fig.add_subplot(gs[1, 0])
    splits = ['train', 'val', 'test']
    x = np.arange(3)
    w = 0.20

    chunks  = np.array([len(df30[df30['split'] == s]) for s in splits])
    frames  = np.array([df30[df30['split'] == s]['num_frames'].sum() for s in splits])
    tracks  = np.array([df30[df30['split'] == s]['total_unique_tracks'].sum() for s in splits])

    for metric, label, offset, color in [
            (chunks,  'Chunks',  -0.29, '#7E57C2'),
            (frames,  'Frames',   0.0,   '#26A69A'),
            (tracks,  'Tracks',   0.29,  '#EC407A')]:
        pct = metric / metric.sum() * 100
        bars = ax.bar(x + offset, pct, w, label=label, color=color,
                       edgecolor='white', linewidth=1.2, zorder=3)
        for bar, val, raw in zip(bars, pct, metric):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1.2,
                    f'{val:.0f}%\n({_comma(raw)})', ha='center', va='bottom',
                    fontsize=6.5, fontweight='bold', linespacing=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels(['Train', 'Val', 'Test'])
    ax.set_ylabel('Percentage (%)')
    ax.set_ylim(0, 105)
    ax.set_title('(b) Split Composition', pad=10)
    ax.legend(framealpha=.9, fontsize=8)

    # ── (c) Per-scene summary (stacked: frames coloured by split) ────────
    ax = fig.add_subplot(gs[1, 1])
    scenes = SCENES_ORDERED
    x = np.arange(len(scenes))
    bottoms = np.zeros(len(scenes))

    for sp, clr in [('train', C_SPLIT['train']), ('val', C_SPLIT['val']),
                     ('test', C_SPLIT['test'])]:
        vals = []
        for sc in scenes:
            m = (df30['original_scene_id'] == sc) & (df30['split'] == sp)
            vals.append(df30.loc[m, 'num_frames'].sum())
        vals = np.array(vals, dtype=float)
        ax.bar(x, vals, 0.6, bottom=bottoms, color=clr, edgecolor='white',
               linewidth=1.2, label=sp.capitalize(), zorder=3)
        bottoms += vals

    for i, sc in enumerate(scenes):
        total_tracks = df30.loc[df30['original_scene_id'] == sc, 'total_unique_tracks'].sum()
        ax.text(i, bottoms[i] + 150,
                f'{_comma(bottoms[i])} frames\n{_comma(total_tracks)} tracks',
            ha='center', va='bottom', fontsize=6, fontweight='bold', linespacing=0.9)

    ax.set_xticks(x)
    ax.set_xticklabels([SCENE_SHORT[s] for s in scenes])
    ax.set_ylabel('Total Frames')
    ax.set_ylim(0, bottoms.max() * 1.25)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_comma))
    ax.set_title('(c) Per-Scene Frame & Track Totals', pad=10)
    ax.legend(framealpha=.9, fontsize=8)

    # ── (d) Time-of-Day Stratified Split ─────────────────────────────────
    ax = fig.add_subplot(gs[1, 2])
    tod_order = sorted(df30['time_of_day'].unique())
    splits = ['train', 'val', 'test']
    x = np.arange(len(tod_order))
    w = 0.22

    for k, sp in enumerate(splits):
        chunk_vals = []
        frame_vals = []
        for tod in tod_order:
            m = (df30['split'] == sp) & (df30['time_of_day'] == tod)
            chunk_vals.append(m.sum())
            frame_vals.append(df30.loc[m, 'num_frames'].sum())
        bars = ax.bar(x + (k - 1) * w, chunk_vals, w, label=sp.capitalize(),
                      color=C_SPLIT[sp], edgecolor='white', linewidth=1.2, zorder=3)
        for bar, cv, fv in zip(bars, chunk_vals, frame_vals):
            if cv > 0:
                x_text = bar.get_x() + bar.get_width() / 2
                x_offset_pts = -16 if sp == 'val' else 0
                ax.annotate(
                    f'{cv}\n({_comma(fv)} fr)',
                    xy=(x_text, bar.get_height() + 0.4),
                    xytext=(x_offset_pts, 0),
                    textcoords='offset points',
                    ha='center', va='bottom', fontsize=6, fontweight='bold',
                    linespacing=0.9
                )

    tod_desc = {'morning': '06:00–11:59', 'afternoon': '12:00–16:59', 'evening': '17:00–05:59'}
    tod_display = [f'{t.capitalize()}\n({tod_desc.get(t, "")})' for t in tod_order]
    ax.set_xticks(x)
    ax.set_xticklabels(tod_display, fontsize=7.5)
    ax.set_ylabel('Number of Chunks')
    ax.set_title('(d) Time of Day Stratified Split', pad=10)
    ax.legend(framealpha=.9, fontsize=7.5)
    for i, tod in enumerate(tod_order):
        clr = C_TOD.get(tod, '#ccc')
        ax.axvspan(i - 0.45, i + 0.45, color=clr, alpha=0.07, zorder=0)

    _save(fig, 'fig1_dataset_overview', output_suffix)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — Traffic Density violin plots
# ══════════════════════════════════════════════════════════════════════════════
def figure2(df30, output_suffix=''):
    fig = plt.figure(figsize=(15, 6.5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.32)
    scenes = SCENES_ORDERED

    # ── (a) Density distribution per scene — violin/box ──────────────────
    ax = fig.add_subplot(gs[0, 0])
    scene_data = [df30.loc[df30['original_scene_id'] == s, 'avg_vehicles_per_frame'].values
                  for s in scenes]
    parts = ax.violinplot(scene_data, positions=range(len(scenes)),
                          showmeans=False, showmedians=False, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(C_SCENE[scenes[i]])
        pc.set_alpha(0.45)
        pc.set_edgecolor(C_SCENE[scenes[i]])
        pc.set_linewidth(1.5)

    bp = ax.boxplot(scene_data, positions=range(len(scenes)), widths=0.22,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='white', linewidth=1.8),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2))
    for patch, sc in zip(bp['boxes'], scenes):
        patch.set_facecolor(C_SCENE[sc])
        patch.set_edgecolor('white')
        patch.set_linewidth(1.2)

    for i, (data, sc) in enumerate(zip(scene_data, scenes)):
        jit = np.random.RandomState(42).normal(0, 0.06, len(data))
        ax.scatter(np.full_like(data, i) + jit, data,
                   c=C_SCENE[sc], s=22, alpha=0.55, edgecolors='white',
                   linewidths=.4, zorder=5)

    ax.axhline(30, color=C_DENS['high'], ls='--', lw=1.3, alpha=.7, label='High (>30)')
    ax.axhline(15, color=C_DENS['medium'], ls='--', lw=1.3, alpha=.7, label='Medium (>15)')
    ax.set_xticks(range(len(scenes)))
    ax.set_xticklabels([SCENE_SHORT[s] for s in scenes])
    ax.set_ylabel('Average Vehicles / Frame')
    ax.set_title('(a) Density Distribution per Scene', pad=10)
    ax.legend(framealpha=.9, fontsize=7.5, loc='upper left')

    # ── (b) Density temporal progression per scene ───────────────────────
    ax = fig.add_subplot(gs[0, 1])
    for sc in scenes:
        sdf = df30[df30['original_scene_id'] == sc].sort_values('chunk_index')
        if len(sdf) <= 1:
            ax.scatter(sdf['chunk_index'], sdf['avg_vehicles_per_frame'],
                       c=C_SCENE[sc], s=50, zorder=5, edgecolors='white',
                       label=f'{SCENE_SHORT[sc]} ({sdf.iloc[0]["time_of_day"][:3]})')
            continue
        marker = 'o' if sdf.iloc[0]['time_of_day'] == 'morning' else 's'
        ax.plot(sdf['chunk_index'], sdf['avg_vehicles_per_frame'],
                f'-{marker}', color=C_SCENE[sc], lw=2, ms=5, alpha=.85,
                label=f'{SCENE_SHORT[sc]} ({sdf.iloc[0]["time_of_day"][:3]})', zorder=4)

    ax.set_xlabel('Chunk Index (temporal order)')
    ax.set_ylabel('Average Vehicles / Frame')
    ax.set_title('(b) Density Progression Over Time', pad=10)
    ax.legend(framealpha=.9, fontsize=7, ncol=1, loc='upper right')
    ax.text(0.5, -0.18, 'mor = Morning (06:00–11:59)  |  aft = Afternoon (12:00–16:59)',
            transform=ax.transAxes, ha='center', fontsize=7.5, fontstyle='italic', color='#555')

    _save(fig, 'fig2_traffic_density', output_suffix)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — Entity Class Analysis (Original Dataset)
# ══════════════════════════════════════════════════════════════════════════════
def figure3_entity(orig_class_df, output_suffix=''):
    if orig_class_df is None or orig_class_df.empty:
        print('  [!] Figure 3 skipped – original dataset not available'); return

    fig = plt.figure(figsize=(15, 6.5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.32)
    scenes = SCENES_ORDERED
    classes = CLASS_ORDER

    # ── (a) Entity class annotation distribution per scene ───────────────
    ax = fig.add_subplot(gs[0, 0])
    x = np.arange(len(scenes))
    w = 0.22
    for k, cls in enumerate(classes):
        vals = [len(orig_class_df[(orig_class_df['scene_id'] == sc) &
                                  (orig_class_df['class_name'] == cls)])
                for sc in scenes]
        bars = ax.bar(x + (k - 1) * w, vals, w, label=CLASS_FULL[cls],
                      color=C_CLASS[cls], edgecolor='white', linewidth=1.2, zorder=3)
        for bar, v in zip(bars, vals):
            if v > 0:
                x_text = bar.get_x() + bar.get_width() / 2
                x_offset_pts = -6 if cls == 'SVE' else 0
                ax.annotate(
                    _comma(v),
                    xy=(x_text, bar.get_height() + 5000),
                    xytext=(x_offset_pts, 0),
                    textcoords='offset points',
                    ha='center', va='bottom', fontsize=6, fontweight='bold',
                    zorder=10
                )
    ax.set_xticks(x)
    ax.set_xticklabels([SCENE_SHORT[s] for s in scenes])
    ax.set_ylabel('Annotations')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_comma))
    ax.set_title('(a) Entity Class Annotation Distribution', pad=10)
    ax.legend(framealpha=.9, fontsize=7.5)

    # ── (b) Unique agents (tracks) per class per scene ───────────────────
    ax = fig.add_subplot(gs[0, 1])
    x = np.arange(len(scenes))
    w = 0.22
    for k, cls in enumerate(classes):
        vals = [orig_class_df[(orig_class_df['scene_id'] == sc) &
                              (orig_class_df['class_name'] == cls)]['track_id'].nunique()
                for sc in scenes]
        bars = ax.bar(x + (k - 1) * w, vals, w, label=CLASS_FULL[cls],
                      color=C_CLASS[cls], edgecolor='white', linewidth=1.2, zorder=3)
        for bar, v in zip(bars, vals):
            if v > 0:
                x_text = bar.get_x() + bar.get_width() / 2
                x_offset_pts = -6 if cls == 'SVE' else 0
                ax.annotate(
                    _comma(v),
                    xy=(x_text, bar.get_height() + 15),
                    xytext=(x_offset_pts, 0),
                    textcoords='offset points',
                    ha='center', va='bottom', fontsize=6, fontweight='bold',
                    zorder=10
                )
    ax.set_xticks(x)
    ax.set_xticklabels([SCENE_SHORT[s] for s in scenes])
    ax.set_ylabel('Unique Agents (Tracks)')
    ax.yaxis.set_major_formatter(plt.FuncFormatter(_comma))
    ax.set_title('(b) Unique Agent Count per Class', pad=10)
    ax.legend(framealpha=.9, fontsize=7.5)

    _save(fig, 'fig3_entity_classes', output_suffix)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 4 — Annotation & Tracking Statistics (METRIC SYSTEM ENABLED!)
# ══════════════════════════════════════════════════════════════════════════════
def figure4_stats(tracks_df, df30, fps=30, output_suffix=''):
    if tracks_df.empty:
        print('  [!] Figure 4 skipped – no track data'); return

    fig = plt.figure(figsize=(19, 12))
    gs = gridspec.GridSpec(2, 3, figure=fig, hspace=0.38, wspace=0.32)

    tdf = _add_bbox_cols(tracks_df.copy())

    # ── (a) Agent Dimensions scatter (METERS) ────────────────────────────
    ax = fig.add_subplot(gs[0, 0])
    samp = tdf.sample(n=min(8000, len(tdf)), random_state=42)
    sc = ax.scatter(samp['bb_width_m'], samp['bb_length_m'],
                    c=samp['bb_length_m'] / samp['bb_width_m'].clip(0.1),
                    cmap='magma', s=5, alpha=0.35, rasterized=True, zorder=3)
    cbar = fig.colorbar(sc, ax=ax, shrink=0.75, pad=0.02)
    cbar.set_label('Aspect ratio', fontsize=8)
    mw, ml = tdf['bb_width_m'].mean(), tdf['bb_length_m'].mean()
    ax.scatter([mw], [ml], c='#00E676', s=100, marker='*', edgecolors='black',
               lw=.8, zorder=10, label=f'μ = ({mw:.2f}m, {ml:.2f}m)')
    ax.set_xlabel('Width (meters)')
    ax.set_ylabel('Length (meters)')
    ax.set_xlim(0, 7)
    ax.set_ylim(0, 12)
    ax.set_title('(a) Agent Dimensions in Meters (calibrated)', pad=10)
    ax.legend(framealpha=.9, fontsize=8, loc='lower right')

    # ── (b) Agent size distributions (METERS) ────────────────────────────
    ax = fig.add_subplot(gs[0, 1])
    bins = np.linspace(0, 10, 45)
    ax.hist(tdf['bb_width_m'],  bins=bins, color='#42A5F5', alpha=.7, edgecolor='white',
            lw=.6, label=f'Width  μ={tdf["bb_width_m"].mean():.2f}m', zorder=3)
    ax.hist(tdf['bb_length_m'], bins=bins, color='#EF5350', alpha=.55, edgecolor='white',
            lw=.6, label=f'Length μ={tdf["bb_length_m"].mean():.2f}m', zorder=4)
    ax.axvline(tdf['bb_width_m'].mean(),  color='#1565C0', ls='--', lw=1.5)
    ax.axvline(tdf['bb_length_m'].mean(), color='#B71C1C', ls='--', lw=1.5)
    ax.set_xlabel('Size (meters)')
    ax.set_ylabel('Count')
    ax.set_title('(b) Agent Size Distributions in Meters', pad=10)
    ax.legend(framealpha=.9, fontsize=7.5)

    # ── (c) Track lifespan per scene ─────────────────────────────────────
    ax = fig.add_subplot(gs[0, 2])
    scene_durations = {}
    for sc, grp in tdf.groupby('_scene'):
        ls = grp.groupby('track_id')['frame_id'].agg(['min', 'max'])
        ls['dur'] = (ls['max'] - ls['min'] + 1) / fps
        scene_durations[sc] = ls['dur'].values

    present_d = [s for s in SCENES_ORDERED if s in scene_durations]
    ddata = [scene_durations[s] for s in present_d]
    bp = ax.boxplot(ddata, patch_artist=True, widths=0.45, showfliers=False,
                    medianprops=dict(color='white', lw=1.8),
                    whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2))
    for patch, sc in zip(bp['boxes'], present_d):
        patch.set_facecolor(C_SCENE[sc]); patch.set_edgecolor('white'); patch.set_lw(1.2)
    ax.set_xticklabels([SCENE_SHORT[s] for s in present_d])
    ax.set_xlabel('Scene')
    ax.set_ylabel('Track Duration (s)')
    ax.set_title('(c) Track Lifespan per Scene', pad=10)
    for i, (s, d) in enumerate(zip(present_d, ddata)):
        median_y = np.median(d) + 0.5
        if s == 'DJI_0916':
            median_y -= 0.35
        ax.text(i + 1, median_y, f'{np.median(d):.1f}s',
                ha='center', fontsize=7, color='black', fontweight='bold',
                rotation=90, va='bottom')

    # ── (d) Average speed per scene (CALIBRATED km/h!) ───────────────────
    ax = fig.add_subplot(gs[1, 0])
    scene_speeds = {}
    
    use_meters = 'local_center_x_m' in tdf.columns
    
    for sc, grp_scene in tdf.groupby('_scene'):
        spds = []
        for tid, grp_t in grp_scene.groupby('track_id'):
            g = grp_t.sort_values('frame_id')
            if len(g) < 3:
                continue
            
            if use_meters:
                dx = np.diff(g['local_center_x_m'].values)
                dy = np.diff(g['local_center_y_m'].values)
                dt = np.diff(g['frame_id'].values).clip(1)
                speed_mps = np.sqrt(dx**2 + dy**2) / dt * fps
                speed_kmh = speed_mps * 3.6
                spds.append(np.mean(speed_kmh))
            else:
                # Fallback to pixels & static conversion scale
                dx = np.diff(g['center_x'].values)
                dy = np.diff(g['center_y'].values)
                dt = np.diff(g['frame_id'].values).clip(1)
                speed_pps = np.sqrt(dx**2 + dy**2) / dt * fps
                speed_kmh = speed_pps * 0.113 * 3.6
                spds.append(np.mean(speed_kmh))
                
        scene_speeds[sc] = np.array(spds)

    present_sp_filtered = []
    sp_data_filtered = []
    for s in SCENES_ORDERED:
        if s in scene_speeds and len(scene_speeds[s]) > 0:
            speeds = scene_speeds[s]
            p98 = np.percentile(speeds, 98)
            filtered = speeds[speeds <= p98]
            if len(filtered) > 0:
                present_sp_filtered.append(s)
                sp_data_filtered.append(filtered)
    present_sp = present_sp_filtered
    sp_data = sp_data_filtered

    parts = ax.violinplot(sp_data, positions=range(len(present_sp)),
                          showmeans=False, showmedians=False, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(C_SCENE[present_sp[i]]); pc.set_alpha(0.45)
        pc.set_edgecolor(C_SCENE[present_sp[i]]); pc.set_lw(1.3)

    bp = ax.boxplot(sp_data, positions=range(len(present_sp)), widths=0.2,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='white', lw=1.8),
                    whiskerprops=dict(linewidth=1.2), capprops=dict(linewidth=1.2))
    for patch, sc in zip(bp['boxes'], present_sp):
        patch.set_facecolor(C_SCENE[sc]); patch.set_edgecolor('white'); patch.set_lw(1.2)

    ax.set_xticks(range(len(present_sp)))
    ax.set_xticklabels([SCENE_SHORT[s] for s in present_sp])
    ax.set_xlabel('Scene')
    ax.set_ylabel('Average Speed (km/h)')
    ax.set_title('(d) Vehicle Speed per Scene (km/h)', pad=10)

    # ── (e) Per-frame vehicle count distribution ─────────────────────────
    ax = fig.add_subplot(gs[1, 1])
    fc = tdf.groupby(['scene_id', 'frame_id']).size()
    bins = np.linspace(0, fc.max(), 45)
    ax.hist(fc.values, bins=bins, color='#5C6BC0', alpha=.8, edgecolor='white', lw=.6, zorder=3)
    ax.axvline(fc.mean(),   color='#FF7043', ls='--', lw=1.5, label=f'Mean {fc.mean():.1f}')
    ax.axvline(fc.median(), color='#66BB6A', ls='--', lw=1.5, label=f'Median {fc.median():.0f}')
    ax.set_xlabel('Vehicles per Frame')
    ax.set_ylabel('Frame Count')
    ax.set_title('(e) Per-Frame Annotation Density', pad=10)
    ax.legend(framealpha=.9, fontsize=8)

    # ── (f) Tracks per chunk vs density (all scenes) ─────────────────────
    ax = fig.add_subplot(gs[1, 2])
    for sc in SCENES_ORDERED:
        sdf = df30[df30['original_scene_id'] == sc]
        ax.scatter(sdf['avg_vehicles_per_frame'], sdf['total_unique_tracks'],
                   c=C_SCENE[sc], s=55, alpha=.7, edgecolors='white', lw=.6,
                   zorder=5, label=SCENE_SHORT[sc])

    xv = df30['avg_vehicles_per_frame'].values
    yv = df30['total_unique_tracks'].values
    coeffs = P.polyfit(xv, yv, 1)
    xfit = np.linspace(xv.min(), xv.max(), 100)
    ax.plot(xfit, P.polyval(xfit, coeffs), '--', color='#555', lw=1.3, alpha=.7)
    r = np.corrcoef(xv, yv)[0, 1]
    ax.text(0.04, 0.95, f'r = {r:.2f}', transform=ax.transAxes,
            fontsize=9, fontweight='bold', va='top',
            bbox=dict(boxstyle='round', fc='white', alpha=.85))
    ax.set_xlabel('Average Vehicles / Frame')
    ax.set_ylabel('Unique Tracks / Chunk')
    ax.set_title('(f) Tracks vs. Density Correlation', pad=10)
    ax.legend(framealpha=.9, fontsize=7.5, ncol=2)

    _save(fig, 'fig4_annotation_statistics', output_suffix)


# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 5 — Spatial Trajectory & Occupancy (METERS!)
# ══════════════════════════════════════════════════════════════════════════════
def figure5_spatial(tracks_df, output_suffix=''):
    if tracks_df.empty:
        print('  [!] Figure 5 skipped – no track data'); return

    present = [s for s in SCENES_ORDERED if s in tracks_df['_scene'].unique()]
    n = len(present)

    fig = plt.figure(figsize=(8 * n, 16))
    gs = gridspec.GridSpec(2, n, figure=fig, hspace=0.15, wspace=0.06,
                           left=0.06)

    heat_cmap = LinearSegmentedColormap.from_list(
        'occ', ['#0D1B2A', '#1B2838', '#7E57C2', '#FF7043', '#FFEB3B'], N=256)
    traj_cmap = plt.cm.turbo

    use_meters = 'local_center_x_m' in tracks_df.columns

    for idx, sc in enumerate(present):
        sdf = tracks_df[tracks_df['_scene'] == sc]
        
        if use_meters:
            x, y = sdf['local_center_x_m'].values, sdf['local_center_y_m'].values
        else:
            x, y = sdf['center_x'].values, sdf['center_y'].values
            
        xr = (x.min() - 5, x.max() + 5)
        yr = (y.min() - 5, y.max() + 5)

        # -- top: heatmap ------------------------------------------------
        ax = fig.add_subplot(gs[0, idx])
        h, xe, ye = np.histogram2d(x, y, bins=100, range=[xr, yr])
        hlog = np.log1p(h.T)
        im = ax.imshow(hlog, extent=[xr[0], xr[1], yr[1], yr[0]],
                        cmap=heat_cmap, aspect='auto', interpolation='bilinear')
        cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
        cbar.set_label('log(count+1)', fontsize=18)
        cbar.ax.tick_params(labelsize=15.75)
        
        ax.set_xlabel('Local X (meters)' if use_meters else 'X (pixels)', fontsize=20.25)
        if idx == 0:
            ax.set_ylabel('Local Y (meters)' if use_meters else 'Y (pixels)', fontsize=20.25)
        ax.tick_params(labelsize=15.75)
        ax.set_title(f'{SCENE_SHORT[sc]}', fontsize=27, pad=10,
                     color=C_SCENE[sc], fontweight='bold')

        # -- bottom: sample trajectories ---------------------------------
        ax = fig.add_subplot(gs[1, idx])
        grps = sdf.groupby('track_id')
        valid = [t for t, g in grps if len(g) >= 8]
        np.random.seed(42)
        chosen = np.random.choice(valid, min(60, len(valid)), replace=False)
        for j, tid in enumerate(chosen):
            g = grps.get_group(tid).sort_values('frame_id')
            c = traj_cmap(j / max(len(chosen) - 1, 1))
            
            if use_meters:
                ax.plot(g['local_center_x_m'], g['local_center_y_m'], '-', color=c, alpha=.55, lw=1.5, zorder=3)
                ax.scatter(g['local_center_x_m'].iloc[0], g['local_center_y_m'].iloc[0],
                           c=[c], s=18, marker='o', edgecolors='white', lw=.3, zorder=5)
            else:
                ax.plot(g['center_x'], g['center_y'], '-', color=c, alpha=.55, lw=1.5, zorder=3)
                ax.scatter(g['center_x'].iloc[0], g['center_y'].iloc[0],
                           c=[c], s=18, marker='o', edgecolors='white', lw=.3, zorder=5)
                           
        ax.invert_yaxis()
        ax.set_aspect('auto')
        ax.set_xlabel('Local X (meters)' if use_meters else 'X (pixels)', fontsize=20.25)
        if idx == 0:
            ax.set_ylabel('Local Y (meters)' if use_meters else 'Y (pixels)', fontsize=20.25)
        ax.tick_params(labelsize=15.75)
        ax.text(0.03, 0.03, f'{len(chosen)} tracks', transform=ax.transAxes,
                fontsize=18, fontstyle='italic', va='bottom',
                bbox=dict(boxstyle='round', fc='white', alpha=.8))

    fig.text(0.015, 0.72, 'Occupancy\nHeatmap', va='center', ha='center',
             fontsize=27, fontweight='bold', rotation=90, color='#444')
    fig.text(0.015, 0.30, 'Sample\nTrajectories', va='center', ha='center',
             fontsize=27, fontweight='bold', rotation=90, color='#444')

    _save(fig, 'fig5_spatial_patterns', output_suffix)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print('=' * 75)
    print(' ProjectPrayag BEV Dataset — Research-Paper Calibrated Figure Generator')
    print('=' * 75)

    print('[1/3] Loading original dataset class data …')
    orig_class = load_original_class_stats()
    if orig_class is not None:
        print(f'  {len(orig_class):,} records from original dataset')

    datasets = [
        ('30Hz', BASE_30, 30, ''),
        ('10Hz', BASE_10, 10, '_10hz'),
    ]

    print('\n[2/3] Generating figures from chunked datasets …')
    for idx, (label, base_path, fps, suffix) in enumerate(datasets, start=1):
        print(f'\n  Dataset {idx}: {label} ({base_path})')
        if not os.path.exists(os.path.join(base_path, 'chunk_manifest.json')):
            print(f'  [!] Skipping {label} — manifest not found')
            continue

        df_chunks = flatten_chunks(load_manifest(base_path))
        configure_scene_palette(df_chunks['original_scene_id'].dropna().tolist())
        print(f'    Chunks: {len(df_chunks):,} across {len(SCENES_ORDERED)} scenes')

        is_10hz = (suffix == '_10hz')
        if is_10hz:
            figure1(df_chunks, output_suffix=suffix)
        else:
            orig_tracks = load_original_tracks_with_metrics(is_10hz=is_10hz)
            orig_scenes = orig_tracks['_scene'].nunique() if not orig_tracks.empty else 0
            print(f'    Original dataset track rows: {len(orig_tracks):,} across {orig_scenes} scenes')

            figure1(df_chunks, output_suffix=suffix)
            figure2(df_chunks, output_suffix=suffix)
            figure3_entity(orig_class, output_suffix=suffix)
            figure4_stats(orig_tracks, df_chunks, fps=fps, output_suffix=suffix)
            figure5_spatial(orig_tracks, output_suffix=suffix)

    print('\n[3/3] Completed figure generation')
    print(f'\n Done — figures saved to {OUTPUT_DIR}')
    print('  Format: PNG @ 300 DPI (calibrated in physical SI units!)')
    print('=' * 75)


if __name__ == '__main__':
    main()
