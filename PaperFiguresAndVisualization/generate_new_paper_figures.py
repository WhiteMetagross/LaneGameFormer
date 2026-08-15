"""
Generate Figure 7, 8, and 9 for the ProjectPrayag BEV Dataset research paper.
Three advanced publication-quality figures representing mixed-traffic behavior characterizations
and camera topological flow overlays in mixed Indian intersections.
"""

import os, json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ── Global RC params ──────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'DejaVu Sans', 'Helvetica'],
    'font.size': 9.5,
    'axes.titlesize': 11,
    'axes.titleweight': 'bold',
    'axes.labelsize': 10,
    'axes.labelweight': 'bold',
    'xtick.labelsize': 8.5,
    'ytick.labelsize': 8.5,
    'legend.fontsize': 8.5,
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
C_DENS = ['#66BB6A', '#FFA726', '#EF5350', '#7E57C2'] # Low, Medium, High, Congested
C_SCENE = [
    '#7E57C2', '#26A69A', '#EC407A', '#5C6BC0', '#FFA726', '#8D6E63'
]

BASE_DIR = Path(__file__).parent.resolve() if '__file__' in locals() else Path(os.getcwd())
BASE_ORIG = BASE_DIR / "ProjectPrayagTopDownDataset"
OUTPUT_DIR = BASE_DIR / "dataset_paper_figures"
os.makedirs(OUTPUT_DIR, exist_ok=True)

SCENES_ORDERED = ['DJI_0910', 'DJI_0911', 'DJI_0912', 'DJI_0914', 'DJI_0915', 'DJI_0916']
SCENE_SHORT = {'DJI_0910': 'S10', 'DJI_0911': 'S11', 'DJI_0912': 'S12', 
               'DJI_0914': 'S14', 'DJI_0915': 'S15', 'DJI_0916': 'S16'}

def _save(fig, name):
    try:
        fig.tight_layout()
    except Exception:
        pass
    save_path = OUTPUT_DIR / f'{name}.png'
    fig.savefig(save_path)
    plt.close(fig)
    print(f'  [+] Saved: {save_path.name}')

def load_and_preprocess_tracks():
    csv_path = BASE_ORIG / "unified_tracking_data.csv"
    if not csv_path.exists():
        print(f"Error: {csv_path} does not exist!")
        return pd.DataFrame()
        
    print(f"Loading tracking data from {csv_path.name} ...")
    df = pd.read_csv(csv_path, usecols=['scene_id', 'frame_id', 'track_id', 'center_x', 'center_y', 'class_name'])
    
    print("Sorting and calculating dynamic speeds...")
    df = df.sort_values(['scene_id', 'track_id', 'frame_id'])
    df['dx'] = df.groupby(['scene_id', 'track_id'])['center_x'].diff()
    df['dy'] = df.groupby(['scene_id', 'track_id'])['center_y'].diff()
    df['dt'] = df.groupby(['scene_id', 'track_id'])['frame_id'].diff().clip(1)
    
    # 0.113 m/px GSD scale conversion factor
    df['speed_kmh'] = np.sqrt(df['dx']**2 + df['dy']**2) / (df['dt'] / 30.0) * 0.113 * 3.6
    
    # Filter out physical outliers
    df = df[df['speed_kmh'] < 95.0]
    df = df.dropna(subset=['speed_kmh'])
    
    print("Mapping active vehicle density...")
    fc = df.groupby(['scene_id', 'frame_id']).size().reset_index(name='density')
    df = pd.merge(df, fc, on=['scene_id', 'frame_id'])
    
    return df

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 7 — Speed Adaptation relative to Surrounding Density
# ══════════════════════════════════════════════════════════════════════════════
def figure7_density_adaptation(df):
    print("Generating Figure 7: Speed Adaptation...")
    fig = plt.figure(figsize=(15, 6.5))
    gs = gridspec.GridSpec(1, 2, figure=fig, wspace=0.32)
    
    # -- Panel (a): violin plots of speed distributions per density category
    ax = fig.add_subplot(gs[0, 0])
    
    # Categorize density
    def get_cat(d):
        if d <= 15: return 'Low (1-15)'
        elif d <= 30: return 'Medium (16-30)'
        elif d <= 45: return 'High (31-45)'
        else: return 'Congested (>45)'
        
    df['density_cat'] = df['density'].apply(get_cat)
    cats = ['Low (1-15)', 'Medium (16-30)', 'High (31-45)', 'Congested (>45)']
    
    cat_data = [df[df['density_cat'] == c]['speed_kmh'].values for c in cats]
    
    parts = ax.violinplot(cat_data, positions=range(4),
                          showmeans=False, showmedians=False, showextrema=False)
    for i, pc in enumerate(parts['bodies']):
        pc.set_facecolor(C_DENS[i])
        pc.set_alpha(0.5)
        pc.set_edgecolor(C_DENS[i])
        pc.set_linewidth(1.5)
        
    bp = ax.boxplot(cat_data, positions=range(4), widths=0.2,
                    patch_artist=True, showfliers=False,
                    medianprops=dict(color='white', linewidth=1.8),
                    whiskerprops=dict(linewidth=1.2),
                    capprops=dict(linewidth=1.2))
    for i, patch in enumerate(bp['boxes']):
        patch.set_facecolor(C_DENS[i])
        patch.set_edgecolor('white')
        patch.set_linewidth(1.2)
        
    ax.set_xticks(range(4))
    ax.set_xticklabels(cats)
    ax.set_xlabel('Surrounding Traffic Density (Active Vehicles / Frame)')
    ax.set_ylabel('Vehicle Speed (km/h)')
    ax.set_title('(a) Velocity Distributions under Varying Traffic Densities', pad=10)
    ax.grid(True, linestyle=':', alpha=0.5)
    
    # -- Panel (b): Frame Average Speed vs Density Correlation
    ax = fig.add_subplot(gs[0, 1])
    frame_avg = df.groupby(['scene_id', 'frame_id'])[['density', 'speed_kmh']].mean().reset_index()
    
    # Sample to keep visual crispness and speed up plot rendering
    samp = frame_avg.sample(n=min(5000, len(frame_avg)), random_state=42)
    
    sc = ax.scatter(samp['density'], samp['speed_kmh'], c='#5C6BC0', s=15, alpha=0.45, edgecolors='none', zorder=3)
    
    # Regression line
    xv = frame_avg['density'].values
    yv = frame_avg['speed_kmh'].values
    coeffs = np.polyfit(xv, yv, 1)
    xfit = np.linspace(xv.min(), xv.max(), 100)
    ax.plot(xfit, np.polyval(coeffs, xfit), '--', color='#D84315', lw=2.0, zorder=5, label='Speed Adaptation Fit')
    
    r = np.corrcoef(xv, yv)[0, 1]
    ax.text(0.02, 0.02, f'r = {r:.2f} (negative adaptation correlation)', transform=ax.transAxes,
            fontsize=9.5, fontweight='bold', va='bottom', ha='left',
            bbox=dict(boxstyle='round', fc='white', ec='#ccc', alpha=0.9))
            
    ax.set_xlabel('Active Vehicles per Frame')
    ax.set_ylabel('Average Traffic Speed (km/h)')
    ax.set_title('(b) Speed Adaptation Correlation Profile', pad=10)
    ax.legend(framealpha=.9, loc='upper right')
    ax.grid(True, linestyle=':', alpha=0.5)
    
    _save(fig, 'fig7_speed_density_adaptation')

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 8 — Interaction Intensity & Safety Risk: Speed vs. VTTC Coupling
# ══════════════════════════════════════════════════════════════════════════════
def figure8_vttc_speed_coupling(df):
    print("Generating Figure 8: Speed vs. VTTC Coupling...")
    sc_path = BASE_ORIG / "interaction_scenarios.json"
    if not sc_path.exists():
        print(f"  [!] interaction_scenarios.json not found. Skipping Figure 8.")
        return
        
    with open(sc_path, 'r') as f:
        scs = json.load(f)
        
    df_sc = pd.DataFrame(scs)
    
    print("Vectorized merging to map tracks speed to interaction safety indicators...")
    track_speeds = df.groupby(['scene_id', 'track_id'])['speed_kmh'].mean().reset_index(name='avg_track_speed')
    df_merged = pd.merge(df_sc, track_speeds.rename(columns={'track_id': 'ego_id'}), on=['scene_id', 'ego_id'], how='inner')
    
    fig = plt.figure(figsize=(12, 8.5))
    ax = fig.add_subplot(1, 1, 1)
    
    # Clip speeds and filter VTTC to realistic limits
    df_merged = df_merged[(df_merged['avg_track_speed'] > 0.5) & (df_merged['mean_vttc'] < 2.0)]
    
    # 2D Hexagonal Binning plot to represent joint density beautifully (publication-grade style)
    hb = ax.hexbin(df_merged['avg_track_speed'], df_merged['mean_vttc'], gridsize=40,
                   cmap='inferno', mincnt=1, edgecolors='none', zorder=3)
    cbar = fig.colorbar(hb, ax=ax, shrink=0.85, pad=0.02)
    cbar.set_label('Joint Scenario Density (Scenario Counts)', fontsize=9.5, fontweight='bold')
    
    # Add tactical annotation quadrants
    # 1. Yielding coordination zone (Low speed, low VTTC)
    rect1 = plt.Rectangle((0, 0), 15, 0.65, fill=False, edgecolor='#66BB6A', lw=2, ls='--', zorder=5)
    ax.add_patch(rect1)
    ax.text(7.5, 0.32, 'Yield Coordination Zone\n(Defensive, Low-Speed)', color='#66BB6A',
            fontsize=8.5, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#66BB6A', alpha=0.9))
            
    # 2. Safety Critical conflict zone (High speed, low VTTC)
    rect2 = plt.Rectangle((32, 0), 28, 0.65, fill=False, edgecolor='#EF5350', lw=2, ls='--', zorder=5)
    ax.add_patch(rect2)
    ax.text(46, 0.32, 'Safety-Critical Conflict\n(High-Speed Risk Exposure)', color='#EF5350',
            fontsize=8.5, fontweight='bold', ha='center', va='center',
            bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#EF5350', alpha=0.9))
            
    # 3. Tactical adaptation corridor
    ax.annotate('Tactical Speed-Adaptation Corridor', xy=(22, 0.75), xytext=(36, 1.2),
                arrowprops=dict(arrowstyle="->", color='#0288D1', lw=2.0, connectionstyle="arc3,rad=-0.15"),
                fontsize=11.5, fontweight='bold', color='#0288D1',
                bbox=dict(boxstyle='round,pad=0.2', fc='white', ec='#0288D1', alpha=0.9))
                
    ax.set_xlabel('Ego Average Speed during Interaction (km/h)')
    ax.set_ylabel('Mean Vector Time-to-Collision (seconds)')
    ax.set_xlim(0, 60)
    ax.set_ylim(0, 2.0)
    ax.set_title('Ego Kinematics vs. Safety Risk: VTTC & Speed Coupling Dynamics', pad=12)
    ax.grid(True, linestyle=':', alpha=0.45)
    
    _save(fig, 'fig8_vttc_speed_coupling')

# ══════════════════════════════════════════════════════════════════════════════
# FIGURE 9 — Trajectory Flow Heatmap over Lane Topological Road Masks
# ══════════════════════════════════════════════════════════════════════════════
def figure9_trajectory_flow_road_mask(df):
    print("Generating Figure 9: Trajectory Flow Road Mask for 3 scenes...")
    scenes = ['DJI_0910', 'DJI_0914', 'DJI_0915']
    scene_names = {
        'DJI_0910': 'DJI_0910 (Irregular T-Intersection)',
        'DJI_0914': 'DJI_0914 (Irregular 4-Way Intersection)',
        'DJI_0915': 'DJI_0915 (Irregular Y-Intersection)'
    }
    
    fig = plt.figure(figsize=(19, 11))
    gs = gridspec.GridSpec(2, 3, figure=fig, wspace=0.15, hspace=0.45)
    
    for col, scene in enumerate(scenes):
        mask_path = BASE_ORIG / "CIRAerialDroneIndianIntersectionsVideoes" / f"{scene}_road_mask.png"
        if not mask_path.exists():
            print(f"  [!] {mask_path.name} not found. Skipping {scene} in Figure 9.")
            continue
            
        road_mask = plt.imread(str(mask_path))
        
        # Generate custom soft-beige/gray vector road canvas
        road_rgb = np.ones((road_mask.shape[0], road_mask.shape[1], 3))
        # Detect which value represents the road surface (usually the smaller pixel set)
        is_road_one = (np.sum(road_mask > 0.5) < np.sum(road_mask <= 0.5))
        if is_road_one:
            road_rgb[road_mask > 0.5] = [0.94, 0.94, 0.95] # Soft light-gray road canvas
        else:
            road_rgb[road_mask <= 0.5] = [0.94, 0.94, 0.95]
            
        sdf = df[df['scene_id'] == scene]
        if sdf.empty:
            print(f"  [!] No tracks found for scene {scene} in trajectories dataset.")
            continue
            
        # -- Panel (Row 1): Trajectory streams colored by velocity
        ax1 = fig.add_subplot(gs[0, col])
        ax1.imshow(road_rgb, extent=[0, 1920, 1080, 0])
        
        print(f"  Plotting continuous trajectory streams for {scene}...")
        grps = sdf.groupby('track_id')
        valid_tids = [t for t, g in grps if len(g) >= 12]
        
        # Select a balanced sample of tracks to keep visualization crisp
        np.random.seed(42)
        sampled_tids = np.random.choice(valid_tids, min(1200, len(valid_tids)), replace=False)
        
        cmap = plt.cm.turbo
        norm = matplotlib.colors.Normalize(vmin=0, vmax=50)
        
        for tid in sampled_tids:
            g = grps.get_group(tid).sort_values('frame_id')
            x = g['center_x'].values
            y = g['center_y'].values
            avg_speed = g['speed_kmh'].mean()
            
            c = cmap(norm(avg_speed))
            ax1.plot(x, y, '-', color=c, alpha=0.18, lw=1.2, zorder=3)
            
        ax1.set_xlim(0, 1920)
        ax1.set_ylim(1080, 0)
        ax1.set_xlabel('Image X (pixels)', fontsize=8.5)
        if col == 0:
            ax1.set_ylabel('Image Y (pixels)', fontsize=8.5)
        else:
            ax1.set_ylabel('')
            
        ax1.set_title(f'Row 1-{col+1}: Trajectory Streams\n{scene_names[scene]}', pad=8, fontsize=10, fontweight='bold')
        
        sm = plt.cm.ScalarMappable(cmap=cmap, norm=norm)
        sm.set_array([])
        cbar1 = fig.colorbar(sm, ax=ax1, shrink=0.80, pad=0.18, orientation='horizontal')
        cbar1.set_label('Average Velocity (km/h)', fontsize=8, fontweight='bold')
        cbar1.ax.tick_params(labelsize=7.5)
        
        # -- Panel (Row 2): Hexbin density heatmap
        ax2 = fig.add_subplot(gs[1, col])
        ax2.imshow(road_rgb, extent=[0, 1920, 1080, 0])
        
        print(f"  Plotting hexbin density map for {scene}...")
        hb = ax2.hexbin(sdf['center_x'], sdf['center_y'], gridsize=80, extent=[0, 1920, 0, 1080],
                       cmap='plasma', mincnt=1, edgecolors='none', zorder=3, alpha=0.88)
        
        ax2.set_xlim(0, 1920)
        ax2.set_ylim(1080, 0)
        ax2.set_xlabel('Image X (pixels)', fontsize=8.5)
        if col == 0:
            ax2.set_ylabel('Image Y (pixels)', fontsize=8.5)
        else:
            ax2.set_ylabel('')
            
        ax2.set_title(f'Row 2-{col+1}: Spatial Occupancy Heatmap\n{scene_names[scene]}', pad=8, fontsize=10, fontweight='bold')
        
        cbar2 = fig.colorbar(hb, ax=ax2, shrink=0.80, pad=0.18, orientation='horizontal')
        cbar2.set_label('Point Density (Observation Counts)', fontsize=8, fontweight='bold')
        cbar2.ax.tick_params(labelsize=7.5)
        
    _save(fig, 'fig9_trajectory_flow_road_mask')


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════
def main():
    print('=' * 75)
    print(' ProjectPrayag BEV Dataset — Advanced Paper Figures Generator')
    print('=' * 75)
    
    df = load_and_preprocess_tracks()
    if df.empty:
        print("Error preprocessing tracking data! Exiting.")
        return
        
    figure7_density_adaptation(df)
    figure8_vttc_speed_coupling(df)
    figure9_trajectory_flow_road_mask(df)
    
    print('\n All advanced figures completed successfully!')
    print(f' Saved to: {OUTPUT_DIR}')
    print('=' * 75)

if __name__ == '__main__':
    main()
