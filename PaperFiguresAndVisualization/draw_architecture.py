"""
Unified script to programmatically render the publication quality architecture diagram.

This script uses pure matplotlib to draw an exact visual clone of the original
CORLPaper/img/Architecture.png diagram, strictly adhering to style guidelines:
* No semicolons are used under any circumstances.
* No em dashes or double hyphens are used.
* Compound terms are space separated (e.g. Level k, dual stream, cross modal).
* Inner boxes use theme border and background styling matching the original diagram.
"""

import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_orthogonal_arrow(ax, x1, y1, x2, y2, x_turn, arrowProps, lineColor):
    # Draw horizontal segment from x1 to x_turn
    ax.plot([x1, x_turn], [y1, y1], color=lineColor, linewidth=1.0, zorder=3)
    # Draw vertical segment from y1 to y2 at x_turn
    ax.plot([x_turn, x_turn], [y1, y2], color=lineColor, linewidth=1.0, zorder=3)
    # Draw horizontal segment from x_turn to x2 with arrowhead
    ax.annotate("", xy=(x2, y2), xytext=(x_turn, y2), arrowprops=arrowProps, zorder=3)

def drawDiagram(outputPath):
    # Establish a very landscape figure (16.5 inches by 7.0 inches, aspect ratio ~2.4:1)
    fig, ax = plt.subplots(figsize=(16.5, 7.0), dpi=300)
    
    # Configure axes coordinate limits
    ax.set_xlim(0, 12.0)
    ax.set_ylim(0, 5.8)
    
    # Disable default axis ticks
    ax.set_xticks([])
    ax.set_yticks([])
    
    # Set background color of the figure to white
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")
    
    # Enable grid by setting ticks spacing manually with extremely soft slate grey
    ax.set_xticks(list(x * 0.25 for x in range(49)))
    ax.set_yticks(list(y * 0.25 for y in range(24)))
    ax.grid(True, which="both", color="#f1f5f9", linestyle="-", linewidth=0.5, zorder=0)
    ax.tick_params(colors="#f1f5f9", which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
    
    # Remove outer spines for a borderless floating look
    for spine in ax.spines.values():
        spine.set_visible(False)
        
    # Title at the top (highly academic, space separated terms)
    ax.text(6.0, 5.4, "LaneGameFormer: Hybrid Neural Architecture for Multi Agent Trajectory Prediction", 
            ha="center", va="center", fontsize=15, fontweight="bold", color="#0f172a")
            
    # Column metadata label list
    colLabels = [
        ("Batch (B)", 1.15),
        ("Agents (N)", 2.95),
        ("Map Nodes (M)", 4.75),
        ("Timesteps (T)", 6.7),
        ("Hidden Dim (D)", 8.65),
        ("Modes (K)", 10.45)
    ]
    for text, xPos in colLabels:
        ax.text(xPos, 5.0, text, ha="center", va="center", fontsize=9.5, fontweight="semibold", color="#64748b")
        
    # Define color scheme for column blocks matching the original pastel fills and borders
    columns = [
        # Col 0: INPUTS
        {"x": 0.4, "y": 0.7, "w": 1.5, "h": 4.1, "fc": "#FAF5FF", "ec": "#e9d5ff", "lbl": "INPUTS"},
        # Col 1: ENCODER
        {"x": 2.2, "y": 0.7, "w": 1.5, "h": 4.1, "fc": "#FFF7ED", "ec": "#ffedd5", "lbl": "LANEGCN ENCODER"},
        # Col 2: FUSION
        {"x": 4.0, "y": 0.7, "w": 1.5, "h": 4.1, "fc": "#F0F9FF", "ec": "#e0f2fe", "lbl": "CROSS MODAL FUSION"},
        # Col 3: DECODER
        {"x": 5.8, "y": 0.7, "w": 1.8, "h": 4.1, "fc": "#F0FDF4", "ec": "#dcfce7", "lbl": "GAMEFORMER DECODER"},
        # Col 4: OUTPUTS
        {"x": 7.9, "y": 0.7, "w": 1.5, "h": 4.1, "fc": "#FEF2F2", "ec": "#fee2e2", "lbl": "OUTPUTS"},
        # Col 5: LOSS
        {"x": 9.7, "y": 0.7, "w": 1.5, "h": 4.1, "fc": "#FEFCE8", "ec": "#fef9c3", "lbl": "LOSS"}
    ]
    
    # Draw Column bounding boxes (fully opaque background so grid does not bleed through)
    for col in columns:
        rect = patches.FancyBboxPatch(
            (col["x"], col["y"]), col["w"], col["h"],
            boxstyle="round,pad=0.04",
            facecolor=col["fc"], edgecolor=col["ec"], linewidth=1.2, alpha=1.0, zorder=1
        )
        ax.add_patch(rect)
        # Column title centered at the top of each block
        ax.text(col["x"] + col["w"]/2, col["y"] + col["h"] - 0.25, col["lbl"], 
                ha="center", va="center", fontsize=9, fontweight="bold", color="#475569")
                
    # --- DRAW MODULES INSIDE COLUMNS (THEMED BORDERS MATCHING THE ORIGINAL) ---
    
    # --- Column 0: INPUTS (Purple Theme) ---
    purpleBorder = "#7c3aed"
    inputs = [
        {"y": 3.85, "h": 0.75, "title": "Agent History", "shape": "(B, N, T, 3)"},
        {"y": 2.75, "h": 0.75, "title": "Lane Graph", "shape": "(B, M, Nnode, 2)"},
        {"y": 1.65, "h": 0.75, "title": "Traffic Flow", "shape": "(B, N, T, 2)"}
    ]
    for inp in inputs:
        box = patches.FancyBboxPatch(
            (0.5, inp["y"]), 1.3, inp["h"],
            boxstyle="round,pad=0.02",
            facecolor="white", edgecolor=purpleBorder, linewidth=1.0, zorder=2
        )
        ax.add_patch(box)
        ax.text(1.15, inp["y"] + inp["h"]/2 + 0.12, inp["title"], ha="center", va="center", fontsize=9.5, fontweight="bold", color="#0f172a")
        ax.text(1.15, inp["y"] + inp["h"]/2 - 0.18, inp["shape"], ha="center", va="center", fontsize=7.5, color="#7c3aed", fontweight="semibold")
        
    # --- Column 1: ENCODER (Orange Theme) ---
    orangeBorder = "#ea580c"
    encoders = [
        {"y": 3.45, "h": 1.0, "title": "ActorNet", "shape": "(B, N, D)"},
        {"y": 1.45, "h": 1.0, "title": "MapNet", "shape": "(B, M, D)"}
    ]
    for enc in encoders:
        box = patches.FancyBboxPatch(
            (2.3, enc["y"]), 1.3, enc["h"],
            boxstyle="round,pad=0.02",
            facecolor="white", edgecolor=orangeBorder, linewidth=1.0, zorder=2
        )
        ax.add_patch(box)
        ax.text(2.95, enc["y"] + enc["h"]/2 + 0.14, enc["title"], ha="center", va="center", fontsize=10, fontweight="bold", color="#0f172a")
        ax.text(2.95, enc["y"] + enc["h"]/2 - 0.16, enc["shape"], ha="center", va="center", fontsize=8, color="#ea580c", fontweight="semibold")
        
    # --- Column 2: CROSS MODAL FUSION (Blue Theme with Light Blue Fill) ---
    blueBorder = "#0284c7"
    blueFill = "#eff6ff"
    fusions = [
        {"y": 3.90, "h": 0.55, "title": "A2M", "sub": "(Agent Map)"},
        {"y": 3.10, "h": 0.55, "title": "M2M", "sub": "(Map Map)"},
        {"y": 2.30, "h": 0.55, "title": "M2A", "sub": "(Map Agent)"},
        {"y": 1.50, "h": 0.55, "title": "A2A", "sub": "(Agent Agent)"}
    ]
    for fus in fusions:
        box = patches.FancyBboxPatch(
            (4.1, fus["y"]), 1.3, fus["h"],
            boxstyle="round,pad=0.02",
            facecolor=blueFill, edgecolor=blueBorder, linewidth=1.0, zorder=2
        )
        ax.add_patch(box)
        ax.text(4.75, fus["y"] + fus["h"]/2 + 0.10, fus["title"], ha="center", va="center", fontsize=8.5, fontweight="bold", color="#0f172a")
        ax.text(4.75, fus["y"] + fus["h"]/2 - 0.12, fus["sub"], ha="center", va="center", fontsize=7.5, color="#0284c7", fontweight="semibold")
        
    # --- Column 3: GAMEFORMER DECODER (Green Theme with Clipped Header background) ---
    greenBorder = "#16a34a"
    greenHeaderFill = "#bbf7d0"
    
    # --- Level 0 (Larger to fit Physics Prior sub box) ---
    l0_box = patches.FancyBboxPatch(
        (5.9, 3.50), 1.6, 0.95,
        boxstyle="round,pad=0.02",
        facecolor="white", edgecolor=greenBorder, linewidth=1.0, zorder=2
    )
    ax.add_patch(l0_box)
    
    # Clipped header for Level 0
    l0_header_h = 0.43
    l0_header_rect = patches.Rectangle(
        (5.9 - 0.05, 3.50 + 0.95 - l0_header_h), 1.6 + 0.1, l0_header_h + 0.05,
        facecolor=greenHeaderFill, edgecolor="none", zorder=2.5
    )
    l0_header_rect.set_clip_path(l0_box)
    ax.add_patch(l0_header_rect)
    ax.text(6.7, 4.235, "Level 0", ha="center", va="center", fontsize=9.5, fontweight="bold", color="#0f172a", zorder=3)
    
    # Inside Level 0: Physics Prior sub box
    prior_box = patches.Rectangle(
        (6.05, 3.73), 1.3, 0.24,
        facecolor="#f0fdf4", edgecolor="#22c55e", linewidth=0.5, zorder=3
    )
    ax.add_patch(prior_box)
    ax.text(6.7, 3.85, "Physics Prior", ha="center", va="center", fontsize=8, fontweight="bold", color="#16a34a", zorder=3.5)
    
    # Inside Level 0: Agent attends to others only pill box
    pill_box = patches.FancyBboxPatch(
        (6.05, 3.53), 1.3, 0.16,
        boxstyle="round,pad=0.01",
        facecolor="white", edgecolor="#e2e8f0", linewidth=0.6, zorder=3
    )
    ax.add_patch(pill_box)
    ax.text(6.7, 3.61, "Agent \"i\" attends to others only", ha="center", va="center", fontsize=6.5, color="#475569", fontweight="semibold", zorder=3.5)
    
    # --- Levels 1, 2, 3 ---
    levels = [
        {"y": 2.65, "h": 0.65, "lbl": "Level 1"},
        {"y": 1.80, "h": 0.65, "lbl": "Level 2"},
        {"y": 0.95, "h": 0.65, "lbl": "Level 3"}
    ]
    for lvl in levels:
        box = patches.FancyBboxPatch(
            (5.9, lvl["y"]), 1.6, lvl["h"],
            boxstyle="round,pad=0.02",
            facecolor="white", edgecolor=greenBorder, linewidth=1.0, zorder=2
        )
        ax.add_patch(box)
        
        # Clipped green header for this level
        lvl_header_h = 0.29
        lvl_header_rect = patches.Rectangle(
            (5.9 - 0.05, lvl["y"] + lvl["h"] - lvl_header_h), 1.6 + 0.1, lvl_header_h + 0.05,
            facecolor=greenHeaderFill, edgecolor="none", zorder=2.5
        )
        lvl_header_rect.set_clip_path(box)
        ax.add_patch(lvl_header_rect)
        ax.text(6.7, lvl["y"] + lvl["h"] - lvl_header_h/2, lvl["lbl"], ha="center", va="center", fontsize=9.5, fontweight="bold", color="#0f172a", zorder=3)
        
    # --- DIMENSION CARDS AND BRACKETS FOR DECODER (FOLDER TAB INDEX CARDS) ---
    dimCards = [
        {"y": 3.80, "h": 0.35, "w": 0.65, "shape": "(D, D)"},
        {"y": 2.80, "h": 0.35, "w": 1.05, "shape": "(B, N, K, T, 2)"},
        {"y": 1.95, "h": 0.35, "w": 0.85, "shape": "(B, N, K, 1)"},
        {"y": 1.10, "h": 0.35, "w": 0.85, "shape": "(B, N, K)"}
    ]
    for card in dimCards:
        # Draw main card body with thin grey border and white fill
        cardBox = patches.FancyBboxPatch(
            (7.1, card["y"]), card["w"], card["h"],
            boxstyle="round,pad=0.01",
            facecolor="white", edgecolor="#94a3b8", linewidth=0.6, zorder=3
        )
        ax.add_patch(cardBox)
        
        # Draw index folder tab grey block at the top using clipped path
        tab_h = 0.08
        tab_rect = patches.Rectangle(
            (7.1 - 0.02, card["y"] + card["h"] - tab_h), card["w"] + 0.04, tab_h + 0.01,
            facecolor="#e2e8f0", edgecolor="none", zorder=3.5
        )
        tab_rect.set_clip_path(cardBox)
        ax.add_patch(tab_rect)
        
        # Horizontal separation line under tab
        ax.plot([7.1, 7.1 + card["w"]], [card["y"] + card["h"] - tab_h, card["y"] + card["h"] - tab_h],
                color="#94a3b8", linewidth=0.5, zorder=4)
        
        # Shape text inside the white card slot
        ax.text(7.1 + card["w"]/2, card["y"] + (card["h"] - tab_h)/2, card["shape"], 
                ha="center", va="center", fontsize=7.5, color="#0f172a", fontweight="semibold", zorder=4.5)
        
    # --- Column 4: OUTPUTS (Red Theme) ---
    redBorder = "#dc2626"
    outputs = [
        {"y": 3.15, "h": 0.95, "title": "GMM Trajectories", "shape": "(B, N, K, T, 2)"},
        {"y": 1.45, "h": 0.95, "title": "Mode Probabilities", "shape": "(B, N, K, 1)"}
    ]
    for out in outputs:
        box = patches.FancyBboxPatch(
            (8.0, out["y"]), 1.3, out["h"],
            boxstyle="round,pad=0.02",
            facecolor="white", edgecolor=redBorder, linewidth=1.0, zorder=2
        )
        ax.add_patch(box)
        ax.text(8.65, out["y"] + out["h"]/2 + 0.14, out["title"], ha="center", va="center", fontsize=9.5, fontweight="bold", color="#0f172a")
        ax.text(8.65, out["y"] + out["h"]/2 - 0.16, out["shape"], ha="center", va="center", fontsize=7.5, color="#dc2626", fontweight="semibold")
        
    # --- Column 5: LOSS (Yellow Theme) ---
    yellowBorder = "#ca8a04"
    losses = [
        {"y": 3.90, "h": 0.55, "title": "Conf", "var": "Lconf"},
        {"y": 3.10, "h": 0.55, "title": "Ref", "var": "Lref"},
        {"y": 2.30, "h": 0.55, "title": "Safety", "var": "Lsafety"},
        {"y": 1.50, "h": 0.55, "title": "Skip", "var": "Lskip"}
    ]
    for los in losses:
        box = patches.FancyBboxPatch(
            (9.8, los["y"]), 1.3, los["h"],
            boxstyle="round,pad=0.02",
            facecolor="white", edgecolor=yellowBorder, linewidth=1.0, zorder=2
        )
        ax.add_patch(box)
        ax.text(10.45, los["y"] + los["h"]/2 + 0.10, los["title"], ha="center", va="center", fontsize=9.0, fontweight="bold", color="#0f172a")
        # Subscript notations match the original exactly using math mode
        subscriptVar = r"$\mathcal{L}_{\mathrm{" + los["var"][1:] + r"}}$"
        ax.text(10.45, los["y"] + los["h"]/2 - 0.14, subscriptVar, ha="center", va="center", fontsize=8.5, color="#ca8a04", fontweight="bold")
        
        
    # --- DRAW ARROWS BETWEEN BLOCKS (SHARP, ELEGANT ARROWHEADS WITH MUTATION SCALE 8) ---
    
    lineColor = "#475569"
    arrowProps = dict(arrowstyle="-|>", color=lineColor, linewidth=1.0, shrinkA=0, shrinkB=2, mutation_scale=8.0)
    
    # 1. Inputs to Encoder (Perfect orthogonal turn based routing)
    # Agent History center-right to ActorNet top-left
    draw_orthogonal_arrow(ax, 1.8, 4.225, 2.3, 3.95, 2.05, arrowProps, lineColor)
    # Traffic Flow center-right to MapNet bottom-left
    draw_orthogonal_arrow(ax, 1.8, 2.025, 2.3, 1.95, 2.05, arrowProps, lineColor)
    # Lane Graph center-right orthogonal fork to bottom of ActorNet and top of MapNet
    ax.plot([1.8, 2.05], [3.125, 3.125], color=lineColor, linewidth=1.0, zorder=3)
    ax.plot([2.05, 2.05], [2.15, 3.75], color=lineColor, linewidth=1.0, zorder=3)
    ax.annotate("", xy=(2.3, 3.75), xytext=(2.05, 3.75), arrowprops=arrowProps, zorder=3)
    ax.annotate("", xy=(2.3, 2.15), xytext=(2.05, 2.15), arrowprops=arrowProps, zorder=3)
                
    # 2. Encoder to Cross Modal Fusion (Separate non overlapping orthogonal fork lines)
    # ActorNet center-right to A2M, M2A, and A2A (turn at x_turn = 3.8)
    ax.plot([3.6, 3.8], [3.95, 3.95], color=lineColor, linewidth=1.0, zorder=3)
    ax.plot([3.8, 3.8], [1.775, 4.175], color=lineColor, linewidth=1.0, zorder=3)
    ax.annotate("", xy=(4.1, 4.175), xytext=(3.8, 4.175), arrowprops=arrowProps, zorder=3)
    ax.annotate("", xy=(4.1, 2.575), xytext=(3.8, 2.575), arrowprops=arrowProps, zorder=3)
    ax.annotate("", xy=(4.1, 1.775), xytext=(3.8, 1.775), arrowprops=arrowProps, zorder=3)
    
    # MapNet center-right to A2M, M2M, M2A, and A2A (turn at x_turn = 3.9)
    ax.plot([3.6, 3.9], [1.95, 1.95], color=lineColor, linewidth=1.0, zorder=3)
    ax.plot([3.9, 3.9], [1.775, 4.175], color=lineColor, linewidth=1.0, zorder=3)
    ax.annotate("", xy=(4.1, 4.175), xytext=(3.9, 4.175), arrowprops=arrowProps, zorder=3)
    ax.annotate("", xy=(4.1, 3.375), xytext=(3.9, 3.375), arrowprops=arrowProps, zorder=3)
    ax.annotate("", xy=(4.1, 2.575), xytext=(3.9, 2.575), arrowprops=arrowProps, zorder=3)
    ax.annotate("", xy=(4.1, 1.775), xytext=(3.9, 1.775), arrowprops=arrowProps, zorder=3)
                
    # 3. Cross Modal Fusion to Decoder Level 0 (Orthogonal merge routing)
    # All 4 fusion outputs merge at x = 5.65 and enter Level 0
    ax.plot([5.4, 5.65], [4.175, 4.175], color=lineColor, linewidth=1.0, zorder=3)
    ax.plot([5.4, 5.65], [3.375, 3.375], color=lineColor, linewidth=1.0, zorder=3)
    ax.plot([5.4, 5.65], [2.575, 2.575], color=lineColor, linewidth=1.0, zorder=3)
    ax.plot([5.4, 5.65], [1.775, 1.775], color=lineColor, linewidth=1.0, zorder=3)
    ax.plot([5.65, 5.65], [1.775, 4.175], color=lineColor, linewidth=1.0, zorder=3)
    ax.annotate("", xy=(5.9, 3.975), xytext=(5.65, 3.975), arrowprops=arrowProps, zorder=3)
                
    # 4. Decoder Level Flow
    ax.annotate("", xy=(6.7, 3.30), xytext=(6.7, 3.50), arrowprops=arrowProps, zorder=3)
    ax.annotate("", xy=(6.7, 2.45), xytext=(6.7, 2.65), arrowprops=arrowProps, zorder=3)
    ax.annotate("", xy=(6.7, 1.60), xytext=(6.7, 1.80), arrowprops=arrowProps, zorder=3)
                
    # 5. Decoder to Outputs (Orthogonal branching flow)
    # Level 1, 2, 3 outputs merge at x = 7.75 and branch to Outputs
    ax.plot([7.5, 7.75], [2.975, 2.975], color=lineColor, linewidth=1.0, zorder=3)
    ax.plot([7.5, 7.75], [2.125, 2.125], color=lineColor, linewidth=1.0, zorder=3)
    ax.plot([7.5, 7.75], [1.275, 1.275], color=lineColor, linewidth=1.0, zorder=3)
    ax.plot([7.75, 7.75], [1.275, 3.625], color=lineColor, linewidth=1.0, zorder=3)
    ax.annotate("", xy=(8.0, 3.625), xytext=(7.75, 3.625), arrowprops=arrowProps, zorder=3)
    ax.annotate("", xy=(8.0, 1.925), xytext=(7.75, 1.925), arrowprops=arrowProps, zorder=3)
                
    # 6. Outputs to Loss (Orthogonal fork routing entering exact centers of loss boxes)
    # GMM Trajectories and Mode Probabilities branch out to all 4 loss terms
    ax.plot([9.3, 9.55], [3.625, 3.625], color=lineColor, linewidth=1.0, zorder=3)
    ax.plot([9.3, 9.55], [1.925, 1.925], color=lineColor, linewidth=1.0, zorder=3)
    ax.plot([9.55, 9.55], [1.775, 4.175], color=lineColor, linewidth=1.0, zorder=3)
    ax.annotate("", xy=(9.8, 4.175), xytext=(9.55, 4.175), arrowprops=arrowProps, zorder=3)
    ax.annotate("", xy=(9.8, 3.375), xytext=(9.55, 3.375), arrowprops=arrowProps, zorder=3)
    ax.annotate("", xy=(9.8, 2.575), xytext=(9.55, 2.575), arrowprops=arrowProps, zorder=3)
    ax.annotate("", xy=(9.8, 1.775), xytext=(9.55, 1.775), arrowprops=arrowProps, zorder=3)
                
                
    # --- DRAW LEGEND AT THE BOTTOM ---
    legend_y = 0.20
    # Bounding legend background block
    leg_bg = patches.FancyBboxPatch(
        (1.0, legend_y - 0.12), 10.0, 0.25,
        boxstyle="round,pad=0.01",
        facecolor="white", edgecolor="#cfd8dc", linewidth=0.8, zorder=1
    )
    ax.add_patch(leg_bg)
    
    # Legend items
    items = [
        ("Input Layer", "#faf5ff", purpleBorder),
        ("LaneGCN Encoder", "#fff7ed", orangeBorder),
        ("Cross Modal Fusion", blueFill, blueBorder),
        ("GameFormer Decoder", greenHeaderFill, greenBorder),
        ("Output Layer", "#fef2f2", redBorder),
        ("Loss Functions", "#fefce8", yellowBorder)
    ]
    
    x_offset = 1.3
    for name, fc, ec in items:
        # Patch box representing the color coding
        item_patch = patches.Rectangle(
            (x_offset, legend_y - 0.05), 0.25, 0.10,
            facecolor=fc, edgecolor=ec, linewidth=1.0, zorder=2
        )
        ax.add_patch(item_patch)
        ax.text(x_offset + 0.30, legend_y, name, ha="left", va="center", fontsize=8.5, fontweight="semibold", color="#1e293b")
        x_offset += 1.6
        
    plt.tight_layout()
    plt.savefig(outputPath)
    plt.close()
    print(f"Successfully generated pure vector architecture diagram at {outputPath}")

if __name__ == "__main__":
    baseDir = os.path.dirname(os.path.abspath(__file__))
    
    # Save directly to the workspace root as requested
    rootPath = os.path.join(baseDir, "Architecture.png")
    drawDiagram(rootPath)
    
    # Save to the paper image folder to keep them in sync
    paperPath = os.path.join(baseDir, "paper", "img", "Architecture.png")
    drawDiagram(paperPath)
