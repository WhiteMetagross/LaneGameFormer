"""
Unified script to generate publication quality research figures for the paper.

This script creates:
1. Figure 1: LaneGameFormer ablation study relative performance change plot.
2. Figure 2: FlowSPF variant comparison plot with displacement and safety panels.

All text, labels, comments, and structure strictly follow the AGENTS style guide:
* No semicolons are used under any circumstances.
* No em dashes or double hyphens are used.
* Compound terms are space separated (e.g. Level k, zero shot, map less).
* The academic name FlowSPF is used consistently.
"""

import os
import matplotlib.pyplot as plt
import numpy as np

def setupPlotStyle():
    """Configure matplotlib with a clean academic visual style."""
    plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["font.sans-serif"] = ["DejaVu Sans", "Liberation Sans", "Arial"]
    plt.rcParams["font.size"] = 10
    plt.rcParams["axes.labelsize"] = 11
    plt.rcParams["axes.titlesize"] = 12
    plt.rcParams["xtick.labelsize"] = 9
    plt.rcParams["ytick.labelsize"] = 10
    plt.rcParams["legend.fontsize"] = 9
    plt.rcParams["axes.grid"] = True
    plt.rcParams["grid.alpha"] = 0.15
    plt.rcParams["grid.linestyle"] = "--"
    plt.rcParams["savefig.dpi"] = 300
    plt.rcParams["savefig.bbox"] = "tight"

def generateAblationPlot(outputPath):
    """Generate Figure 1: LaneGameFormer relative ablation study plot.
    
    This plot shows the relative percentage change in minADE@1 compared to the base model.
    The extreme outlier K0 (Level 0) is truncated visually with an explicit label.
    """
    fig, ax = plt.subplots(figsize=(8, 4.5))
    
    # Ablation variants and their relative performance changes in minADE@1
    # positive represents improvement (lower error), negative represents degradation (higher error)
    variants = [
        "Full DB (A1)",
        "Map less (M0)",
        "TTC Only (A2)",
        "CPA Only (A3)",
        "Base Model",
        "No DB (A4)",
        "No Safety (S1)",
        "Level 0 (K0)"
    ]
    
    # Relative changes calculated precisely from the paper evaluation results
    changes = [
        0.172,
        0.012,
        0.012,
        0.012,
        0.000,
        -0.004,
        -0.115,
        -0.200 # Truncated for visual scale, actual is -929.47%
    ]
    
    # Beautiful palette using green shades for improvement, warm coral for degradation, and grey for baseline
    colors = [
        "#0f9d58", # Emerald Green
        "#34a853", # Medium Green
        "#4caf50", # Soft Green
        "#81c784", # Light Green
        "#78909c", # Grey for Base Model
        "#ff8a80", # Light Coral
        "#e57373", # Medium Coral
        "#b71c1c"  # Dark Red for Level 0
    ]
    
    # Plot the horizontal bars
    bars = ax.barh(variants, changes, color=colors, height=0.6)
    
    # Configure axes limits to highlight the fine grained variations
    ax.set_xlim(-0.25, 0.25)
    
    # Custom formatting for the X axis showing percentages
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f"{x:+.2f}%" if x != 0 else "0.0%"))
    
    # Draw a distinct vertical line at the baseline
    ax.axvline(x=0.0, color="#37474f", linewidth=1.2, linestyle="-", alpha=0.8)
    
    # Highlight the baseline with a shaded background block
    ax.axvspan(-0.002, 0.002, color="#cfd8dc", alpha=0.3)
    
    # Add values and descriptions at the end of each bar
    for bar, val, var in zip(bars, changes, variants):
        width = bar.get_width()
        if var == "Base Model":
            ax.text(0.01, bar.get_y() + bar.get_height()/2, "Baseline", 
                    va="center", ha="left", color="#37474f", fontweight="bold", fontsize=9)
        elif var == "Level 0 (K0)":
            # Distinct label for the truncated Level 0 outlier
            ax.text(-0.19, bar.get_y() + bar.get_height()/2, "-929.5% (Catastrophic)", 
                    va="center", ha="left", color="white", fontweight="bold", fontsize=8.5)
            # Add a hatch pattern to the Level 0 bar to denote visual truncation
            bar.set_hatch("//")
        else:
            if val >= 0:
                ax.text(width + 0.005, bar.get_y() + bar.get_height()/2, f"+{val:.3f}%", 
                        va="center", ha="left", color="#0f9d58", fontweight="semibold", fontsize=9)
            else:
                ax.text(width - 0.005, bar.get_y() + bar.get_height()/2, f"{val:.3f}%", 
                        va="center", ha="right", color="#b71c1c", fontweight="semibold", fontsize=9)
                
    # Remove top and right spines for a modern, premium look
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#b0bec5")
    ax.spines["bottom"].set_color("#b0bec5")
    
    # Invert y axis so the best improvements are at the top
    ax.invert_yaxis()
    
    # Titles and labels following the space separated compound terms guidelines
    ax.set_xlabel("Relative Prediction Performance Change (%)", labelpad=10, fontweight="semibold")
    ax.set_title("LaneGameFormer Ablation Study: Architectural Component Impact Analysis", 
                 pad=15, fontweight="bold", color="#263238")
    
    # Add a visual direction indicator box
    props = dict(boxstyle="round,pad=0.5", facecolor="#f5f7f8", edgecolor="#cfd8dc", alpha=0.9)
    ax.text(0.18, 5.8, "Better $\\rightarrow$\n$\\leftarrow$ Worse", 
            va="center", ha="center", bbox=props, color="#37474f", fontsize=9)
            
    plt.tight_layout()
    plt.savefig(outputPath)
    plt.close()
    print(f"Successfully generated ablation plot at {outputPath}")

def generateFlowSpfPlot(outputPath):
    """Generate Figure 2: FlowSPF variants comparison plot.
    
    This side by side double panel grouped bar chart compares the four FlowSPF variants
    on displacement accuracy and safety compliance in physical meters.
    """
    # Create a 1 row, 2 column figure to cleanly separate displacement error from safety rates
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    
    variants = [
        "A: Kinematic Only",
        "B: Conventional TTC",
        "C: CPA VTTC Yield",
        "D: Space Time SCI"
    ]
    
    # Custom scientific color palette representing the variants
    # Variant C (best displacement) is highlighted in emerald green
    # Variant D (best collision rate) is highlighted in active coral/orange
    # Grays and dark blues for the rest
    varColors = ["#78909c", "#455a64", "#0f9d58", "#e65100"]
    
    # Data for Subplot 1: Displacement Errors in meters
    minAde = [4.1847, 4.2030, 3.9205, 4.2179]
    minFde = [8.2426, 8.2740, 7.6728, 8.3063]
    
    x = np.arange(len(variants))
    width = 0.35
    
    # Subplot 1: Displacement Accuracy
    rects1 = ax1.bar(x - width/2, minAde, width, label="minADE@1", color="#37474f", alpha=0.85)
    rects2 = ax1.bar(x + width/2, minFde, width, label="minFDE@1", color="#cfd8dc", edgecolor="#37474f", linewidth=0.8)
    
    ax1.set_ylabel("Displacement Error (meters)", fontweight="semibold")
    ax1.set_title("Displacement Error Comparison", fontweight="bold", color="#263238", pad=10)
    ax1.set_xticks(x)
    ax1.set_xticklabels(variants, rotation=15, ha="right")
    ax1.set_ylim(0, 9.8)
    ax1.legend(loc="lower right")
    
    # Add value labels on top of the displacement bars
    for rect in rects1:
        height = rect.get_height()
        ax1.annotate(f"{height:.2f}m",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 2),  # 2 points vertical offset
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)
                    
    for rect in rects2:
        height = rect.get_height()
        # Draw the best FDE label in bold green, others in grey
        isBest = (height == min(minFde))
        color = "#0f9d58" if isBest else "#37474f"
        weight = "bold" if isBest else "normal"
        ax1.annotate(f"{height:.2f}m",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=color, fontweight=weight)
                    
    # Subplot 2: Safety Compliance
    collisionRate = [2.27, 2.28, 2.31, 2.17]
    offRoadRate = [16.24, 16.34, 15.91, 16.85]
    
    rects3 = ax2.bar(x - width/2, collisionRate, width, label="Collision Rate", color="#e65100", alpha=0.85)
    rects4 = ax2.bar(x + width/2, offRoadRate, width, label="Off Road Rate", color="#ffe0b2", edgecolor="#e65100", linewidth=0.8)
    
    ax2.set_ylabel("Rate (%)", fontweight="semibold")
    ax2.set_title("Safety and Boundary Violations", fontweight="bold", color="#263238", pad=10)
    ax2.set_xticks(x)
    ax2.set_xticklabels(variants, rotation=15, ha="right")
    ax2.set_ylim(0, 19.5)
    ax2.legend(loc="center right")
    
    # Add value labels on top of the safety bars
    for rect in rects3:
        height = rect.get_height()
        isBest = (height == min(collisionRate))
        color = "#d84315" if isBest else "#37474f"
        weight = "bold" if isBest else "normal"
        ax2.annotate(f"{height:.2f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=color, fontweight=weight)
                    
    for rect in rects4:
        height = rect.get_height()
        isBest = (height == min(offRoadRate))
        color = "#0f9d58" if isBest else "#37474f"
        weight = "bold" if isBest else "normal"
        ax2.annotate(f"{height:.2f}%",
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 2),
                    textcoords="offset points",
                    ha="center", va="bottom", fontsize=8, color=color, fontweight=weight)
                    
    # Clean spines for both panels
    for ax in [ax1, ax2]:
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_color("#b0bec5")
        ax.spines["bottom"].set_color("#b0bec5")
        ax.set_axisbelow(True)
        
    plt.suptitle("FlowSPF Algorithmic Variants Empirical Comparison (10 Hz Test Split)", 
                 fontsize=13, fontweight="bold", y=0.98, color="#263238")
    plt.tight_layout()
    plt.savefig(outputPath)
    plt.close()
    print(f"Successfully generated FlowSPF variant comparison plot at {outputPath}")

if __name__ == "__main__":
    setupPlotStyle()
    
    # Establish local file paths
    # The figures must reside in the paper/img/ directory
    baseDir = os.path.dirname(os.path.abspath(__file__))
    imgDir = os.path.join(baseDir, "paper", "img")
    
    if not os.path.exists(imgDir):
        os.makedirs(imgDir)
        
    ablationPath = os.path.join(imgDir, "ablation_study.png")
    flowSpfPath = os.path.join(imgDir, "flow_spf_comparison.png")
    
    generateAblationPlot(ablationPath)
    generateFlowSpfPlot(flowSpfPath)
