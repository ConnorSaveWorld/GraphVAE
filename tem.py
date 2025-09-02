import matplotlib.pyplot as plt
import numpy as np
# Rebuild Matplotlib's font cache to ensure newly installed fonts are detected
import matplotlib.font_manager as fm
fm.fontManager = fm.FontManager()

# --- Data ---
# Your model's performance data
layers = [128, 256, 512, 768]
auc_ours = [81.17, 82.63, 81.90, 81.08]
# Standard deviations (optional, can be used for error bars if desired)
# std_devs = [0.0784, 0.1185, 0.1342, 0.0430, 0.0732]

# --- Plot Styling ---
# Define colors to match the academic style of the example
colors = {
    'dark_blue': '#3B5998',
}

# Set font properties for a professional look
# Enforce Times New Roman; raise an informative error if it's not available
if any('Times New Roman' in f.name for f in fm.fontManager.ttflist):
    plt.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams['font.serif'] = ['Times New Roman']
else:
    raise RuntimeError("Times New Roman font is required but was not found on this system. "
                       "Install the font (e.g., via 'sudo apt-get install ttf-mscorefonts-installer' "
                       "on Debian/Ubuntu-based systems, or copy the TTF files into ~/.fonts) and rerun.")
plt.rcParams['mathtext.fontset'] = 'stix'  # For math text if any

# --- Create the Plot ---
fig, ax = plt.subplots(figsize=(8, 6))

# Plot your model's performance
ax.plot(layers, auc_ours,
        marker='o',
        markersize=15,
        linestyle='-',
        linewidth=3.5,
        color=colors['dark_blue'],
        label='FVDM (Ours)',
        zorder=10) # zorder makes sure the line is on top of the grid

# --- Formatting and Labels ---
# Set title and labels with appropriate font sizes
# ax.set_title('(a)', fontsize=36, fontweight='bold', pad=20)
ax.set_xlabel('# latent dimensions', fontsize=32, labelpad=15)
ax.set_ylabel('AUC (%)', fontsize=32, labelpad=15)

# Set axis limits and ticks for clarity
ax.set_xlim(0.5, 5.5)
ax.set_ylim(79, 84) # Adjust Y-axis to focus on the performance range
ax.set_xticks(layers)
ax.set_yticks(np.arange(78, 85, 2))

# Customize tick parameters
ax.tick_params(axis='both', which='major', labelsize=24,
               direction='in', length=10, width=2, top=True, right=True)
ax.tick_params(axis='both', which='minor',
               direction='in', length=5, width=1, top=True, right=True)

# Add minor ticks for a more detailed look
ax.minorticks_on()

# Add a grid
ax.grid(True, which='major', linestyle='-', linewidth=0.7, color='gray', alpha=0.5)

# Customize the plot's border (spines)
for spine in ax.spines.values():
    spine.set_linewidth(2)

# Add a legend
legend = ax.legend(fontsize=20, frameon=True, edgecolor='black')
legend.get_frame().set_linewidth(1.5)

# Ensure tight layout and save the figure
plt.tight_layout()
plt.savefig("latent_dim.png", dpi=300)
plt.show()