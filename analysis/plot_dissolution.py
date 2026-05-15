import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from matplotlib.ticker import LogLocator, NullFormatter

ANALYSIS_DIR = Path(__file__).parent.absolute()
ROOT = ANALYSIS_DIR.parent
CASE_TO_PLOT = 'nh3_gly_cn_low'

# Load the data
dissolution_df = pd.read_csv(ROOT / 'data' / 'dissolution.csv')
pourbaix_df = pd.read_csv(ANALYSIS_DIR / 'pourbaix_dissolution_energy.csv')
dissolution_df = dissolution_df.loc[:, ~dissolution_df.columns.str.startswith('Unnamed')]
pourbaix_df = pourbaix_df[pourbaix_df['case'] == CASE_TO_PLOT].copy()

# The experimental dissolution data uses "material", while the Pourbaix
# calculation output uses "metal" and contains one row per ligand case.
dissolution_df = dissolution_df.rename(columns={'material': 'metal'})
merged_df = pd.merge(dissolution_df, pourbaix_df, on='metal', how='inner')

if merged_df.empty:
    raise ValueError(
        f'No matching metals found between dissolution.csv and {CASE_TO_PLOT} Pourbaix data'
    )

merged_df = merged_df.sort_values('delta_G_pbx_best_eV_per_metal')
merged_df['material_type'] = merged_df['metal'].str.contains(r'[A-Z][a-z]?[A-Z]').map(
    {True: 'Alloy', False: 'Single metal'}
)

fig, ax = plt.subplots(figsize=(6.8, 4.8))

style = {
    'Single metal': {'marker': 'o', 'facecolor': '#4c78a8'},
    'Alloy': {'marker': 's', 'facecolor': '#f58518'},
}

for material_type, group in merged_df.groupby('material_type', sort=False):
    ax.scatter(
        group['delta_G_pbx_best_eV_per_metal'],
        group['metal_dissolution_mu_mole'],
        s=72,
        marker=style[material_type]['marker'],
        facecolor=style[material_type]['facecolor'],
        edgecolor='black',
        linewidth=0.6,
        label=material_type,
        zorder=3,
    )

label_offsets = {
    'Ni': (4, -12),
    'TiNi': (4, 7),
    'Au': (4, 7),
    'Pd': (-6, -12),
    'Cu': (4, 7),
}
for _, row in merged_df.iterrows():
    ax.annotate(
        row['metal'],
        (row['delta_G_pbx_best_eV_per_metal'], row['metal_dissolution_mu_mole']),
        xytext=label_offsets.get(row['metal'], (4, 6)),
        textcoords='offset points',
        fontsize=9,
        ha='right' if row['metal'] == 'Pd' else 'left',
    )

ax.axvline(0, color='0.35', linestyle='--', linewidth=1, zorder=1)
ax.text(
    0.02,
    0.97,
    'Pourbaix-favored dissolution',
    transform=ax.transAxes,
    ha='left',
    va='top',
    fontsize=9,
    color='0.35',
)
ax.set_xlabel('Pourbaix dissolution energy, best aqueous - solid ref (eV/metal)')
ax.set_ylabel('Metal dissolved (umol)')
ax.set_title('Metal Dissolution vs Pourbaix Energy: NH3 + Gly + Low CN', fontsize=13, pad=10)
ax.set_yscale('log')
ax.set_xlim(
    merged_df['delta_G_pbx_best_eV_per_metal'].min() - 0.15,
    merged_df['delta_G_pbx_best_eV_per_metal'].max() + 0.18,
)
ax.set_ylim(8e-3, 2e3)
ax.yaxis.set_major_locator(LogLocator(base=10))
ax.yaxis.set_minor_formatter(NullFormatter())
ax.grid(True, which='major', color='0.88', linewidth=0.8)
ax.grid(True, which='minor', axis='y', color='0.93', linewidth=0.5)
ax.legend(frameon=False, loc='center right', fontsize=9, handletextpad=0.5)
fig.tight_layout()
output_path = ANALYSIS_DIR / 'dissolution_vs_pourbaix.png'
try:
    plt.savefig(output_path, dpi=300)
except OSError:
    output_path = Path.home() / 'dissolution_vs_pourbaix.png'
    plt.savefig(output_path, dpi=300)
print(f'Saved plot to {output_path}')
# plt.show()
