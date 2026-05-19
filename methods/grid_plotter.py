import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
try:
    from ase.formula import Formula
except ModuleNotFoundError:
    class Formula:
        def __init__(self, formula):
            self.formula = formula

        def __eq__(self, other):
            return self.formula == other

        def reduce(self):
            return (self, 1)

        def __format__(self, spec):
            return self.formula
import re
import os

AREA_LABEL_FONT_SIZE = 20
AXIS_LABEL_FONT_SIZE = 26
TICK_LABEL_FONT_SIZE = AXIS_LABEL_FONT_SIZE
REGION_ALPHA = 0.50
WATER_LINE_WIDTH = 2.0
LIGAND_ELEMENT_COUNTS = {
    'NH3': {'N': 1, 'H': 3},
    'CN': {'C': 1, 'N': 1},
    'Gly': {'C': 2, 'H': 4, 'N': 1, 'O': 2},
    'NO2': {'N': 1, 'O': 2},
}

class GridPlotter:
    def __init__(self, pH_range, V_range, data, grid_size, save_fig, dir, filename):
        self.dir = dir
        self.filename = filename
        self.pH_range = pH_range
        self.V_range = V_range
        self.grid_size = grid_size
        self.pH_values = np.linspace(pH_range[0], pH_range[1], grid_size)
        self.V_values = np.linspace(V_range[0], V_range[1], grid_size)
        self.pH_grid, self.V_grid = np.meshgrid(self.pH_values, self.V_values)
        
        self.data=data
        self.save_fig = save_fig
        self.pNH3_grid = self.generate_pNH3_grid()
        self.pNO2_grid = self.generate_pNO2_grid()
        self.pGly_grid = self.generate_pGly_grid()
        self.pCN_grid = self.generate_pCN_grid()
        
        self.ligand_grid_dict = {
        'NH3': self.pNH3_grid,
        'NO2': self.pNO2_grid,
        'Gly': self.pGly_grid,
         'CN': self.pCN_grid
    }
        
    def _large_pL_grid(self, large_pL=100):
        # print('hitting large pL grid')
        return np.full_like(self.pH_grid, large_pL, dtype=float)
    
    def generate_pNO2_grid(self, pKa=3.92):
        NO2_tot = self.data.ligand_concentration['NO2']
        if NO2_tot <= 0:
            return self._large_pL_grid()
        return (- np.log10(NO2_tot) + np.log10(1 + 10 ** (pKa - self.pH_grid)))
    
    def generate_pNH3_grid(self, pKa=9.25):
        NH3_tot = self.data.ligand_concentration['NH3']
        if NH3_tot <= 0:
            return self._large_pL_grid()
        return (- np.log10(NH3_tot) + np.log10(1 + 10 ** (pKa - self.pH_grid)))
    
    def generate_pCN_grid(self, pKa=9.2):
        CN_tot = self.data.ligand_concentration['CN']
        if CN_tot <= 0:
            return self._large_pL_grid()
        return (- np.log10(CN_tot) + np.log10(1 + 10 ** (pKa - self.pH_grid)))
    
    def generate_pGly_grid(self, pKa1=2.35, pKa2=9.78):
        Gly_tot = self.data.ligand_concentration['Gly']
        if Gly_tot <= 0:
            return self._large_pL_grid()
        return (- np.log10(Gly_tot) + (pKa1 - self.pH_grid) 
                + np.log10(1 + 10 ** (self.pH_grid - pKa1) 
                           + 1/(10 ** (self.pH_grid - pKa2))))
    def format_formula(self, formula):
        """
        Converts a chemical formula to a LaTeX-friendly format with subscripts.
        Example: "H2O" -> "H$_2$O"
        """

        if '[' in formula or 'aq' in formula or 'Gly' in formula:
            
            formatted_formula = re.sub(r"([A-Za-z\)\]])(\d+)", r"\1$_{\2}$", formula)
            formatted_formula = re.sub(r"\[([\d\+\-]+)\]", r"$^{\1}$", formatted_formula)

            formatted_formula = re.sub(r"\$_\{1\}\$", "", formatted_formula)
            formatted_formula = re.sub(r"\^\{1([+-])\}", r"^{\1}", formatted_formula)

        else:
            formula = Formula(formula)
            # print(dir(formula))
            if formula == 'NiHO2':
                formula = Formula('NiOOH')
            else:
                formula = formula.reduce()[0]
            formatted_formula = f'{formula:latex}'
        return formatted_formula

    def apply_publication_style(self):
        plt.rcParams.update({
            'font.size': AREA_LABEL_FONT_SIZE,
            'axes.labelsize': AXIS_LABEL_FONT_SIZE,
            'axes.titlesize': AXIS_LABEL_FONT_SIZE,
            'xtick.labelsize': TICK_LABEL_FONT_SIZE,
            'ytick.labelsize': TICK_LABEL_FONT_SIZE,
            'legend.fontsize': AREA_LABEL_FONT_SIZE,
            'figure.titlesize': AXIS_LABEL_FONT_SIZE,
            'savefig.dpi': 300,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
        })

    def parse_formula_counts(self, formula):
        if formula in LIGAND_ELEMENT_COUNTS:
            return dict(LIGAND_ELEMENT_COUNTS[formula])

        counts = {}
        for element, number in re.findall(r'([A-Z][a-z]?)(\d*)', formula):
            counts[element] = counts.get(element, 0) + int(number or 1)
        return counts

    def get_formula_element_counts(self, formula):
        clean_formula = re.sub(r'\[[^\]]+\]', '', formula.replace('(aq)', ''))
        counts = {}

        for group, multiplier in re.findall(r'\(([A-Za-z0-9]+)\)(\d*)', clean_formula):
            multiplier = int(multiplier or 1)
            for element, count in self.parse_formula_counts(group).items():
                counts[element] = counts.get(element, 0) + count * multiplier

        clean_formula = re.sub(r'\([A-Za-z0-9]+\)\d*', '', clean_formula)
        for element, count in self.parse_formula_counts(clean_formula).items():
            counts[element] = counts.get(element, 0) + count

        return counts

    def get_product_composition(self, product):
        composition = getattr(product, 'composition', None)
        if composition is not None:
            return composition
        return self.parse_formula_counts(product.formula.split('[')[0])

    def classify_product(self, product):
        if product.phase == 'complex':
            return 'metal_ligand_complex'
        if product.phase == 'aqueous_ion':
            return 'aqueous_metal_ion'

        composition = self.get_product_composition(product)
        oxygen_count = composition.get('O', 0)
        hydrogen_count = composition.get('H', 0)
        if oxygen_count > 0 and hydrogen_count > 0:
            return 'metal_oxyhydroxide'
        if oxygen_count > 0:
            return 'metal_oxide'
        if hydrogen_count > 0:
            return 'metal_hydride'
        return 'metal'

    def estimate_metal_oxidation_state(self, product, category):
        composition = self.get_product_composition(product)
        metal_count = max(composition.get(self.data.metal, 0), 1)
        charge = composition.get('charge', 0)
        oxygen_count = composition.get('O', 0)
        hydrogen_count = composition.get('H', 0)

        if category == 'metal':
            return 0
        if category == 'metal_hydride':
            return (charge + hydrogen_count) / metal_count
        if oxygen_count > 0 or hydrogen_count > 0:
            return (charge + 2 * oxygen_count - hydrogen_count) / metal_count
        if category in ['aqueous_metal_ion', 'metal_ligand_complex']:
            return charge / metal_count
        return 0

    def color_intensity_for_product(self, product, category, used_intensities=None):
        element_counts = self.get_formula_element_counts(product.formula)
        metal_count = element_counts.get(self.data.metal, 0)
        other_count = sum(
            count for element, count in element_counts.items()
            if element != self.data.metal
        )

        if category == 'metal' and other_count == 0:
            intensity = 0.36
        elif category == 'metal_hydride':
            intensity = 0.42
        elif category == 'metal_oxide' and element_counts.get('O', 0) > 0:
            oxygen_count = element_counts['O']
            oxygen_per_metal = oxygen_count / max(metal_count, 1)
            intensity = 0.82 - 0.16 * oxygen_per_metal
        elif other_count > 0:
            metal_fraction = metal_count / (metal_count + other_count)
            intensity = 0.22 + 0.72 * metal_fraction
        else:
            oxidation_state = abs(self.estimate_metal_oxidation_state(product, category))
            intensity = 0.26 + 0.14 * oxidation_state

        if category == 'metal_oxyhydroxide':
            intensity += 0.08
        elif category == 'metal_ligand_complex':
            intensity += 0.10

        intensity = min(0.88, max(0.22, intensity))
        if used_intensities is None:
            return intensity

        if category == 'metal_oxide':
            used_intensities.append(intensity)
            return intensity

        while any(abs(intensity - used) < 0.055 for used in used_intensities):
            intensity += 0.065
            if intensity > 0.90:
                intensity = 0.25
        used_intensities.append(intensity)
        return intensity

    def get_color_metric_for_product(self, product, category):
        element_counts = self.get_formula_element_counts(product.formula)
        metal_count = element_counts.get(self.data.metal, 0)
        other_count = sum(
            count for element, count in element_counts.items()
            if element != self.data.metal
        )

        if category == 'metal_oxide' and element_counts.get('O', 0) > 0:
            oxygen_count = element_counts['O']
            oxygen_per_metal = oxygen_count / max(metal_count, 1)
            return f'O/{self.data.metal}', oxygen_per_metal

        if other_count > 0:
            metal_to_other = metal_count / other_count
            return f'{self.data.metal}/other', metal_to_other

        oxidation_state = self.estimate_metal_oxidation_state(product, category)
        return 'estimated oxidation state', oxidation_state

    def get_color_for_category(self, product, category, used_intensities=None):
        color_maps = {
            'metal': 'Greys',
            'metal_hydride': 'Greys',
            'metal_oxide': 'YlOrBr',
            'metal_oxyhydroxide': 'Oranges',
            'aqueous_metal_ion': 'PuBuGn',
            'metal_ligand_complex': 'RdPu',
        }
        color_map = plt.cm.get_cmap(color_maps.get(category, 'Greys'))
        intensity = self.color_intensity_for_product(product, category, used_intensities)
        return color_map(intensity), intensity
        
    def place_label_within_bounds(self, ax, x, y, label, rotation,color='k', tol=0.05,max_shift=0.2):
        rotation = 90 if abs(rotation) == 90 else 0
        x_min, x_max = min(self.pH_range), max(self.pH_range)
        y_min, y_max = min(self.V_range), max(self.V_range)

        ha = 'center'
        va = 'center'
        if x < x_min + 0.1 * (x_max - x_min):
            ha = 'left'
        elif x > x_max - 0.1 * (x_max - x_min):
            ha = 'right'
        if y < y_min + 0.1 * (y_max - y_min):
            va = 'bottom'
        elif y > y_max - 0.1 * (y_max - y_min):
            va = 'top'
        if label == 'Ag(OH)$_{2}$$^{-}$(aq)':
            x +=1
            fontsize = AREA_LABEL_FONT_SIZE
            return ax.text(x, y, label, ha=ha, va=va, color=color, rotation=rotation, fontsize=fontsize, zorder=5)
            
        if self.data.ligand_concentration['Gly'] != 0:
            if label == 'Ag$_{2}$O(s)':
                fontsize = AREA_LABEL_FONT_SIZE
                return ax.text(x, y, label, ha=ha, va=va, color=color, rotation=rotation, fontsize=fontsize, zorder=5)
            elif self.data.ligand_concentration['CN'] != 0 and label == 'AgO(s)' :
                fontsize = AREA_LABEL_FONT_SIZE
                x += 1
                return ax.text(x, y, label, ha=ha, va=va, color=color, rotation=rotation, fontsize=fontsize, zorder=5)
            elif label == 'Fe(Gly)$_{2}$$^{2+}$(aq)':
                fontsize = AREA_LABEL_FONT_SIZE
                x += 1
                return ax.text(x, y, label, ha=ha, va=va, color=color, rotation=rotation, fontsize=fontsize, zorder=5)
            elif label == 'FeO$_{2}$$^{2-}$(aq)':
                fontsize = AREA_LABEL_FONT_SIZE
                x += 0.3
                return ax.text(x, y, label, ha=ha, va=va, color=color, rotation=rotation, fontsize=fontsize, zorder=5)
            elif label == 'FeO(s)' or label == 'Fe$_{3}$O$_{4}$(s)':
                fontsize = AREA_LABEL_FONT_SIZE
                x += 1
                if label == 'FeO(s)':
                    y -= 0.1
                return ax.text(x, y, label, ha=ha, va=va, color=color, rotation=rotation, fontsize=fontsize, zorder=5)
            elif label == 'CuO$_{2}$$^{2-}$(aq)' or label == 'Cu$_{2}O$(s)' :
                fontsize = AREA_LABEL_FONT_SIZE
                x += 0.6
                return ax.text(x, y, label, ha=ha, va=va, color=color, rotation=rotation, fontsize=fontsize, zorder=5)

            else:
            
                return ax.text(x, y, label, ha=ha, va=va, color=color, rotation=rotation, fontsize=AREA_LABEL_FONT_SIZE, zorder=5)
        
        else:
            
            return ax.text(x, y, label, ha=ha, va=va, color=color, rotation=rotation, fontsize=AREA_LABEL_FONT_SIZE, zorder=5)

    def _label_extent(self, label, rotation=0):
        width = min(0.22, 0.006 * len(label) + 0.03)
        height = 0.035
        if abs(rotation) == 90:
            width, height = height, min(0.22, 0.006 * len(label) + 0.03)
        return width, height

    def _data_to_axes_fraction(self, x, y):
        x_min, x_max = min(self.pH_range), max(self.pH_range)
        y_min, y_max = min(self.V_range), max(self.V_range)
        return ((x - x_min) / (x_max - x_min), (y - y_min) / (y_max - y_min))

    def _axes_fraction_to_data(self, x_frac, y_frac):
        x_min, x_max = min(self.pH_range), max(self.pH_range)
        y_min, y_max = min(self.V_range), max(self.V_range)
        return (x_min + x_frac * (x_max - x_min), y_min + y_frac * (y_max - y_min))

    def _label_overlaps(self, candidate, placed_labels, pad=0.01):
        x, y, width, height = candidate
        for placed_x, placed_y, placed_width, placed_height in placed_labels:
            if (abs(x - placed_x) < (width + placed_width) / 2 + pad and
                    abs(y - placed_y) < (height + placed_height) / 2 + pad):
                return True
        return False

    def _region_needs_offset_label(self, pH_stable, eU_stable, label, rotation, placed_labels):
        pH_span = max(pH_stable) - min(pH_stable)
        eU_span = max(eU_stable) - min(eU_stable)
        pH_axis_span = max(self.pH_range) - min(self.pH_range)
        eU_axis_span = max(self.V_range) - min(self.V_range)
        thin_region = pH_span / pH_axis_span < 0.04 or eU_span / eU_axis_span < 0.04

        x_frac, y_frac = self._data_to_axes_fraction(np.mean(pH_stable), np.mean(eU_stable))
        width, height = self._label_extent(label, rotation)
        overlaps_existing_label = self._label_overlaps((x_frac, y_frac, width, height), placed_labels)
        return thin_region or overlaps_existing_label

    def _place_nonoverlapping_label(self, ax, x, y, label, rotation, placed_labels, force_offset=False, color='k'):
        rotation = 90 if abs(rotation) == 90 else 0
        x_frac, y_frac = self._data_to_axes_fraction(x, y)
        width, height = self._label_extent(label, rotation)
        candidate_offsets = [(0, 0)]
        if force_offset:
            candidate_offsets = []
        candidate_offsets += [
            (0.12, 0), (-0.12, 0), (0, 0.08), (0, -0.08),
            (0.18, 0), (-0.18, 0), (0, 0.13), (0, -0.13),
            (0.24, 0), (-0.24, 0), (0, 0.18), (0, -0.18),
        ]

        for dx, dy in candidate_offsets:
            label_x_frac = min(max(x_frac + dx, 0.03), 0.97)
            label_y_frac = min(max(y_frac + dy, 0.03), 0.97)
            candidate = (label_x_frac, label_y_frac, width, height)
            if not self._label_overlaps(candidate, placed_labels):
                label_x, label_y = self._axes_fraction_to_data(label_x_frac, label_y_frac)
                placed_labels.append(candidate)
                if dx == 0 and dy == 0:
                    return self.place_label_within_bounds(ax, label_x, label_y, label, rotation, color=color)
                return ax.annotate(
                    label,
                    xy=(x, y),
                    xytext=(label_x, label_y),
                    ha='center',
                    va='center',
                    color=color,
                    rotation=rotation,
                    fontsize=AREA_LABEL_FONT_SIZE,
                    zorder=5,
                    arrowprops=dict(arrowstyle='-', lw=0.6, color=color, shrinkA=2, shrinkB=2),
                )

        placed_labels.append((x_frac, y_frac, width, height))
        return self.place_label_within_bounds(ax, x, y, label, rotation, color=color)


    def count_species_by_category(self, stable_regions):
        totals = {
            'metal': 0,
            'metal_hydride': 0,
            'metal_oxide': 0,
            'metal_oxyhydroxide': 0,
            'aqueous_metal_ion': 0,
            'metal_ligand_complex': 0,
        }
        for i, (product, stable_indices) in enumerate(stable_regions.items()):
            if  np.any(stable_indices):
                totals[self.classify_product(product)] += 1
        return totals
    
        
    def save_stable_species(self, stable_regions):
        for i, (product, stable_indices) in enumerate(stable_regions.items()):
            if  np.any(stable_indices):
                print(product.formula, product.phase)
        
    
    def add_H2_O2_lines(self, ax, legend_elements):
        PREFAC = 0.0591
        xlim = self.pH_range
        ylim = self.V_range
        h_line = np.transpose([[xlim[0], -xlim[0] * PREFAC], [xlim[1], -xlim[1] * PREFAC]])
        o_line = np.transpose([[xlim[0], -xlim[0] * PREFAC + 1.23], [xlim[1], -xlim[1] * PREFAC + 1.23]])
        
        lw = WATER_LINE_WIDTH
        # Plot the hydrogen and oxygen lines on the axis
        h_line_plot, = ax.plot(
            h_line[0], h_line[1], color='blue', linestyle='--',
            linewidth=lw, label='Hydrogen Line', zorder=4
        )
        o_line_plot, = ax.plot(
            o_line[0], o_line[1], color='red', linestyle='--',
            linewidth=lw, label='Oxygen Line', zorder=4
        )
        legend_elements.append(h_line_plot)
        legend_elements.append(o_line_plot)
    
    def add_plot_accessories(self, ax, legend_elements, pH_exp_range=(11.5,13.5), V_exp_range=(-2, 2.3),rxn_box = True): 
        # V_exp_range is referenced to RHE
        # SHE = RHE - kB T ln(10) pH
        PREFAC = 0.0591
        box_left = pH_exp_range[0]
        box_right = pH_exp_range[1]
        
        
        V_left_bottom = V_exp_range[0] - PREFAC * box_left
        V_right_bottom = V_exp_range[0] - PREFAC * box_right
        V_left_top = V_exp_range[1] - PREFAC * box_left
        V_right_top = V_exp_range[1] - PREFAC * box_right
        
        lw = 1
        # Draw the four sides of the box as a function of pH
        if rxn_box:
            style = 'k-'
            bottom, = ax.plot([box_left, box_right], [V_left_bottom, V_right_bottom], style, lw=lw, label = f'Exp condition\nV vs RHE={V_exp_range[0]}-{V_exp_range[1]}\npH={pH_exp_range[0]}-{pH_exp_range[1]}', zorder=4)  # Bottom side (V as a function of pH)
            top, = ax.plot([box_left, box_right], [V_left_top, V_right_top], style, lw=lw, label = f'{V_exp_range[1]}V vs RHE', zorder=4)        # Top side (V as a function of pH)
            left, = ax.plot([box_left, box_left], [V_left_bottom, V_left_top], style, lw=lw, label = f'pH={pH_exp_range[0]}', zorder=4)       # Left side (fixed pH = box_left)
            right, = ax.plot([box_right, box_right], [V_right_bottom, V_right_top], style, lw=lw,label = f'pH={pH_exp_range[1]}', zorder=4)     # Right side (fixed pH = box_right)
            legend_elements.append(bottom)

        

    def compute_rotation(self,pH_stable, eU_stable):
        rotation = 0
        pH_range = max(pH_stable) - min(pH_stable)
        eU_range = max(eU_stable) - min(eU_stable)
        
        if pH_range/18 < eU_range/5:
            rotation = 90
        return rotation
        
    
    def plot_stable_regions(self, stable_regions, species_label_dict, rxn_box = True, show_legend = True):
        self.apply_publication_style()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlabel('pH')
        ax.set_ylabel(r'$E_{SHE}(V)$')
        ax.set_ylim(self.V_range)
        ax.set_xlim(self.pH_range)
        ax.yaxis.set_tick_params(which='both', direction='in', left=True, right=False)

        legend_elements = []
        placed_labels = []
        used_color_intensities = {}
        
        for i, (product, stable_indices) in enumerate(stable_regions.items()):
            
            if product.phase == 'complex':
                product_label = species_label_dict[product.formula]
                product_label = self.format_formula(product_label)
            else:
                product_label = self.format_formula(product.formula)
                
            pH_stable = self.pH_grid[stable_indices]
            eU_stable = self.V_grid[stable_indices]
            if len(pH_stable) > 0:
                category = self.classify_product(product)
                if product.phase == 'bulk': #'bulk', 'aqueous_ion','ligand','complex'
                    product_label+='(s)'

                if product.phase == 'aqueous_ion' or product.phase == 'complex':
                    if '(aq)' not in product_label:
                        product_label+='(aq)'
                color, color_intensity = self.get_color_for_category(
                    product,
                    category,
                    used_color_intensities.setdefault(category, []),
                )
                metric_label, metric_value = self.get_color_metric_for_product(product, category)
                print(
                    f"Color metric: {product.formula} | category={category} | "
                    f"{metric_label}={metric_value:.3f} | intensity={color_intensity:.3f}"
                )
                
                ax.scatter(
                    pH_stable, eU_stable, s=1, label=product_label,
                    color=color, alpha=REGION_ALPHA, rasterized=True,
                    edgecolors='none', zorder=1
                )
                
                legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                              markerfacecolor=color, markersize=10, label=product_label))

                centroid_pH = np.mean(pH_stable)
                centroid_eU = np.mean(eU_stable)
                
                rotation = self.compute_rotation(pH_stable, eU_stable)
                force_offset = self._region_needs_offset_label(
                    pH_stable, eU_stable, product_label, rotation, placed_labels
                )
                self._place_nonoverlapping_label(
                    ax, centroid_pH, centroid_eU, product_label, rotation,
                    placed_labels, force_offset=force_offset
                )
                
        self.add_H2_O2_lines(ax, legend_elements)
        self.add_plot_accessories(ax, legend_elements, rxn_box=rxn_box)
        
        if self.save_fig:
            activity = self.data.ion_activity
            output_dir = os.path.join("figures", self.dir)
            os.makedirs(output_dir, exist_ok=True)
            if self.filename == None:
                base_filename = f'{self.data.metal}-NH3-H2O_T={self.data.T}_activity={activity:.0e}_[NH3]={self.data.ligand_concentration["NH3"]}M_[Gly]={self.data.ligand_concentration["Gly"]}M_[CN]={self.data.ligand_concentration["CN"]}'
            elif self.filename == 'pdf':
                base_filename = f'{self.data.metal}-NH3-H2O_T={self.data.T}_activity={activity:.0e}_[NH3]={self.data.ligand_concentration["NH3"]}M_[Gly]={self.data.ligand_concentration["Gly"]}M_[CN]={self.data.ligand_concentration["CN"]}'
            else:
                base_filename = os.path.splitext(self.filename)[0]
            for extension in ['png', 'pdf']:
                output_path = f'{output_dir}/{base_filename}.{extension}'
                plt.savefig(output_path, bbox_inches='tight')
                print('saved figure to', output_path)
    
