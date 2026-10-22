from dataclasses import dataclass
import os
import re

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
try:
    from scipy import ndimage
except ModuleNotFoundError:
    ndimage = None
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
CM_PER_INCH = 2.54
RSC_DOUBLE_COLUMN_WIDTH_CM = 17.1
RSC_MAX_FIGURE_HEIGHT_CM = 23.3
COMPOSITE_COLUMNS = 3
COMPOSITE_ROWS = 2
COMPOSITE_COLUMN_GAP_CM = 0.3
COMPOSITE_ROW_GAP_CM = 0.3
RSC_PANEL_WIDTH_CM = (
    RSC_DOUBLE_COLUMN_WIDTH_CM
    - (COMPOSITE_COLUMNS - 1) * COMPOSITE_COLUMN_GAP_CM
) / COMPOSITE_COLUMNS


@dataclass(frozen=True)
class PublicationStyle:
    """Final-size typography and dimensions for an RSC six-panel figure.

    The exported panel is intended to be placed at 100% scale in Illustrator.
    RSC specifies a 17.1 cm double-column width, a maximum height of 23.3 cm,
    and requires text to remain legible at publication size. Seven-point text
    follows RSC's explicit Arial/Helvetica benchmark for chemical-structure
    labels. The 0.3 cm panel gaps are a project layout choice, not an RSC rule.
    """

    panel_width_cm: float = RSC_PANEL_WIDTH_CM
    panel_height_cm: float = RSC_PANEL_WIDTH_CM
    side_legend_width_cm: float = 5.2
    side_legend_row_height_cm: float = 0.42
    side_legend_vertical_padding_cm: float = 0.8
    axes_left_fraction: float = 0.20
    axes_bottom_fraction: float = 0.18
    axes_right_fraction: float = 0.97
    axes_top_fraction: float = 0.97
    font_family: tuple = ('Arial', 'Helvetica', 'Liberation Sans', 'DejaVu Sans')
    area_label_font_size: float = 7.0
    axis_label_font_size: float = 9 #8.0
    tick_label_font_size: float = 8 #7.0
    legend_label_font_size: float = 8 #7.0
    domain_number_font_size: float = 7.0
    region_alpha: float = 0.50
    phase_image_alpha: float = 0.95
    water_line_width: float = 0.8
    accessory_line_width: float = 0.6
    annotation_line_width: float = 0.5
    legend_marker_size: float = 4.5

    @property
    def figure_size_inches(self):
        """Return the final panel dimensions in Matplotlib's inch units."""
        return (
            self.panel_width_cm / CM_PER_INCH,
            self.panel_height_cm / CM_PER_INCH,
        )

    def legend_figure_size_inches(self, entry_count):
        """Return a canvas that fits the key without reducing the plot area."""
        legend_height_cm = max(
            self.panel_height_cm,
            entry_count * self.side_legend_row_height_cm
            + self.side_legend_vertical_padding_cm,
        )
        return (
            (self.panel_width_cm + self.side_legend_width_cm) / CM_PER_INCH,
            legend_height_cm / CM_PER_INCH,
        )


DEFAULT_PUBLICATION_STYLE = PublicationStyle()

LABEL_X_OFFSETS = {
    'Ni(OH)$_{4}$$^{2-}$(aq)': 0.5,
    'PdO(s)': -1.0,
    'AgO(s)': 1.6,
    'Cd(Gly)$^{+}$(aq)': -0.2,
    'Cd(Gly)$_{2}$(aq)': 0.2,
    'Fe(Gly)$^{+}$(aq)': -1.0,
    'Fe(Gly)$_{2}$(aq)': -0.2,
}
LIGAND_ELEMENT_COUNTS = {
    'NH3': {'N': 1, 'H': 3},
    'CN': {'C': 1, 'N': 1},
    'Gly': {'C': 2, 'H': 4, 'N': 1, 'O': 2},
    'NO2': {'N': 1, 'O': 2},
}

class GridPlotter:
    def __init__(
        self,
        pH_range,
        V_range,
        data,
        grid_size,
        save_fig,
        dir,
        filename,
        output_formats=('png',),
        dpi=600,
        style=None,
    ):
        self.dir = dir
        self.filename = filename
        self.output_formats = tuple(output_formats)
        self.dpi = dpi
        self.style = style or DEFAULT_PUBLICATION_STYLE
        self.phase_key = ()
        self.color_phase_key = ()
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
        """Return -log10 of free glycinate from the diprotic glycine balance."""
        Gly_tot = self.data.ligand_concentration['Gly']
        if Gly_tot <= 0:
            return self._large_pL_grid()
        return (
            -np.log10(Gly_tot)
            + np.log10(
                1
                + 10 ** (pKa2 - self.pH_grid)
                + 10 ** (pKa1 + pKa2 - 2 * self.pH_grid)
            )
        )
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
            'font.family': 'sans-serif',
            'font.sans-serif': list(self.style.font_family),
            'font.size': self.style.area_label_font_size,
            'axes.labelsize': self.style.axis_label_font_size,
            'axes.titlesize': self.style.axis_label_font_size,
            'xtick.labelsize': self.style.tick_label_font_size,
            'ytick.labelsize': self.style.tick_label_font_size,
            'legend.fontsize': self.style.legend_label_font_size,
            'figure.titlesize': self.style.axis_label_font_size,
            'mathtext.fontset': 'dejavusans',
            'savefig.dpi': self.dpi,
            'pdf.fonttype': 42,
            'ps.fonttype': 42,
            'svg.fonttype': 'none',
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
        color_map = plt.get_cmap(color_maps.get(category, 'Greys'))
        intensity = self.color_intensity_for_product(product, category, used_intensities)
        return color_map(intensity), intensity
        
    def place_label_within_bounds(self, ax, x, y, label, rotation, color='k'):
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
        x += LABEL_X_OFFSETS.get(label, 0.0)

        return ax.text(
            x, y, label, ha=ha, va=va, color=color, rotation=rotation,
            fontsize=self.style.area_label_font_size, zorder=5,
        )

    def _label_extent(self, label, rotation=0, font_size=None):
        font_size = font_size or self.style.area_label_font_size
        font_scale = font_size / self.style.area_label_font_size
        # Approximate rendered text bounds in axes fractions. Chemical labels
        # contain mathtext markup, so a conservative estimate is preferable
        # to allowing overlap in a final-width 5.5 cm panel.
        width = min(0.55, 0.015 * len(label) + 0.04)
        height = 0.05
        width *= font_scale
        height *= font_scale
        if abs(rotation) == 90:
            width, height = height, width
        return width, height

    def _data_to_axes_fraction(self, x, y):
        x_min, x_max = min(self.pH_range), max(self.pH_range)
        y_min, y_max = min(self.V_range), max(self.V_range)
        return ((x - x_min) / (x_max - x_min), (y - y_min) / (y_max - y_min))

    def _axes_fraction_to_data(self, x_frac, y_frac):
        x_min, x_max = min(self.pH_range), max(self.pH_range)
        y_min, y_max = min(self.V_range), max(self.V_range)
        return (x_min + x_frac * (x_max - x_min), y_min + y_frac * (y_max - y_min))

    def _label_overlaps(self, candidate, placed_labels, pad=0.002):
        x, y, width, height = candidate
        for placed_x, placed_y, placed_width, placed_height in placed_labels:
            if (abs(x - placed_x) < (width + placed_width) / 2 + pad and
                    abs(y - placed_y) < (height + placed_height) / 2 + pad):
                return True
        return False

    def _draw_label_text(self, ax, x, y, label, rotation, color='k', font_size=None):
        font_size = font_size or self.style.area_label_font_size
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

        return ax.text(
            x, y, label, ha=ha, va=va, color=color, rotation=rotation,
            fontsize=font_size, zorder=5
        )

    def _largest_connected_region_mask(self, stable_indices):
        if ndimage is None:
            return stable_indices

        component_labels, num_components = ndimage.label(stable_indices)
        if num_components <= 1:
            return stable_indices

        component_sizes = np.bincount(component_labels.ravel())
        component_sizes[0] = 0
        largest_component = int(np.argmax(component_sizes))
        return component_labels == largest_component

    def _region_points(self, region_mask):
        return self.pH_grid[region_mask], self.V_grid[region_mask]

    def _choose_region_label_anchor(self, region_mask):
        pH_stable, eU_stable = self._region_points(region_mask)
        if ndimage is not None:
            distance_to_boundary = ndimage.distance_transform_edt(region_mask)
            anchor_row, anchor_col = np.unravel_index(
                int(np.argmax(distance_to_boundary)),
                distance_to_boundary.shape,
            )
            return self.pH_values[anchor_col], self.V_values[anchor_row]

        target_pH = 0.5 * (min(pH_stable) + max(pH_stable))
        target_eU = 0.5 * (min(eU_stable) + max(eU_stable))
        pH_span = max(max(pH_stable) - min(pH_stable), 1e-12)
        eU_span = max(max(eU_stable) - min(eU_stable), 1e-12)
        distance = (
            ((pH_stable - target_pH) / pH_span) ** 2
            + ((eU_stable - target_eU) / eU_span) ** 2
        )
        anchor_idx = int(np.argmin(distance))
        return pH_stable[anchor_idx], eU_stable[anchor_idx]

    def _region_label_candidates(self, region_mask, max_candidates=80):
        pH_stable, eU_stable = self._region_points(region_mask)
        if ndimage is not None:
            distance_to_boundary = ndimage.distance_transform_edt(region_mask)
            candidate_indices = np.argsort(distance_to_boundary.ravel())[::-1]
            yielded = 0
            for flat_idx in candidate_indices:
                row, col = np.unravel_index(int(flat_idx), distance_to_boundary.shape)
                if not region_mask[row, col]:
                    continue
                yield self.pH_values[col], self.V_values[row]
                yielded += 1
                if yielded >= max_candidates:
                    return

        pH_span = max(max(pH_stable) - min(pH_stable), 1e-12)
        eU_span = max(max(eU_stable) - min(eU_stable), 1e-12)
        target_pH = 0.5 * (min(pH_stable) + max(pH_stable))
        target_eU = 0.5 * (min(eU_stable) + max(eU_stable))
        distance = (
            ((pH_stable - target_pH) / pH_span) ** 2
            + ((eU_stable - target_eU) / eU_span) ** 2
        )
        candidate_indices = list(np.argsort(distance)[:max_candidates])

        seen = set()
        for idx in candidate_indices:
            if idx in seen:
                continue
            seen.add(idx)
            yield pH_stable[idx], eU_stable[idx]

    def _place_direct_label(self, ax, region_mask, label, rotation, color='k'):
        x, y = self._choose_region_label_anchor(region_mask)
        return self._draw_label_text(ax, x, y, label, rotation, color=color)

    # def _place_nonoverlapping_label_in_region(
    #     self, ax, region_mask, label, rotation, placed_labels, color='k'
    # ):
    #     rotation = 90 if abs(rotation) == 90 else 0
    #     width, height = self._label_extent(label, rotation)

    #     for x, y in self._region_label_candidates(region_mask):
    #         x_frac, y_frac = self._data_to_axes_fraction(x, y)
    #         candidate = (x_frac, y_frac, width, height)
    #         if not self._label_overlaps(candidate, placed_labels):
    #             placed_labels.append(candidate)
    #             return self._draw_label_text(ax, x, y, label, rotation, color=color)

    #     x, y = self._choose_region_label_anchor(region_mask)
    #     x_frac, y_frac = self._data_to_axes_fraction(x, y)
    #     placed_labels.append((x_frac, y_frac, width, height))
    #     return self._draw_label_text(ax, x, y, label, rotation, color=color)

    def _region_label_placement_flags(self, pH_stable, eU_stable, label, rotation, placed_labels):
        pH_span = max(pH_stable) - min(pH_stable)
        eU_span = max(eU_stable) - min(eU_stable)
        pH_axis_span = max(self.pH_range) - min(self.pH_range)
        eU_axis_span = max(self.V_range) - min(self.V_range)
        thin_region = pH_span / pH_axis_span < 0.04 or eU_span / eU_axis_span < 0.04

        x_frac, y_frac = self._data_to_axes_fraction(np.mean(pH_stable), np.mean(eU_stable))
        width, height = self._label_extent(label, rotation)
        overlaps_existing_label = self._label_overlaps((x_frac, y_frac, width, height), placed_labels)
        return thin_region or overlaps_existing_label, overlaps_existing_label

    def _label_indicator_props(self, label_indicator, color):
        if label_indicator is None or label_indicator == 'none':
            return None
        arrowstyle_by_indicator = {
            'line': '-',
            'arrow': '->',
        }
        if label_indicator not in arrowstyle_by_indicator:
            raise ValueError("label_indicator must be 'line', 'arrow', 'none', or None.")
        return dict(
            arrowstyle=arrowstyle_by_indicator[label_indicator],
            lw=self.style.annotation_line_width,
            color=color,
            shrinkA=2,
            shrinkB=2,
        )

    def _place_nonoverlapping_label(
        self, ax, x, y, label, rotation, placed_labels, force_offset=False,
        color='k', label_indicator='line', font_size=None
    ):
        font_size = font_size or self.style.area_label_font_size
        rotation = 90 if abs(rotation) == 90 else 0
        x_frac, y_frac = self._data_to_axes_fraction(x, y)
        width, height = self._label_extent(label, rotation, font_size=font_size)
        candidate_offsets = [(0, 0)]
        if force_offset:
            candidate_offsets = []
        candidate_offsets += [
            (0, 0.08), (0, -0.08), (0.15, 0), (-0.15, 0),
            (0.15, 0.10), (-0.15, 0.10), (0.15, -0.10), (-0.15, -0.10),
            (0, 0.16), (0, -0.16), (0.25, 0), (-0.25, 0),
            (0.25, 0.18), (-0.25, 0.18), (0.25, -0.18), (-0.25, -0.18),
            (0, 0.26), (0, -0.26), (0.36, 0), (-0.36, 0),
        ]

        for dx, dy in candidate_offsets:
            margin = 0.01
            label_x_frac = min(
                max(x_frac + dx, width / 2 + margin),
                1 - width / 2 - margin,
            )
            label_y_frac = min(
                max(y_frac + dy, height / 2 + margin),
                1 - height / 2 - margin,
            )
            candidate = (label_x_frac, label_y_frac, width, height)
            if not self._label_overlaps(candidate, placed_labels):
                label_x, label_y = self._axes_fraction_to_data(label_x_frac, label_y_frac)
                placed_labels.append(candidate)
                if dx == 0 and dy == 0:
                    return self._draw_label_text(
                        ax, label_x, label_y, label, rotation, color=color, font_size=font_size
                    )
                return ax.annotate(
                    label,
                    xy=(x, y),
                    xytext=(label_x, label_y),
                    ha='center',
                    va='center',
                    color=color,
                    rotation=rotation,
                    fontsize=font_size,
                    zorder=5,
                    arrowprops=self._label_indicator_props(label_indicator, color),
                )

        # If every non-overlapping candidate is occupied, keep the label fully
        # inside the axes and accept the smallest local overlap.
        label_x_frac = min(max(x_frac, width / 2 + margin), 1 - width / 2 - margin)
        label_y_frac = min(max(y_frac, height / 2 + margin), 1 - height / 2 - margin)
        label_x, label_y = self._axes_fraction_to_data(label_x_frac, label_y_frac)
        placed_labels.append((label_x_frac, label_y_frac, width, height))
        return self._draw_label_text(
            ax, label_x, label_y, label, rotation, color=color, font_size=font_size
        )

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
        
        lw = self.style.water_line_width
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
    
    def add_plot_accessories(self, ax, legend_elements, pH_exp_range=(11.5,13.5), V_exp_range=(-2, 2.0),rxn_box = True):
        # V_exp_range is referenced to RHE
        # SHE = RHE - kB T ln(10) pH
        PREFAC = 0.0591
        box_left = pH_exp_range[0]
        box_right = pH_exp_range[1]
        
        
        V_left_bottom = V_exp_range[0] - PREFAC * box_left
        V_right_bottom = V_exp_range[0] - PREFAC * box_right
        V_left_top = V_exp_range[1] - PREFAC * box_left
        V_right_top = V_exp_range[1] - PREFAC * box_right
        
        lw = self.style.accessory_line_width
        # Draw the four sides of the box as a function of pH
        if rxn_box:
            style = 'k-'
            bottom, = ax.plot([box_left, box_right], [V_left_bottom, V_right_bottom], style, lw=lw, label = f'Exp condition\nV vs RHE={V_exp_range[0]}-{V_exp_range[1]}\npH={pH_exp_range[0]}-{pH_exp_range[1]}', zorder=4)  # Bottom side (V as a function of pH)
            top, = ax.plot([box_left, box_right], [V_left_top, V_right_top], style, lw=lw, label = f'{V_exp_range[1]}V vs RHE', zorder=4)        # Top side (V as a function of pH)
            left, = ax.plot([box_left, box_left], [V_left_bottom, V_left_top], style, lw=lw, label = f'pH={pH_exp_range[0]}', zorder=4)       # Left side (fixed pH = box_left)
            right, = ax.plot([box_right, box_right], [V_right_bottom, V_right_top], style, lw=lw,label = f'pH={pH_exp_range[1]}', zorder=4)     # Right side (fixed pH = box_right)
            legend_elements.append(bottom)

        

    def compute_rotation(self, pH_stable, eU_stable):
        rotation = 0
        pH_fraction = (
            max(pH_stable) - min(pH_stable)
        ) / (max(self.pH_range) - min(self.pH_range))
        potential_fraction = (
            max(eU_stable) - min(eU_stable)
        ) / (max(self.V_range) - min(self.V_range))

        if pH_fraction < potential_fraction:
            rotation = 90
        return rotation

    def _region_label_position(self, region_mask):
        """Return an interior point near the largest connected region's centroid."""
        component = self._largest_connected_region_mask(region_mask)
        coordinates = np.argwhere(component)
        centroid = coordinates.mean(axis=0)
        nearest = coordinates[np.argmin(np.sum((coordinates - centroid) ** 2, axis=1))]
        row, column = nearest
        return self.pH_values[column], self.V_values[row]

    def _spread_numbered_markers(self, ax, domains):
        """Keep numbered circles inside their regions and clear of other circles."""
        marker_radius_px = self.style.domain_number_font_size * ax.figure.dpi / 72
        minimum_spacing_px = 2.4 * marker_radius_px
        bounds = ax.get_window_extent()
        placed_pixels = []
        positions = []
        for x, y, _, _, region_mask in domains:
            original_pixel = ax.transData.transform((x, y))
            component = self._largest_connected_region_mask(region_mask)
            coordinates = np.argwhere(component)
            coordinates = coordinates[::max(1, len(coordinates) // 3000)]
            candidates = np.column_stack((
                self.pH_values[coordinates[:, 1]],
                self.V_values[coordinates[:, 0]],
            ))
            pixels = ax.transData.transform(candidates)
            inside = (
                (pixels[:, 0] >= bounds.x0 + marker_radius_px)
                & (pixels[:, 0] <= bounds.x1 - marker_radius_px)
                & (pixels[:, 1] >= bounds.y0 + marker_radius_px)
                & (pixels[:, 1] <= bounds.y1 - marker_radius_px)
            )
            if np.any(inside):
                candidates, pixels = candidates[inside], pixels[inside]
            distance_from_centroid = np.linalg.norm(pixels - original_pixel, axis=1)
            if placed_pixels:
                clearance = np.min(
                    np.linalg.norm(
                        pixels[:, None, :] - np.asarray(placed_pixels)[None, :, :],
                        axis=2,
                    ),
                    axis=1,
                )
                clear = clearance >= minimum_spacing_px
                if np.any(clear):
                    choice = np.argmin(np.where(clear, distance_from_centroid, np.inf))
                else:
                    choice = np.argmax(clearance - 0.05 * distance_from_centroid)
            else:
                choice = np.argmin(distance_from_centroid)
            positions.append(tuple(candidates[choice]))
            placed_pixels.append(pixels[choice])
        return positions
        
    
    def plot_stable_regions(
        self, stable_regions, species_label_dict, rxn_box=True, show_legend=True,
        avoid_label_overlap=False, label_indicator='none', overlap_label_font_size=None,
        pH_exp_range=(11.5, 13.5), V_exp_range=(-2, 2.0),
        label_style='direct',
        ax=None,
        legend_category_order=None,
    ):
        """Draw stable domains, optionally on an axis supplied by a composite figure."""
        if label_style not in {'direct', 'numbered', 'legend'}:
            raise ValueError("label_style must be 'direct', 'numbered', or 'legend'")
        self.apply_publication_style()
        has_side_legend = show_legend and label_style in {'numbered', 'legend'}
        if ax is None:
            legend_entry_count = sum(
                1 for stable_indices in stable_regions.values() if np.any(stable_indices)
            )
            figure_size = (
                self.style.legend_figure_size_inches(legend_entry_count)
                if has_side_legend
                else self.style.figure_size_inches
            )
            fig, ax = plt.subplots(figsize=figure_size)
            if has_side_legend:
                total_width_cm = self.style.panel_width_cm + self.style.side_legend_width_cm
                total_height_cm = figure_size[1] * CM_PER_INCH
                extra_height_cm = total_height_cm - self.style.panel_height_cm
                vertical_offset_cm = extra_height_cm / 2
                # Preserve the same axes dimensions as the 5.5 cm direct-label
                # panel; additional canvas width and height belong to the key.
                fig.subplots_adjust(
                    left=(
                        self.style.axes_left_fraction * self.style.panel_width_cm
                    ) / total_width_cm,
                    bottom=(
                        self.style.axes_bottom_fraction * self.style.panel_height_cm
                        + vertical_offset_cm
                    ) / total_height_cm,
                    right=(
                        self.style.axes_right_fraction * self.style.panel_width_cm
                    ) / total_width_cm,
                    top=(
                        self.style.axes_top_fraction * self.style.panel_height_cm
                        + vertical_offset_cm
                    ) / total_height_cm,
                )
            else:
                fig.subplots_adjust(
                    left=self.style.axes_left_fraction,
                    bottom=self.style.axes_bottom_fraction,
                    right=self.style.axes_right_fraction,
                    top=self.style.axes_top_fraction,
                )
        else:
            fig = ax.figure
        ax.set_xlabel('pH')
        ax.set_ylabel(r'$E_{\mathrm{SHE}}$ (V)')
        ax.set_ylim(self.V_range)
        ax.set_xlim(self.pH_range)
        ax.xaxis.set_tick_params(which='both', direction='in', bottom=True, top=False)
        ax.yaxis.set_tick_params(which='both', direction='in', left=True, right=False)

        legend_elements = []
        placed_labels = []
        used_color_intensities = {}
        numbered_domains = []
        color_legend_domains = []
        phase_image = np.zeros((*self.pH_grid.shape, 4), dtype=np.float32)
        
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
                
                phase_image[stable_indices, :3] = color[:3]
                phase_image[stable_indices, 3] = (
                    self.style.phase_image_alpha
                    if label_style in {'numbered', 'legend'}
                    else self.style.region_alpha
                )
                
                legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                              markerfacecolor=color,
                                              markersize=self.style.legend_marker_size,
                                              label=product_label))

                if label_style == 'numbered':
                    x, y = self._region_label_position(stable_indices)
                    numbered_domains.append((x, y, product_label, category, stable_indices))
                    continue
                if label_style == 'legend':
                    color_legend_domains.append((np.mean(eU_stable), product_label, category, color))
                    continue

                centroid_pH = np.mean(pH_stable)
                centroid_eU = np.mean(eU_stable)

                rotation = self.compute_rotation(pH_stable, eU_stable)
                if avoid_label_overlap:
                    force_offset, overlaps_existing_label = self._region_label_placement_flags(
                        pH_stable, eU_stable, product_label, rotation, placed_labels
                    )
                    label_font_size = (
                        overlap_label_font_size
                        if overlaps_existing_label and overlap_label_font_size is not None
                        else self.style.area_label_font_size
                    )
                    self._place_nonoverlapping_label(
                        ax, centroid_pH, centroid_eU, product_label, rotation,
                        placed_labels, force_offset=force_offset, label_indicator=label_indicator,
                        font_size=label_font_size
                    )
                else:
                    self.place_label_within_bounds(
                        ax, centroid_pH, centroid_eU, product_label, rotation
                    )
                
        ax.imshow(
            phase_image,
            origin='lower',
            extent=(*self.pH_range, *self.V_range),
            aspect='auto', interpolation='nearest', zorder=1,
        )
        if label_style in {'numbered', 'legend'}:
            category_order = {
                'metal_oxide': 0,
                'metal_oxyhydroxide': 0,
                'aqueous_metal_ion': 1,
                'metal_ligand_complex': 1,
                'metal': 2,
                'metal_hydride': 3,
            }
            if label_style == 'numbered':
                numbered_domains.sort(
                    key=lambda domain: (category_order[domain[3]], -domain[1], domain[0])
                )
                self.phase_key = tuple(domain[2] for domain in numbered_domains)
                self.color_phase_key = ()
                marker_positions = self._spread_numbered_markers(ax, numbered_domains)
                for number, (x, y) in enumerate(marker_positions, start=1):
                    ax.text(
                        x, y, str(number), ha='center', va='center',
                        fontsize=self.style.domain_number_font_size,
                        fontweight='bold', color='black', zorder=6,
                        bbox={'boxstyle': 'circle,pad=0.18', 'facecolor': 'none',
                              'edgecolor': 'black', 'linewidth': 0.8},
                    )
            else:
                self.phase_key = ()
                legend_order = (
                    category_order if legend_category_order is None
                    else legend_category_order
                )
                color_legend_domains.sort(
                    key=lambda domain: (legend_order[domain[2]], -domain[0], domain[1])
                )
                self.color_phase_key = tuple(
                    (domain[1], domain[3]) for domain in color_legend_domains
                )
        else:
            self.phase_key = ()
            self.color_phase_key = ()

        self.add_H2_O2_lines(ax, legend_elements)
        self.add_plot_accessories(
            ax,
            legend_elements,
            pH_exp_range=pH_exp_range,
            V_exp_range=V_exp_range,
            rxn_box=rxn_box,
        )

        if label_style == 'numbered' and show_legend:
            row_step = min(0.052, 0.90 / max(len(numbered_domains), 1))
            first_row_y = 0.5 + (len(numbered_domains) - 1) * row_step / 2
            for row, (_, _, phase_label, _, _) in enumerate(numbered_domains):
                row_y = first_row_y - row * row_step
                ax.text(
                    1.045, row_y, str(row + 1),
                    transform=ax.transAxes, ha='center', va='center',
                    fontsize=self.style.domain_number_font_size,
                    fontweight='bold', clip_on=False,
                    bbox={'boxstyle': 'circle,pad=0.25', 'facecolor': 'none',
                          'edgecolor': 'black', 'linewidth': 0.8},
                )
                ax.text(
                    1.09, row_y, phase_label, transform=ax.transAxes,
                    ha='left', va='center',
                    fontsize=self.style.legend_label_font_size, clip_on=False,
                )
        elif label_style == 'legend' and show_legend:
            handles = [
                Line2D(
                    [0], [0], marker='o', linestyle='None',
                    markerfacecolor=tuple(
                        self.style.phase_image_alpha * channel
                        + (1 - self.style.phase_image_alpha)
                        for channel in color[:3]
                    ),
                    markeredgecolor='none',
                    markersize=self.style.legend_marker_size, label=phase_label,
                )
                for _, phase_label, _, color in color_legend_domains
            ]
            ax.legend(
                handles=handles,
                loc='center left', bbox_to_anchor=(1.02, 0.5),
                frameon=False, fontsize=self.style.legend_label_font_size,
                labelspacing=0.35,
            )
        
        if self.save_fig:
            activity = self.data.ion_activity
            output_dir = self.dir if os.path.isabs(self.dir) else os.path.join("figures", self.dir)
            os.makedirs(output_dir, exist_ok=True)
            if self.filename == None:
                base_filename = f'{self.data.metal}-NH3-H2O_T={self.data.T}_activity={activity:.0e}_[NH3]={self.data.ligand_concentration["NH3"]}M_[Gly]={self.data.ligand_concentration["Gly"]}M_[CN]={self.data.ligand_concentration["CN"]}'
            elif self.filename == 'pdf':
                base_filename = f'{self.data.metal}-NH3-H2O_T={self.data.T}_activity={activity:.0e}_[NH3]={self.data.ligand_concentration["NH3"]}M_[Gly]={self.data.ligand_concentration["Gly"]}M_[CN]={self.data.ligand_concentration["CN"]}'
            else:
                base_filename = self.filename
                if os.path.splitext(base_filename)[1].lower() in {'.png', '.svg', '.pdf', '.tif', '.tiff'}:
                    base_filename = os.path.splitext(base_filename)[0]
            for extension in self.output_formats:
                extension = extension.lstrip('.').lower()
                output_path = f'{output_dir}/{base_filename}.{extension}'
                save_format = 'tiff' if extension in {'tif', 'tiff'} else extension
                fig.savefig(output_path, format=save_format, dpi=self.dpi)
                print('saved figure to', output_path)

            plt.close(fig)
        return ax
    
