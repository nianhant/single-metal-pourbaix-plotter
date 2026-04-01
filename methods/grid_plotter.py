import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import matplotlib.ticker as ticker
from ase.formula import Formula
import re
import os

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
            formula = formula.reduce()[0]
            formatted_formula = f'{formula:latex}'
        return formatted_formula

    def get_color_for_label(self,label, i, total_species):        
        warmer_color_map = plt.cm.get_cmap('summer')  # Warmer colors
        cooler_color_map = plt.cm.get_cmap('cool')  # Cooler colors
        normalized_index = i / total_species
        if '(s)' in label:
            return warmer_color_map(normalized_index)
        elif '(aq)' in label:
            return cooler_color_map(normalized_index)
        else:
            return 'gray'
        
    def place_label_within_bounds(self, ax, x, y, label, rotation,color='k', tol=0.05,max_shift=0.2):
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
            rotation = 45
            x +=1
            fontsize = 10
            ax.text(x, y, label, ha=ha, va=va, color=color, rotation=rotation,fontsize=fontsize)
            return
            
        if self.data.ligand_concentration['Gly'] != 0:
            if label == 'Ag$_{2}$O(s)':
                fontsize = 10 
                ax.text(x, y, label, ha=ha, va=va, color=color, rotation=rotation,fontsize=fontsize)
            elif self.data.ligand_concentration['CN'] != 0 and label == 'AgO(s)' :
                fontsize = 10 
                x += 1
                ax.text(x, y, label, ha=ha, va=va, color=color, rotation=-3.39)
            elif label == 'Fe(Gly)$_{2}$$^{2+}$(aq)':
                fontsize = 12
                x += 1
                ax.text(x, y, label, ha=ha, va=va, color=color, rotation=rotation,fontsize=fontsize)
            elif label == 'FeO$_{2}$$^{2-}$(aq)':
                fontsize = 12
                x += 0.3
                ax.text(x, y, label, ha=ha, va=va, color=color, rotation=rotation,fontsize=fontsize)
            elif label == 'FeO(s)' or label == 'Fe$_{3}$O$_{4}$(s)':
                fontsize = 12
                x += 1
                if label == 'FeO(s)':
                    y -= 0.1
                ax.text(x, y, label, ha=ha, va=va, color=color, rotation=-3.39,fontsize=10)
            elif label == 'CuO$_{2}$$^{2-}$(aq)' or label == 'Cu$_{2}O$(s)' :
                fontsize = 13
                x += 0.6
                ax.text(x, y, label, ha=ha, va=va, color=color, rotation=rotation,fontsize=fontsize)

            else:
            
                ax.text(x, y, label, ha=ha, va=va, color=color, rotation=rotation)
        
        else:
            
            ax.text(x, y, label, ha=ha, va=va, color=color, rotation=rotation)


    def count_solid_species(self, stable_regions):
        total_solid = 0
        total_aq = 0
        for i, (product, stable_indices) in enumerate(stable_regions.items()):
            if  np.any(stable_indices):
                if product.phase == 'bulk':
                    total_solid += 1
                if product.phase == 'aqueous_ion' or product.phase == 'complex':
                    total_aq += 1
        return total_solid, total_aq
    
        
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
        
        lw = 1
        # Plot the hydrogen and oxygen lines on the axis
        h_line_plot, = ax.plot(h_line[0], h_line[1], "b--", linewidth=lw, label='Hydrogen Line')
        o_line_plot, = ax.plot(o_line[0], o_line[1], "r--", linewidth=lw, label='Oxygen Line')
        legend_elements.append(h_line_plot)
        legend_elements.append(o_line_plot)
    
    def add_plot_accessories(self, ax, legend_elements, pH_exp_range=(11.5,13.5), V_exp_range=(-2, 2.3), ):
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
        style = 'g-'
        bottom, = ax.plot([box_left, box_right], [V_left_bottom, V_right_bottom], style, lw=lw, label = f'Exp condition\nV vs RHE={V_exp_range[0]}-{V_exp_range[1]}\npH={pH_exp_range[0]}-{pH_exp_range[1]}')  # Bottom side (V as a function of pH)
        top, = ax.plot([box_left, box_right], [V_left_top, V_right_top], style, lw=lw, label = f'{V_exp_range[1]}V vs RHE')        # Top side (V as a function of pH)
        left, = ax.plot([box_left, box_left], [V_left_bottom, V_left_top], style, lw=lw, label = f'pH={pH_exp_range[0]}')       # Left side (fixed pH = box_left)
        right, = ax.plot([box_right, box_right], [V_right_bottom, V_right_top], style, lw=lw,label = f'pH={pH_exp_range[1]}')     # Right side (fixed pH = box_right)
        legend_elements.append(bottom)

        

    def compute_rotation(self,pH_stable, eU_stable):
        rotation = 0
        pH_range = max(pH_stable) - min(pH_stable)
        eU_range = max(eU_stable) - min(eU_stable)
        
        if pH_range/18 < eU_range/5:
            rotation = 90
        return rotation
        
    
    def plot_stable_regions(self, stable_regions, species_label_dict):
        total_solid, total_aq = self.count_solid_species( stable_regions)
#         self.save_stable_species(stable_regions)

        fig, ax = plt.subplots(figsize=(8, 8))
        ax.set_xlabel('pH')
        ax.set_ylabel(r'$E_{SHE}(V)$')
        ax.set_ylim(self.V_range)
        ax.set_xlim(self.pH_range)
        ax.yaxis.set_tick_params(which='both', direction='in', left=True, right=True)

        legend_elements = []
        
        solid_index = 0
        aq_index = 0
        for i, (product, stable_indices) in enumerate(stable_regions.items()):
            
            if product.phase == 'complex':
                product_label = species_label_dict[product.formula]
                product_label = self.format_formula(product_label)
            else:
                product_label = self.format_formula(product.formula)
                
            pH_stable = self.pH_grid[stable_indices]
            eU_stable = self.V_grid[stable_indices]
            if len(pH_stable) > 0:
                if product.phase == 'bulk': #'bulk', 'aqueous_ion','ligand','complex'
                    solid_index += 1
                    product_label+='(s)'
                    color = self.get_color_for_label(product_label, solid_index, total_solid)

                if product.phase == 'aqueous_ion' or product.phase == 'complex':
                    aq_index += 1
                    if '(aq)' not in product_label:
                        product_label+='(aq)'
                    color = self.get_color_for_label(product_label, aq_index, total_aq)
                
                ax.scatter(pH_stable, eU_stable, s=1, label=product_label, color=color, alpha=1)
                
                legend_elements.append(Line2D([0], [0], marker='o', color='w', 
                                              markerfacecolor=color, markersize=10, label=product_label))

                centroid_pH = np.mean(pH_stable)
                centroid_eU = np.mean(eU_stable)
                
                rotation = self.compute_rotation(pH_stable, eU_stable)
                
                
                
                self.place_label_within_bounds(ax, centroid_pH, centroid_eU, product_label,rotation)
                
        self.add_H2_O2_lines(ax, legend_elements)
        self.add_plot_accessories(ax, legend_elements)
        
        ax.legend(handles=legend_elements,loc='center left', bbox_to_anchor=(1.05, 0.58), borderaxespad=0.)
        
    
        if self.save_fig:
            activity = self.data.ion_activity
            output_dir = os.path.join("figures", self.dir)
            os.makedirs(output_dir, exist_ok=True)
            if self.filename == None:
                self.filename = f'{output_dir}/{self.data.metal}-NH3-H2O_T={self.data.T}_activity={activity:.0e}_[NH3]={self.data.ligand_concentration["NH3"]}M_[Gly]={self.data.ligand_concentration["Gly"]}M_[CN]={self.data.ligand_concentration["CN"]}.png'
            else:
                self.filename = f'{output_dir}/{self.filename}'
            plt.savefig(self.filename, bbox_inches='tight')
            print('saved figure to', self.filename)
    