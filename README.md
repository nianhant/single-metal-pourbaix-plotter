# Single-Metal Pourbaix Plotter

Generate one paper-release Pourbaix diagram for one metal and one condition.

```bash
python examples/generate_pourbaix.py
```

To override one setting without editing the script:

```bash
python examples/generate_pourbaix.py --activity 1e-5 --nh3 0.02 --gly 0.005 --cn 0
```

The release example currently generates a Ni diagram at 298.15 K with:

- metal activity: `1e-4`
- ligand concentrations: `NH3=0.02 M`, `NO2=0 M`, `Gly=0.005 M`, `CN=0 M`

Outputs are written under `figures/pourbaix_diagrams/Ni/`.

The script first uses cached local formation-energy JSON files in `data/`. If
`data/<metal>_solid_formation_energy.json` or
`data/<metal>_ion_formation_energy.json` is missing, it fetches the missing
data from Materials Project and saves it back to `data/` for future runs. Set
the key outside the script:

```bash
export MP_API_KEY=your_key_here
```
