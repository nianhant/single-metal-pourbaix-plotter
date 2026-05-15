# Single-Metal Pourbaix Plotter

Generate one paper-release Pourbaix diagram for one metal and one condition.

```bash
python examples/generate_pourbaix.py
```

The release example currently generates a Ni diagram at 298.15 K with:

- metal activity: `1e-4`
- ligand concentrations: `NH3=0.02 M`, `Gly=0.005 M`, `CN=0 M`

Outputs are written under `figures/pourbaix_diagrams/Ni/`.
