"""Checks for the free-glycinate activity used in Pourbaix diagrams."""

from types import SimpleNamespace
import unittest

import numpy as np

from methods.grid_plotter import GridPlotter


class GlycineBalanceTests(unittest.TestCase):
    """Verify the free-glycinate grid at several acid-base conditions."""

    def test_glycinate_balance_across_ph(self) -> None:
        """Free glycinate obeys both dissociations and total glycine balance."""
        ph = np.array([2.35, 7.0, 9.0, 9.78, 10.5, 13.5])
        total_glycine = 0.1
        plotter = object.__new__(GridPlotter)
        plotter.data = SimpleNamespace(ligand_concentration={"Gly": total_glycine})
        plotter.pH_grid = ph

        glycinate = 10 ** (-plotter.generate_pGly_grid())
        zwitterion = glycinate * 10 ** (9.78 - ph)
        glycinium = zwitterion * 10 ** (2.35 - ph)

        np.testing.assert_allclose(glycinate + zwitterion + glycinium, total_glycine)
        np.testing.assert_allclose(glycinate[3], total_glycine / 2, rtol=1e-7)
        np.testing.assert_allclose(glycinate[4], 0.084, rtol=2e-4)

    def test_no_glycine_yields_large_pgly(self) -> None:
        """A zero-total-glycine condition suppresses free ligand activity."""
        plotter = object.__new__(GridPlotter)
        plotter.data = SimpleNamespace(ligand_concentration={"Gly": 0.0})
        plotter.pH_grid = np.array([7.0, 10.5])

        np.testing.assert_array_equal(plotter.generate_pGly_grid(), [100.0, 100.0])
