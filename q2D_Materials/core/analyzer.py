"""
Minimal q2D Analyzer - Basic structure analysis.

This is a minimal analyzer class for basic perovskite structure analysis.
"""

from ase.io import read


class q2D_analyzer:
    """
    Minimal analyzer for 2D quantum materials.

    Basic functionality for loading and analyzing perovskite structures.
    """

    def __init__(self, file_path=None):
        """
        Initialize the q2D_analyzer.

        Parameters:
        file_path (str, optional): Path to the VASP file to load initially
        """
        self.file_path = file_path
        if file_path:
            self.experiment_name = file_path.split('/')[-1].split('.')[0]
            self.cell = read(file_path)
            # Ensure PBC is enabled for crystal structures
            self.cell.pbc = True
