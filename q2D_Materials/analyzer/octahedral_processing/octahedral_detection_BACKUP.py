# BACKUP of classify_atoms_by_topology before topology propagation algorithm implementation
# Created: 2026-02-01

# This file contains the old implementation for reference
# The old algorithm used:
# 1. Sharing count threshold (max_sharing) for small cells
# 2. Z-clustering fallback for large cells (max_sharing == 1)
# 3. Special handling for 1x1 cells

# See octahedral_detection.py for the new topology propagation algorithm
