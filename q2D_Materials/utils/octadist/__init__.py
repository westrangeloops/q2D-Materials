# OctaDist  Copyright (C) 2019-2024  Rangsiman Ketkaew et al.
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
# ++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++++

__src__ = "source code"

# Import the main modules we need
from . import io
from . import calc
from . import elements
from . import draw
from . import structure
from . import tools
from . import util
from . import linear
from . import plane
from . import plot
from . import popup
from . import projection
# from . import scripting  # Commented out - has dependency issues

# Expose key functions for easy import
from .io import extract_octa
from .calc import CalcDistortion

__all__ = [
    'io', 'calc', 'elements', 'draw', 'structure', 'tools', 'util',
    'linear', 'plane', 'plot', 'popup', 'projection',  # 'scripting',
    'extract_octa', 'CalcDistortion'
]
