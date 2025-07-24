"""
SVC Materials Core Module - Refactored Modular Architecture

This module provides octahedral analysis functionality through a modular architecture:

Main Classes:
- q2D_analyzer: Main analysis class (refactored)
- GeometryCalculator: Geometric calculations and coordinate transformations
- AngularAnalyzer: Angular analysis between octahedra
- ConnectivityAnalyzer: Connectivity and network analysis

Legacy support:
- The original q2D_analyzer from analyzer.py is still available for backward compatibility
"""

# Import the refactored modular components
from .geometry import GeometryCalculator
from .angular_analysis import AngularAnalyzer
from .connectivity import ConnectivityAnalyzer

# Import the refactored main analyzer
from .analyzer import q2D_analyzer

# Also import the original analyzer for backward compatibility
try:
    from .analyzer import q2D_analyzer as q2D_analyzer_legacy
except ImportError:
    q2D_analyzer_legacy = None

__all__ = [
    'q2D_analyzer',
    'GeometryCalculator', 
    'AngularAnalyzer',
    'ConnectivityAnalyzer',
    'q2D_analyzer_legacy'
] 