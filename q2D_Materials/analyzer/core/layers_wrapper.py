"""Layers wrapper providing convenient layer-specific analysis methods.

This module provides the Layers class which wraps analyzer layer data and
provides an intuitive API for layer-specific analysis like B-X-B angles.
"""

from typing import Dict, Optional, Any, Iterator, List
from .layer_analysis import get_intralayer_bxb, get_interlayer_bxb, get_all_interlayer_bxb


class Layers:
    """Wrapper for layer data with convenient analysis methods.
    
    Provides access to layer information and layer-specific analysis methods.
    Supports both dict-like access (layers['0']) and method calls (layers.get_bxb(index=0)).
    
    Attributes
    ----------
    analyzer : q2D_analyzer
        Reference to parent analyzer instance
    """
    
    def __init__(self, analyzer):
        """Initialize Layers wrapper.
        
        Parameters
        ----------
        analyzer : q2D_analyzer
            Parent analyzer instance
        """
        self._analyzer = analyzer
        self._layers_dict: Optional[Dict] = None
    
    def _ensure_layers_loaded(self) -> Dict:
        """Ensure layers dict is loaded from analyzer."""
        if self._layers_dict is None:
            self._layers_dict = self._analyzer.get_layers()
        return self._layers_dict
    
    def get_bxb(self, index: Optional[int] = None, layer_id: Optional[str] = None) -> Dict[str, Any]:
        """Get intra-layer B-Xequatorial-B angles for a specific layer.
        
        Calculates B-X-B angles within a single layer using only EQUATORIAL X atoms.
        This measures in-plane octahedral connectivity, excluding axial X atoms.
        
        Parameters
        ----------
        index : int, optional
            Layer index (0-based). If provided, layer_id is ignored.
            Example: index=0 gets the first layer.
        layer_id : str, optional
            Layer ID (e.g., '0', '1'). Used if index is None.
            
        Returns
        -------
        dict
            Dictionary with B-Xeq-B angle data:
            - 'bxb_angles': numpy array of B-Xeq-B angles in degrees (equatorial X only)
            - 'bxb_mean': mean angle
            - 'bxb_std': standard deviation
            - 'count': number of angles
            - 'layer_id': the layer ID
            
        Raises
        ------
        ValueError
            If neither index nor layer_id provided, or if layer not found
        """
        # Determine layer_id from index or parameter
        if index is not None:
            layers = self._ensure_layers_loaded()
            layer_ids = sorted(layers.keys(), key=lambda x: int(x) if x.isdigit() else float('inf'))
            if index < 0 or index >= len(layer_ids):
                raise ValueError(f"Layer index {index} out of range [0, {len(layer_ids)-1}]")
            layer_id = layer_ids[index]
        elif layer_id is not None:
            layer_id = str(layer_id)
        else:
            raise ValueError("Either 'index' or 'layer_id' must be provided")
        
        # Calculate intra-layer B-X-B angles
        return get_intralayer_bxb(self._analyzer, layer_id)
    
    def get_interlayer_bxb(self, layer_id1: str, layer_id2: str) -> Dict[str, Any]:
        """Get inter-layer B-Xaxial-B angles between two layers.
        
        Calculates B-X-B angles between two layers using only AXIAL/INTERLAYER X atoms
        that bridge between the layers. This measures octahedral tilting between layers.
        
        Parameters
        ----------
        layer_id1 : str
            First layer ID
        layer_id2 : str
            Second layer ID
            
        Returns
        -------
        dict
            Dictionary with inter-layer B-Xaxial-B angle data:
            - 'bxb_angles': numpy array of B-Xaxial-B angles in degrees (interlayer X only)
            - 'bxb_mean': mean angle
            - 'bxb_std': standard deviation
            - 'count': number of angles
            - 'layer_pair': tuple of (layer_id1, layer_id2)
            
        Raises
        ------
        ValueError
            If either layer not found
        """
        return get_interlayer_bxb(self._analyzer, str(layer_id1), str(layer_id2))
    
    def get_all_interlayer_bxb(self) -> Dict[str, Any]:
        """Get all inter-layer B-Xaxial-B angles in the structure.
        
        Calculates B-Xaxial-B angles for all adjacent layer pairs using only
        axial/interlayer X atoms that bridge between layers.
        
        Returns
        -------
        dict
            Dictionary with all inter-layer B-Xaxial-B angles:
            - 'bxb_angles': combined array of all inter-layer B-Xaxial-B angles
            - 'bxb_mean': mean angle across all pairs
            - 'bxb_std': standard deviation across all pairs
            - 'count': total number of angles
            - 'pairs': list of layer pairs analyzed
            - 'pair_data': dict with per-pair angle data
        """
        return get_all_interlayer_bxb(self._analyzer)
    
    def __getitem__(self, key: str) -> Dict[str, Any]:
        """Get layer data by ID (dict-like access).
        
        Parameters
        ----------
        key : str
            Layer ID (e.g., '0', '1')
            
        Returns
        -------
        dict
            Layer data dictionary with keys like 'octahedra', 'octahedra_count', 'z_coord'
            
        Example
        -------
        >>> layer_data = analyzer.layers['0']
        >>> print(layer_data['z_coord'])
        """
        layers = self._ensure_layers_loaded()
        if key not in layers:
            raise KeyError(f"Layer '{key}' not found. Available layers: {list(layers.keys())}")
        return layers[key]
    
    def __iter__(self) -> Iterator[str]:
        """Iterate over layer IDs.
        
        Example
        -------
        >>> for layer_id in analyzer.layers:
        ...     print(f"Layer {layer_id} has {analyzer.layers[layer_id]['octahedra_count']} octahedra")
        """
        layers = self._ensure_layers_loaded()
        return iter(sorted(layers.keys(), key=lambda x: int(x) if x.isdigit() else float('inf')))
    
    def __len__(self) -> int:
        """Get number of layers."""
        layers = self._ensure_layers_loaded()
        return len(layers)
    
    def __contains__(self, key: str) -> bool:
        """Check if layer ID exists."""
        layers = self._ensure_layers_loaded()
        return key in layers
    
    def __repr__(self) -> str:
        """String representation."""
        layers = self._ensure_layers_loaded()
        return f"Layers({len(layers)} layers: {list(layers.keys())})"
    
    def keys(self) -> List[str]:
        """Get all layer IDs."""
        layers = self._ensure_layers_loaded()
        return sorted(layers.keys(), key=lambda x: int(x) if x.isdigit() else float('inf'))
    
    def values(self) -> List[Dict]:
        """Get all layer data."""
        layers = self._ensure_layers_loaded()
        return [layers[k] for k in self.keys()]
    
    def items(self):
        """Iterate over (layer_id, layer_data) pairs."""
        layers = self._ensure_layers_loaded()
        for k in self.keys():
            yield k, layers[k]
