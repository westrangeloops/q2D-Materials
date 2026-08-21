"""Slabs wrapper providing convenient stack-specific analysis for multi-slab structures."""

from typing import Any, Dict, Iterator, List, Optional, TYPE_CHECKING

from .layer_analysis import get_intralayer_bxb, get_interlayer_bxb, get_all_interlayer_bxb

if TYPE_CHECKING:
    from .layers_wrapper import Layers


class Slabs:
    """Wrapper for slab (stack) data with layer-scoped analysis helpers.

    Attributes
    ----------
    analyzer : q2D_analyzer
        Reference to parent analyzer instance
    """

    def __init__(self, analyzer):
        self._analyzer = analyzer
        self._slabs_dict: Optional[Dict[str, Dict[str, Any]]] = None

    def _ensure_slabs_loaded(self) -> Dict[str, Dict[str, Any]]:
        if self._slabs_dict is None:
            self._slabs_dict = self._analyzer.get_slabs()
        return self._slabs_dict

    def layers_of(self, slab_id: str) -> "LayersView":
        """Return a layer view restricted to layers belonging to this slab."""
        slabs = self._ensure_slabs_loaded()
        key = str(slab_id)
        if key not in slabs:
            raise KeyError(f"Slab '{key}' not found. Available: {list(slabs.keys())}")
        return LayersView(self._analyzer, slabs[key].get('layer_ids', []))

    def __getitem__(self, key: str) -> Dict[str, Any]:
        slabs = self._ensure_slabs_loaded()
        if key not in slabs:
            raise KeyError(f"Slab '{key}' not found. Available: {list(slabs.keys())}")
        return slabs[key]

    def __iter__(self) -> Iterator[str]:
        slabs = self._ensure_slabs_loaded()
        return iter(sorted(slabs.keys(), key=lambda x: int(x) if x.isdigit() else float('inf')))

    def __len__(self) -> int:
        return len(self._ensure_slabs_loaded())

    def __contains__(self, key: str) -> bool:
        return key in self._ensure_slabs_loaded()

    def __repr__(self) -> str:
        slabs = self._ensure_slabs_loaded()
        return f"Slabs({len(slabs)} stacks: {list(slabs.keys())})"

    def keys(self) -> List[str]:
        return list(iter(self))

    def values(self) -> List[Dict[str, Any]]:
        slabs = self._ensure_slabs_loaded()
        return [slabs[k] for k in self.keys()]

    def items(self):
        slabs = self._ensure_slabs_loaded()
        for k in self.keys():
            yield k, slabs[k]


class LayersView:
    """Filtered Layers-like view for a subset of layer IDs within one slab."""

    def __init__(self, analyzer, layer_ids: List[str]):
        self._analyzer = analyzer
        self._layer_ids = [str(lid) for lid in layer_ids]

    def get_bxb(self, index: Optional[int] = None, layer_id: Optional[str] = None) -> Dict[str, Any]:
        if index is not None:
            if index < 0 or index >= len(self._layer_ids):
                raise ValueError(f"Layer index {index} out of range for this slab")
            layer_id = self._layer_ids[index]
        elif layer_id is None:
            raise ValueError("Either 'index' or 'layer_id' must be provided")
        return get_intralayer_bxb(self._analyzer, str(layer_id))

    def get_interlayer_bxb(self, layer_id1: str, layer_id2: str) -> Dict[str, Any]:
        return get_interlayer_bxb(self._analyzer, str(layer_id1), str(layer_id2))

    def get_all_interlayer_bxb(self) -> Dict[str, Any]:
        """Inter-layer B-X-B only between consecutive layers in this slab."""
        if len(self._layer_ids) < 2:
            return {
                'bxb_angles': [],
                'bxb_mean': float('nan'),
                'bxb_std': float('nan'),
                'count': 0,
                'pairs': [],
                'pair_data': {},
            }
        sorted_ids = sorted(self._layer_ids, key=lambda x: int(x) if x.isdigit() else float('inf'))
        all_angles = []
        pair_data = {}
        pairs = []
        for i in range(len(sorted_ids) - 1):
            id1, id2 = sorted_ids[i], sorted_ids[i + 1]
            try:
                result = get_interlayer_bxb(self._analyzer, id1, id2)
                pair_data[(id1, id2)] = result
                pairs.append((id1, id2))
                if result.get('count', 0) > 0:
                    all_angles.extend(result['bxb_angles'].tolist())
            except ValueError:
                continue
        import numpy as np
        arr = np.array(all_angles) if all_angles else np.array([])
        return {
            'bxb_angles': arr,
            'bxb_mean': float(np.mean(arr)) if len(arr) else float('nan'),
            'bxb_std': float(np.std(arr)) if len(arr) else float('nan'),
            'count': len(arr),
            'pairs': pairs,
            'pair_data': pair_data,
        }

    def __getitem__(self, key: str) -> Dict[str, Any]:
        if key not in self._layer_ids:
            raise KeyError(f"Layer '{key}' not in this slab: {self._layer_ids}")
        return self._analyzer.get_layers()[key]

    def __iter__(self) -> Iterator[str]:
        return iter(sorted(self._layer_ids, key=lambda x: int(x) if x.isdigit() else float('inf')))

    def __len__(self) -> int:
        return len(self._layer_ids)

    def __repr__(self) -> str:
        return f"LayersView({self._layer_ids})"
