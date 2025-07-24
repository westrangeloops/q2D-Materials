"""
Simple example demonstrating the layer analysis functionality.

This shows how the new LayersAnalyzer works with the window-based approach:
- Sorts octahedra by Z-coordinate
- Groups them into layers using a 2 Å window
- Provides simple layer statistics and summaries
"""

def demonstrate_layer_analysis():
    """
    Demonstrate the layer analysis workflow.
    """
    print("=== Simple Layer Analysis Example ===\n")
    
    print("How the Layer Analysis Works:")
    print("1. Takes all octahedra central atoms (the 'b' atoms like Pb)")
    print("2. Sorts them by Z-coordinate (lowest first)")
    print("3. Uses first octahedron as reference for layer 1")
    print("4. Groups all octahedra within 2 Å Z-window as same layer")
    print("5. When octahedron is outside window, starts new layer")
    print("6. Continues until all octahedra are assigned to layers")
    print()
    
    print("Example with mock data:")
    print("Octahedra Z-coordinates: [0.1, 0.3, 0.5, 3.2, 3.4, 6.8, 7.0]")
    print("Z-window: 2.0 Å")
    print()
    
    print("Layer assignment process:")
    print("- Start with 0.1 Å as reference for layer 1")
    print("- 0.3 Å: |0.3-0.1| = 0.2 Å < 2.0 Å → layer 1")
    print("- 0.5 Å: |0.5-0.1| = 0.4 Å < 2.0 Å → layer 1")
    print("- 3.2 Å: |3.2-0.1| = 3.1 Å > 2.0 Å → new layer 2 (reference: 3.2 Å)")
    print("- 3.4 Å: |3.4-3.2| = 0.2 Å < 2.0 Å → layer 2")
    print("- 6.8 Å: |6.8-3.2| = 3.6 Å > 2.0 Å → new layer 3 (reference: 6.8 Å)")
    print("- 7.0 Å: |7.0-6.8| = 0.2 Å < 2.0 Å → layer 3")
    print()
    
    print("Result:")
    print("Layer 1: octahedra at Z = [0.1, 0.3, 0.5] Å")
    print("Layer 2: octahedra at Z = [3.2, 3.4] Å")
    print("Layer 3: octahedra at Z = [6.8, 7.0] Å")
    print()
    
    print("Usage in your code:")
    print("```python")
    print("# Load structure and analyze")
    print("analyzer = q2D_analyzer('your_structure.vasp', b='Pb', x='Cl')")
    print()
    print("# Get layer summary")
    print("print(analyzer.get_layer_summary())")
    print()
    print("# Get octahedra in specific layer")
    print("layer1_octahedra = analyzer.get_octahedra_in_layer('layer_1')")
    print()
    print("# Find which layer contains specific octahedron")
    print("layer = analyzer.get_layer_by_octahedron('octahedron_1')")
    print()
    print("# Adjust Z-window and re-analyze")
    print("analyzer.set_layer_window(z_window=1.5)  # Tighter grouping")
    print("analyzer.set_layer_window(z_window=3.0)  # Looser grouping")
    print()
    print("# Export results")
    print("analyzer.export_layer_analysis('layers.json')")
    print("```")
    print()
    
    print("Key Features:")
    print("✓ Simple window-based algorithm (no complex clustering)")
    print("✓ Uses actual central atom Z-coordinates from octahedra")
    print("✓ Automatically integrated into existing analyzer")
    print("✓ Adjustable Z-window parameter")
    print("✓ Provides layer statistics and separations")
    print("✓ Easy to understand and modify")


if __name__ == "__main__":
    demonstrate_layer_analysis() 