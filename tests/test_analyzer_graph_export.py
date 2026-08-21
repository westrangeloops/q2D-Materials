"""
Test to export graph data for bulk, DJ, and RP perovskite structures.

Compares layer detection across different structure types:
- Bulk: No spacers, all octahedra connected via PBC
- DJ: Bifunctional spacer [NH3+]CCCC[NH3+] (1,4-butanediammonium)
- RP: Monofunctional spacer [NH3+]CCC (propylammonium)
"""

import sys
from pathlib import Path
from ase.io import write

# Add parent directory to path so we can import q2D_Materials
sys.path.insert(0, str(Path(__file__).parent.parent))

from q2D_Materials.core.creator import q2D_creator
from q2D_Materials.analyzer import q2D_analyzer


def print_parameters_used(params_dict, title="Parameters Used"):
    """Print parameters used in structure creation."""
    print(f"\n  {title}:")
    for key, value in sorted(params_dict.items()):
        if value is not None:
            if isinstance(value, (list, tuple)):
                value_str = ', '.join(str(v) for v in value) if len(str(value)) < 100 else f"{type(value).__name__} with {len(value)} items"
            else:
                value_str = str(value)
            print(f"    {key}: {value_str}")


def print_parameters_found(analyzer, title="Parameters Found"):
    """Print parameters found by analyzer."""
    print(f"\n  {title}:")
    
    try:
        # Get structure with analyzed parameters
        analyzed_structure = analyzer.to_q2DStructure()
        
        # Structure type
        print(f"    structure_type: {analyzer.structure_type}")
        
        # Ions
        if analyzed_structure.A_ions:
            a_ions_str = ', '.join(analyzed_structure.A_ions) if isinstance(analyzed_structure.A_ions, list) else str(analyzed_structure.A_ions)
            print(f"    A_ions: {a_ions_str}")
        else:
            print(f"    A_ions: None")
            
        if analyzed_structure.B_ions:
            b_ions_str = ', '.join(analyzed_structure.B_ions) if isinstance(analyzed_structure.B_ions, list) else str(analyzed_structure.B_ions)
            print(f"    B_ions: {b_ions_str}")
        else:
            print(f"    B_ions: None")
            
        if analyzed_structure.X_ions:
            x_ions_str = ', '.join(analyzed_structure.X_ions) if isinstance(analyzed_structure.X_ions, list) else str(analyzed_structure.X_ions)
            print(f"    X_ions: {x_ions_str}")
        else:
            print(f"    X_ions: None")
        
        # Spacer
        spacers = analyzer.get_spacers()
        if spacers:
            spacer_info = []
            for i, spacer in enumerate(spacers[:3]):  # Show first 3
                if hasattr(spacer, 'get_chemical_formula'):
                    formula = spacer.get_chemical_formula(mode='hill')
                else:
                    formula = str(spacer)
                spacer_info.append(formula)
            spacer_str = ', '.join(spacer_info)
            if len(spacers) > 3:
                spacer_str += f" ... and {len(spacers) - 3} more"
            print(f"    spacer: {spacer_str} ({len(spacers)} total)")
        elif analyzed_structure.spacer:
            spacer_str = str(analyzed_structure.spacer)
            if len(spacer_str) > 100:
                spacer_str = spacer_str[:100] + "..."
            print(f"    spacer: {spacer_str}")
        else:
            print(f"    spacer: None")
        
        # BX distance if available
        if hasattr(analyzed_structure, 'BX_dist') and analyzed_structure.BX_dist:
            print(f"    BX_dist: {analyzed_structure.BX_dist:.4f} Å")
        else:
            print(f"    BX_dist: None")
        
    except Exception as e:
        print(f"    Error extracting parameters: {e}")
        import traceback
        traceback.print_exc()
    
    # Additional analysis metrics (always available)
    try:
        octahedra = analyzer.get_octahedra()
        layers = analyzer.get_layers()
        a_sites = analyzer.get_a_sites()
        print(f"    octahedra_count: {len(octahedra)}")
        print(f"    layers_count: {len(layers)}")
        print(f"    a_sites_count: {len(a_sites)}")
    except Exception as e:
        print(f"    Error getting analysis metrics: {e}")


def create_analyze_and_compare(
    A_ions,
    B_ions,
    X_ions,
    structure_type,
    template,
    thickness,
    xy_expansion,
    name=None,
    vasp_file=None,
    json_file=None,
    **extra_params
):
    """
    Create a structure, print parameters used, analyze it, and show what analyzer found.
    
    Parameters
    ----------
    A_ions : str or list
        A-site cation(s)
    B_ions : str or list
        B-site cation(s)
    X_ions : str or list
        X-site anion(s)
    structure_type : str
        Structure type ('bulk' or 'monolayer')
    template : str or dict
        Template name or template dictionary
    thickness : int
        Number of layers/thickness
    xy_expansion : tuple
        XY expansion factors (nx, ny)
    name : str, optional
        Name for the structure (for display)
    vasp_file : str, optional
        Output VASP file path
    json_file : str, optional
        Output JSON graph file path
    **extra_params
        Additional parameters to pass to create_structure (e.g., spacer, layer_sequence)
    
    Returns
    -------
    analyzer : q2D_analyzer
        The analyzer object after analysis
    """
    # Prepare all parameters
    params_used = {
        'A_ions': A_ions,
        'B_ions': B_ions,
        'X_ions': X_ions,
        'structure_type': structure_type,
        'template': template,
        'thickness': thickness,
        'xy_expansion': xy_expansion,
    }
    params_used.update(extra_params)
    
    # Generate default names if not provided
    if name is None:
        a_str = str(A_ions) if not isinstance(A_ions, list) else '-'.join(A_ions)
        b_str = str(B_ions) if not isinstance(B_ions, list) else '-'.join(B_ions)
        x_str = str(X_ions) if not isinstance(X_ions, list) else '-'.join(X_ions)
        name = f"{a_str}{b_str}{x_str}_{structure_type}"
    
    if vasp_file is None:
        safe_name = "".join([c for c in str(name) if c.isalnum() or c in ('-', '_')]).strip()
        vasp_file = f"{safe_name}.vasp"
    
    if json_file is None:
        safe_name = "".join([c for c in str(name) if c.isalnum() or c in ('-', '_')]).strip()
        json_file = f"{safe_name}_graph.json"
    
    # Print header
    print("\n" + "="*60)
    print(f"STRUCTURE: {name}")
    print("="*60)
    
    # Print parameters used
    print_parameters_used(params_used, "Parameters Used")
    
    # Create structure
    print("\n  Creating structure...")
    q2d = q2D_creator()
    structure = q2d.create_structure(**params_used)
    
    print(f"  Structure created: {len(structure)} atoms")
    print(f"  Formula: {structure.get_chemical_formula()}")
    
    # Save as VASP file
    if vasp_file:
        write(vasp_file, structure)
        print(f"  Saved to: {vasp_file}")
    
    # Analyze the structure
    print("\n  Analyzing structure...")
    analyzer = q2D_analyzer(structure)
    analyzer.analyze()
    
    # Print parameters found
    print_parameters_found(analyzer, "Parameters Found by Analyzer")
    
    # Export graph data if requested
    if json_file:
        exported_path = analyzer.export_graph_data(json_file)
        print(f"\n  Exported graph: {exported_path}")
    
    return analyzer


def analyze_and_export(structure, name, vasp_file, json_file, params_used=None):
    """Analyze a structure and export graph data."""
    # Save as VASP file
    write(vasp_file, structure)
    print(f"  Created: {vasp_file}")
    print(f"  Total atoms: {len(structure)}")
    print(f"  Formula: {structure.get_chemical_formula()}")

    # Print parameters used if provided
    if params_used:
        print_parameters_used(params_used)
    else:
        print("\n  Parameters Used: Not provided")

    # Analyze the structure
    analyzer = q2D_analyzer(vasp_file)
    analyzer.analyze()

    # Print parameters found
    print_parameters_found(analyzer)

    # Print detailed analysis results
    octahedra = analyzer.get_octahedra()
    layers = analyzer.get_layers()
    spacers = analyzer.get_spacers()
    a_sites = analyzer.get_a_sites()
    structure_type = analyzer.structure_type

    print(f"\n  Detailed Analysis:")
    print(f"    Structure type: {structure_type}")
    print(f"    Octahedra count: {len(octahedra)}")
    print(f"    Layers count: {len(layers)}")
    for layer_id, layer_info in layers.items():
        z_coord = layer_info.get('z_coord', 'N/A')
        z_str = f"{z_coord:.2f}" if isinstance(z_coord, (int, float)) else str(z_coord)
        print(f"      Layer {layer_id}: {len(layer_info['octahedra'])} octahedra, z={z_str}")
    print(f"    Spacers count: {len(spacers)}")
    print(f"    A-sites count: {len(a_sites)}")

    # Print X-atom classifications if available
    if hasattr(analyzer, '_x_atom_classifications') and analyzer._x_atom_classifications:
        x_class = analyzer._x_atom_classifications
        axial = sum(1 for v in x_class.values() if v['type'] == 'axial')
        interlayer = sum(1 for v in x_class.values() if v['type'] == 'interlayer')
        intralayer = sum(1 for v in x_class.values() if v['type'] == 'intralayer')
        print(f"    X-atom classifications:")
        print(f"      Axial (terminal): {axial}")
        print(f"      Interlayer: {interlayer}")
        print(f"      Intralayer: {intralayer}")

    # Export graph data as JSON
    exported_path = analyzer.export_graph_data(json_file)
    print(f"\n  Exported graph: {exported_path}")

    return analyzer


def create_bulk_structure():
    """Create bulk CsPbI3 structure."""
    print("\n" + "="*60)
    print("BULK STRUCTURE: CsPbI3")
    print("="*60)

    # Store parameters used
    params_used = {
        'A_ions': 'Cs',
        'B_ions': 'Pb',
        'X_ions': 'I',
        'structure_type': 'bulk',
        'template': 'cubic',
        'thickness': 2,
        'xy_expansion': (2, 2),
    }

    q2d = q2D_creator()
    structure = q2d.create_structure(**params_used)

    return analyze_and_export(
        structure,
        "Bulk CsPbI3",
        "CsPbI3_bulk.vasp",
        "CsPbI3_bulk_graph.json",
        params_used=params_used
    )


def create_dj_structure():
    """Create DJ structure with bifunctional spacer [NH3+]CCCC[NH3+]."""
    print("\n" + "="*60)
    print("DJ STRUCTURE: (BDA)PbI4")
    print("Spacer: [NH3+]CCCC[NH3+] (1,4-butanediammonium)")
    print("="*60)

    # Store parameters used
    params_used = {
        'A_ions': 'MA',
        'B_ions': 'Pb',
        'X_ions': 'I',
        'structure_type': 'bulk',
        'template': 'cubic',
        'layer_sequence': 'DJ',
        'thickness': 2,
        'xy_expansion': (2, 2),
        'spacer': '[NH3+]CCCC[NH3+]',  # 1,4-butanediammonium (bifunctional)
    }

    q2d = q2D_creator()
    structure = q2d.create_structure(**params_used)

    return analyze_and_export(
        structure,
        "DJ (BDA)PbI4",
        "DJ_BDA_PbI4.vasp",
        "DJ_BDA_PbI4_graph.json",
        params_used=params_used
    )


def create_rp_structure():
    """Create RP structure with monofunctional spacer [NH3+]CCC."""
    print("\n" + "="*60)
    print("RP STRUCTURE: (PA)2PbI4")
    print("Spacer: [NH3+]CCC (propylammonium)")
    print("="*60)

    # Store parameters used
    params_used = {
        'A_ions': 'MA',
        'B_ions': 'Pb',
        'X_ions': 'I',
        'structure_type': 'bulk',
        'template': 'cubic',
        'layer_sequence': 'RP',
        'thickness': 2,
        'xy_expansion': (2, 2),
        'spacer': '[NH3+]CCC',  # Propylammonium (monofunctional)
    }

    q2d = q2D_creator()
    structure = q2d.create_structure(**params_used)

    return analyze_and_export(
        structure,
        "RP (PA)2PbI4",
        "RP_PA2_PbI4.vasp",
        "RP_PA2_PbI4_graph.json",
        params_used=params_used
    )


def main():
    """Create and analyze bulk, DJ, and RP structures."""
    print("="*60)
    print("PEROVSKITE STRUCTURE ANALYZER TEST")
    print("Comparing Bulk, DJ, and RP layer detection")
    print("="*60)

    # Create and analyze each structure type
    bulk_analyzer = create_bulk_structure()
    dj_analyzer = create_dj_structure()
    rp_analyzer = create_rp_structure()

    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    print(f"{'Structure':<20} {'Type':<10} {'Layers':<8} {'Octahedra':<10} {'Spacers':<8} {'A-sites':<8}")
    print("-"*70)

    for name, analyzer in [("Bulk CsPbI3", bulk_analyzer),
                           ("DJ (BDA)PbI4", dj_analyzer),
                           ("RP (PA)2PbI4", rp_analyzer)]:
        spacers = analyzer.get_spacers()
        a_sites = analyzer.get_a_sites()
        print(f"{name:<20} {analyzer.structure_type:<10} {len(analyzer.get_layers()):<8} "
              f"{len(analyzer.get_octahedra()):<10} {len(spacers):<8} {len(a_sites):<8}")
        
        # Show spacer details for DJ and RP
        if spacers:
            print(f"  {' '*20} Spacers:")
            for i, spacer in enumerate(spacers[:3]):  # Show first 3
                formula = spacer.get_chemical_formula(mode='hill') if hasattr(spacer, 'get_chemical_formula') else str(spacer)
                print(f"  {' '*20}   {i+1}. {formula}")
            if len(spacers) > 3:
                print(f"  {' '*20}   ... and {len(spacers) - 3} more")

    print("\n" + "="*60)
    print("EXPORTED FILES")
    print("="*60)
    print("  - CsPbI3_bulk_graph.json")
    print("  - DJ_BDA_PbI4_graph.json")
    print("  - RP_PA2_PbI4_graph.json")
    print("\nUpload these files to the Structure Analyzer web interface!")


if __name__ == "__main__":
    main()

