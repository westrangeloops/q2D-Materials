# Load a molecule from a file
from ase.io import read
import numpy as np
from dscribe.descriptors import MBTR
from scipy.spatial.distance import cosine, euclidean
import matplotlib.pyplot as plt

# Load molecules from files
try:
    molecule_target = read('Graphs/BULKS_RESULTS/MAPbBr3_n1_[NH3+]C1=CC=C([NH3+])C=C1_analysis/MAPbBr3_n1_[NH3+]C1=CC=C([NH3+])C=C1_large_molecules.xyz')
    molecule_reference = read('Graphs/BULKS_RESULTS/MAPbBr3_n1_[NH3+]C1=CC=C([NH3+])C=C1_analysis/MAPbBr3_n1_[NH3+]C1=CC=C([NH3+])C=C1_large_molecules.xyz')
except FileNotFoundError:
    print("Warning: Molecule files not found. Using dummy molecules for demonstration.")
    # Create dummy molecules for testing
    from ase import Atoms
    molecule_target = Atoms('H2O', positions=[[0, 0, 0], [0.757, 0.587, 0], [-0.757, 0.587, 0]])
    molecule_reference = Atoms('H2O', positions=[[0, 0, 0.1], [0.757, 0.587, 0.05], [-0.757, 0.587, 0.05]])

print(f"Target molecule: {len(molecule_target)} atoms")
print(f"Reference molecule: {len(molecule_reference)} atoms")
print(f"Target species: {molecule_target.get_chemical_symbols()}")
print(f"Reference species: {molecule_reference.get_chemical_symbols()}")

# Get unique species across both molecules
all_species = list(set(molecule_target.get_chemical_symbols() + molecule_reference.get_chemical_symbols()))
print(f"All species: {all_species}")

def create_mbtr_k1():
    """Create MBTR descriptor for k=1 (atomic numbers)"""
    return MBTR(
        species=all_species,
        k1={
            "geometry": {"function": "atomic_number"},
            "grid": {"min": 0, "max": 100, "n": 100, "sigma": 0.1}
        },
        periodic=False,
        normalization="l2_each"
    )

def create_mbtr_k2():
    """Create MBTR descriptor for k=2 (distances)"""
    return MBTR(
        species=all_species,
        k2={
            "geometry": {"function": "distance"},
            "grid": {"min": 0, "max": 10, "n": 100, "sigma": 0.1},
            "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-3}
        },
        periodic=False,
        normalization="l2_each"
    )

def create_mbtr_k3():
    """Create MBTR descriptor for k=3 (angles)"""
    return MBTR(
        species=all_species,
        k3={
            "geometry": {"function": "angle"},
            "grid": {"min": 0, "max": 180, "n": 100, "sigma": 5},
            "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-3}
        },
        periodic=False,
        normalization="l2_each"
    )

def compare_descriptors(desc1, desc2, k_term):
    """Compare two descriptors and return difference metrics"""
    # Calculate differences
    absolute_diff = np.abs(desc1 - desc2)
    relative_diff = np.abs(desc1 - desc2) / (np.abs(desc1) + np.abs(desc2) + 1e-10)
    
    # Calculate similarity metrics
    cosine_sim = 1 - cosine(desc1, desc2)
    euclidean_dist = euclidean(desc1, desc2)
    l2_diff = np.linalg.norm(desc1 - desc2)
    
    print(f"\n=== K={k_term} Term Comparison ===")
    print(f"Descriptor shape: {desc1.shape}")
    print(f"Cosine Similarity: {cosine_sim:.6f}")
    print(f"Euclidean Distance: {euclidean_dist:.6f}")
    print(f"L2 Difference: {l2_diff:.6f}")
    print(f"Mean Absolute Difference: {np.mean(absolute_diff):.6f}")
    print(f"Max Absolute Difference: {np.max(absolute_diff):.6f}")
    print(f"Mean Relative Difference: {np.mean(relative_diff):.6f}")
    print(f"Std of Differences: {np.std(absolute_diff):.6f}")
    
    return {
        'cosine_similarity': cosine_sim,
        'euclidean_distance': euclidean_dist,
        'l2_difference': l2_diff,
        'absolute_diff': absolute_diff,
        'relative_diff': relative_diff,
        'mean_abs_diff': np.mean(absolute_diff),
        'max_abs_diff': np.max(absolute_diff),
        'std_diff': np.std(absolute_diff)
    }

def plot_descriptor_differences(results_dict, all_species):
    """Plot the descriptor differences for visualization"""
    fig, axes = plt.subplots(3, 2, figsize=(15, 12))
    fig.suptitle('MBTR Descriptor Differences Between Target and Reference Molecules', fontsize=16)
    
    for i, (k_term, results) in enumerate(results_dict.items()):
        # Plot absolute differences
        ax1 = axes[i, 0]
        ax1.plot(results['absolute_diff'], 'b-', alpha=0.7)
        ax1.set_title(f'K={k_term} Absolute Differences')
        ax1.set_xlabel('Feature Index')
        ax1.set_ylabel('Absolute Difference')
        ax1.grid(True, alpha=0.3)
        
        # Plot relative differences
        ax2 = axes[i, 1]
        ax2.plot(results['relative_diff'], 'r-', alpha=0.7)
        ax2.set_title(f'K={k_term} Relative Differences')
        ax2.set_xlabel('Feature Index')
        ax2.set_ylabel('Relative Difference')
        ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('mbtr_descriptor_differences.png', dpi=300, bbox_inches='tight')
    print("\nPlot saved as 'mbtr_descriptor_differences.png'")
    return fig

# Main comparison
print("\n" + "="*60)
print("COMPARING MOLECULES USING MBTR DESCRIPTORS")
print("="*60)

try:
    results = {}
    
    # K=1 comparison (atomic numbers)
    print("\nCalculating K=1 descriptors...")
    mbtr_k1 = create_mbtr_k1()
    desc1_k1 = mbtr_k1.create(molecule_target)
    desc2_k1 = mbtr_k1.create(molecule_reference)
    results['1'] = compare_descriptors(desc1_k1, desc2_k1, 1)
    
    # K=2 comparison (distances)
    print("\nCalculating K=2 descriptors...")
    mbtr_k2 = create_mbtr_k2()
    desc1_k2 = mbtr_k2.create(molecule_target)
    desc2_k2 = mbtr_k2.create(molecule_reference)
    results['2'] = compare_descriptors(desc1_k2, desc2_k2, 2)
    
    # K=3 comparison (angles)
    print("\nCalculating K=3 descriptors...")
    mbtr_k3 = create_mbtr_k3()
    desc1_k3 = mbtr_k3.create(molecule_target)
    desc2_k3 = mbtr_k3.create(molecule_reference)
    results['3'] = compare_descriptors(desc1_k3, desc2_k3, 3)
    
    # Summary comparison
    print("\n" + "="*60)
    print("SUMMARY COMPARISON")
    print("="*60)
    print(f"{'K-term':<8} {'Cosine Sim':<12} {'L2 Diff':<12} {'Mean Abs Diff':<15}")
    print("-" * 50)
    for k_term, result in results.items():
        print(f"K={k_term:<6} {result['cosine_similarity']:<12.6f} {result['l2_difference']:<12.6f} {result['mean_abs_diff']:<15.6f}")
    
    # Create visualization
    print("\nCreating visualization...")
    plot_descriptor_differences(results, all_species)
    
    # Detailed matrix analysis
    print("\n" + "="*60)
    print("DETAILED MATRIX ANALYSIS")
    print("="*60)
    
    for k_term, result in results.items():
        print(f"\nK={k_term} Term Matrix Statistics:")
        print(f"  Shape: {result['absolute_diff'].shape}")
        print(f"  Non-zero differences: {np.count_nonzero(result['absolute_diff'])}")
        print(f"  Percentage non-zero: {100 * np.count_nonzero(result['absolute_diff']) / len(result['absolute_diff']):.2f}%")
        print(f"  95th percentile diff: {np.percentile(result['absolute_diff'], 95):.6f}")
        print(f"  99th percentile diff: {np.percentile(result['absolute_diff'], 99):.6f}")
        
        # Find largest differences
        largest_diffs_idx = np.argsort(result['absolute_diff'])[-5:]
        print(f"  Top 5 largest differences (indices): {largest_diffs_idx}")
        print(f"  Top 5 largest differences (values): {result['absolute_diff'][largest_diffs_idx]}")

except ImportError as e:
    print(f"Error: {e}")
    print("Please install DScribe: pip install dscribe")
except Exception as e:
    print(f"An error occurred: {e}")
    import traceback
    traceback.print_exc()