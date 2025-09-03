from setuptools import setup, find_packages

setup(
    name="SVC_materials",
    version="0.2.0",
    description="Comprehensive 2D perovskite and layered material structure creation and analysis",
    long_description="""
    SVC-Materials provides advanced functionality for creating and analyzing 
    2D perovskite and layered material structures. Features include:
    
    - Multiple perovskite phases: bulk, Ruddlesden-Popper, Dion-Jacobson, monolayer
    - Double perovskite structures
    - Advanced molecular orientation and placement control
    - Integration with ASE (Atomic Simulation Environment)
    - Comprehensive analysis tools for structural properties
    - SMILES-based molecule input support
    """,
    packages=find_packages(),
    install_requires=[
        'numpy>=1.19.0',
        'pandas>=1.3.0', 
        'ase>=3.22.0',
    ],
    extras_require={
        'smiles': ['rdkit-pypi'],
        'visualization': ['matplotlib', 'plotly'],
        'analysis': ['scipy', 'scikit-learn'],
    },
    python_requires='>=3.7',
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Chemistry',
        'Topic :: Scientific/Engineering :: Physics',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
    ],
    keywords='perovskite materials science crystallography ase',
    author="SVC Materials Team",
    maintainer_email="contact@svc-materials.org",
)