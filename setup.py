from setuptools import find_packages, setup
setup(
    name='svc_maestra',
    packages=find_packages(include=['svc_materials']),
    install_requires=[
    'numpy',
    'matplotlib',
    'pandas',
    'ase',
    'rdkit'
    'octadist'
    ],
    url='https://github.com/westrangeloops/SVC-Maestra.git',
    author='Jesus Camilo Díaz Olivella - MTSG-UDEA',
    author_email='jesusc.diaz@udea.edu.co',
    version='1.0',
    description='SVC-Maestra is a package to create DionJacobson perovskite',
    license='MIT',
)