![Github_portada](https://github.com/westrangeloops/SVC-Maestra/assets/34012719/759685f8-0789-47a2-8302-471e389a86a8)

# SVC-Materials
SVC-Maestra is a Python package for creating DJ perovskite, currently developed by the Theoritical materials science group of the University of Antioquia. SVC-Maestra simplifies the generation of initial input structures for quasi-2D perovskite systems, providing researchers with a reliable starting point for subsequent first principles calculations. By facilitating accurate structure generation, it supports systematic and rational analysis of electronic, optical, and structural properties of perovskite materials through physics-based simulations.
The script needs the position of the six coordinates to correctly work; it is important to know beforehand the desired position of in which the nitrogen of the spacer to be positioned, for example in the perovskite MAPbI3 we would like to have the nitrogen of the organic spacer aligned with the methyl-ammonium (MA) of the perovskite. Here, P1 and P3 are the positions of the nitrogen of the MA, one nitrogen.

## Installation
​Just clone the repo and use the workspace Jupyter notebook

## Usage

#To use SVC-Maestra, you can import the package and use the following functions:

``` 
from svc_maestra_lib.svc_maestra import q2D_creator
form svc_maestra_lib.svc_maestra import q2D_analysis
```
## Examples
There is under the folder examples a jupyter notebook included
​
## Important information:
SVC-Materials is disitributed with a copy of OctaDist, if you want more information about all wonderful octadist capabilities please refer to the original page:
https://octadist.github.io/

## Contributing
We welcome contributions to SVC-Materials. If you would like to contribute, please fork the repository and submit a pull request.

## License

SVC-Maestra is licensed under the MIT license.
