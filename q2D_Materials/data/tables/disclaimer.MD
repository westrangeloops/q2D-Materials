# Disclaimer

## Data Source

The tables and data files in this directory (`A-ion_data.xlsx`, `B-ion_data.xlsx`, `X-ion_data.xlsx`, and `Perovskite_ions_data.xlsx`) are sourced from the repository and paper:

**Towards an Interoperable Perovskite Description, Or How to Keep Track of Close to 300 Organic Ions Used in Hybrid Perovskites**

### About The Project

This repository is associated with a paper that provides a comprehensive database of perovskite ions and compositions. The project includes:

- **perovskite_composition_schema.json**: Defines the data schema for perovskite compositions
- **ion_schema.json**: Defines the data schema for perovskite ions
- **Data_ions folder**: Contains Excel files with data for all perovskite ions identified in the project
- **Perovskite composition files folder**: Contains example files of perovskite composition files
- **perovskite_to_json.py**: Contains the class `PerovskiteToJson` with functionality for converting perovskite data to a perovskite composition JSON file
- **GUI_perovskite_to_json.py**: A graphical user interface for simplified data entry and for converting data to a perovskite composition JSON file
- **demo_notebook.ipynb**: Demonstrates how to format the required data in order to be able to convert it to a perovskite composition JSON file
- **demo_notebook_NOMAD.ipynb**: Demonstrates how to access and manipulate data for perovskite compositions and perovskite ions stored in the NOMAD database

### Usage in q2D-Materials

The data from these tables is used as **default parameters** in the q2D-Materials package. Specifically, ionic radii and other properties from these tables are used to:

- Automatically calculate B-X bond distances when not explicitly provided
- Set default values for structure generation parameters
- Provide reference data for perovskite ion properties

### Citation

If you use this code or data in your research, we kindly ask that you cite the published paper:

**Ayman Maqsood, Hampus Näsström, Chen Chen, Li Qiutong, Jingshan Luo, Rayan Chakraborty, Volker Blum, Eva Unger, Claudia Draxl, José A Márquez*, T. Jesper Jacobsson*, Towards an Interoperable Perovskite Description, Or How to Keep Track of Close to 300 Organic Ions Used in Hybrid Perovskites, XXX, 2024**

### License

This code is licensed under the terms of the **Creative Commons Attribution 4.0 International License**.


---

**Note**: The data in these tables represents a comprehensive compilation of perovskite ion properties and should be used as reference information for default parameter calculations in the q2D-Materials package.

