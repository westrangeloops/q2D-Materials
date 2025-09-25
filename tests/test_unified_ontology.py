from q2D_Materials.core.analyzer import q2D_analyzer

analyzer = q2D_analyzer(file_path="tests/test_structures/MAPbCl3_n1_l1.vasp")

print(analyzer.unified_ontology)