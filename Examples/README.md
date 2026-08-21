# Examples

These guides are grouped by intent. Each Markdown file explains how to configure the input, run the snippet inside `devenv shell`, and visualize the output (often by re-running `Examples/plot.py`).

## Creating structures

1. [1_Creator.md](1_Creator.md) — Start with the `q2D_creator`, pick layers, and export a bulk structure.
2. [2_Templates.md](2_Templates.md) — Learn how templates are built from JSON and how to add new layer types.
3. [3_GlazerBuilder.md](3_GlazerBuilder.md) — Apply Glazer tilts with the builder helpers.
4. [4_Jagodzinski.md](4_Jagodzinski.md) — Walk through the Jagodzinski-style stacking patterns.
5. [5_Monolayer.md](5_Monolayer.md) — Build monolayer slabs with vacuum and spacer options.
6. [6_Twist.md](6_Twist.md) — Create twisted interfaces from two monolayers.
7. [7_DionJacobson.md](7_DionJacobson.md) — Compose Dion–Jacobson blocks with paired spacers.
8. [8_Ruddlessden_Popper.md](8_Ruddlessden_Popper.md) — Step through RP slabs with plan/side/isometric views.
9. [9_Salts.md](9_Salts.md) — Mix salt layers and spot-check ionic ordering.
10. [10_SlabArchitect.md](10_SlabArchitect.md) — Assemble slab stacks with custom layer heights.
11. [11_Collision_Avoidance.md](11_Collision_Avoidance.md) — Detect and resolve overlapping layers before export.

## Analyzing existing structures

12. [12_Analysis.md](12_Analysis.md) — Inspect lattice parameters, bond lengths, and tolerances.
13. [13_GlazerAnalysis.md](13_GlazerAnalysis.md) — Measure octahedral tilts and Glazer patterns.
14. [14_RDF.md](14_RDF.md) — Generate radial distribution functions for ions.
15. [15_BX.md](15_BX.md) — Track B–X pair frequencies across a batch of samples.
16. [16_GraphQuery.md](16_GraphQuery.md) — Query the graph representation of a structure.
17. [17_MoleculeAnalyzer.md](17_MoleculeAnalyzer.md) — Analyze organic spacers and their penetration depths.
18. [18_TwisterAnalyzer.md](18_TwisterAnalyzer.md) — Analyze twister slabs, stacking registry (\(r\) vs \(R\)), and dual-ratio maps for five creator examples.
19. [19_MoleculeModifier.md](19_MoleculeModifier.md) — Replace spacers or add new functional groups.
20. [20_MoleculeValidator.md](20_MoleculeValidator.md) — Validate linker chemistry against SMARTS patterns.
21. [21_BackboneQuery.md](21_BackboneQuery.md) — Walk the inorganic backbone graph and annotate connectors.
22. [22_DistortionAnalysis.md](22_DistortionAnalysis.md) — Quantify distortions and compare them to reference phases.

For a live preview of any example, open the Markdown file and rerun `python3 Examples/plot.py` inside the devenv shell.
