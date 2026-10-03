# Global Scan Results
Corresponds to Section 3.1 and Section 4.2 of the paper.

## Files
- `all_inclusions_summary_*.csv`: Full database of all detected inclusions (coordinates, size, morphology)
- `all_inclusions_summary_*.json`: JSON format full data
- `inclusions_with_morphology.csv`: Inclusions with morphological parameter calculations
- `morphology_summary.csv`: Statistical summary of morphology parameters
- `aspect_ratio_vs_gas_liquid.png`: Morphology distribution plot
- `Fig6.tif`: Paper Figure 6 - scan result visualization

## Analysis Scripts
Scan analysis code is located at `src/measurement/`:
- `analysis_scan.py`
- `integrated_scan_analysis.py`
