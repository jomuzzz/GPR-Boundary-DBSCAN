## Overview
This repository provides a graphical analysis tool for Ground Penetrating Radar (GPR) signals.  
It includes functions for envelope extraction, valley/peak detection,  
boundary point identification, DBSCAN clustering, and curve fitting for structural analysis.  
The tool is developed as part of an ongoing research project, and the related paper is currently under peer review.

## Using the Application
To use the tool, prepare your GPR data in CSV format.  
Most GPR systems allow exporting raw A-scan or B-scan data as text or binary files.  
You can convert the data into CSV by arranging each trace as a column (B-scan)  
or saving a single A-scan as a single column.

### Steps to use the tool:
1. Convert your radar data to a CSV file  
   - Each column represents one trace (for B-scan)  
   - Each row represents a sample point  
   - No header is required  
2. Open the application (`gpr_boundary_analyzer.py`).  
3. Click **"Open CSV File"** and select your data file.  
4. Navigate through A-scan and B-scan views.  
5. Follow the interface workflow:  
   - Initial boundary detection  
   - Optional PCA filtering  
   - DBSCAN clustering  
   - Curve fitting and valley extraction  
6. Export results or images as needed.

The tool will guide you through each step via the on-screen instructions.
