# Thermography Analysis Toolkit

Python package implementing advanced thermographic non-destructive evaluation (NDE) methods for defect detection in thermal sequences.

## Key Features

- **Multi-format support**: Load thermal sequences from binary (.bin) or CSV files
- **5 advanced algorithms**: PPT, PCT, TSR, HOS, and DMD implementations
- **GPU-ready**: PyTorch compatibility for key functions
- **Preprocessing**: Savitzky-Golay filtering and binarization tools

## Installation

```bash
pip install numpy scipy scikit-learn torch
```

Simply copy the `toolbox_pulse.py` file into your project.

## Implemented Methods

### 1. Pulse Phase Thermography (PPT)
```python
phasegram, magnitude = PPT(data, mode_num=15)
```
**Reference**:  
Maldague, X. (2001). *Theory and Practice of Infrared Thermography for NDT*. Wiley.  
[DOI:10.1002/9781118706239](https://doi.org/10.1002/9781118706239)

### 2. Principal Component Thermography (PCT)
```python
EOFs, singular_values = PCT(data, n_components=5)
```
**Reference**:  
Rajic, N. (2002). *Principal Component Thermography*. Defence Science and Technology Organisation.  
[DSTO-TR-1291](https://apps.dtic.mil/sti/citations/ADA406088)

### 3. Thermographic Signal Reconstruction (TSR)
```python
coefficients = TSR(data, polynomial_order=4)
```
**Reference**:  
Shepard, S. (2003). *Thermographic Signal Reconstruction*. QIRT Journal.  
[DOI:10.3166/qirt.10.85-96](https://doi.org/10.3166/qirt.10.85-96)

### 4. Higher-Order Statistics (HOS)
```python
HOS_matrix = HOS(data)  # [skewness, kurtosis, 5th moment]
```
**Reference**:  
Omar, M. (2015). *Higher-Order Statistics in Infrared Thermography*. NDT&E International.  
[DOI:10.1016/j.ndteint.2015.04.003](https://doi.org/10.1016/j.ndteint.2015.04.003)

### 5. Dynamic Mode Decomposition (DMD)
```python
modes, eigenvalues = DMD(data, truncation=10)
```
**Reference**:  
Schmid, P. (2010). *Dynamic Mode Decomposition*. JFM.  
[DOI:10.1017/jfm.2010.192](https://doi.org/10.1017/jfm.2010.192)

## Usage Example

```python
from toolbox_pulse import thermograms

# Initialize with camera resolution
analyzer = thermograms(height=512, width=640)

# Load binary thermal sequence
data = analyzer.loadfrombinfiles("path/to/thermal_sequence")

# Apply PCT analysis
EOFs, _ = analyzer.PCT(data, n_components=5)

# Generate defect mask
defect_mask = analyzer.binarize_mask(EOFs[0], threshold_value=0.7)
```

## Data Requirements
- Input sequences: 3D numpy array `(N_frames, height, width)`
- Supported bit depths: 8/16/32-bit (uint8, uint16, float32)
- Normalized temperature scales recommended

## License
MIT License - Free for academic and commercial use with attribution.
