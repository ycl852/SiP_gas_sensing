# SiP_gas_sensing
This is the code for the article

## Overview
 These codes were written for a scientific paper 'Laboratory-grade infrared spectroscopy in a centimetre-scale photonic package' and demo of the results. The codes enable reproduction of key results including spectral reconstruction and concentration prediction.

##  Usage Notice 
 - Unauthorized use of this code and database for other purpose is prohibited.
 - This code is provided for academic verification purposes only. Commercial use or application beyond the described scope requires explicit permission.
 - The database  for demo, not code, is for demonstrating the operation of the model, and unauthorized or commercial use is prohibited. 
 - For non-commercial purposes or use of the database for research, please contact the author of the paper.
 
## System requirements
 - All python codes are recommended using python IDE (PyCharm)
 - See `requirements.txt` for the list of Python packages.

## Singel-gas detection
 - Data ----Code Implementation in `dataset_single.py`----
1. Time-Domain Signals in `raw_data/20241216/O_ch4_experimentData_20241216_slidingAverage.pkl'`
2. Reference Spectra in `raw_data/20241216/commercialSpectrum_20241216_slidingAverage.pkl`
3. Gas Concentration (100 measurements per concentration)
 - Spectral reconstruction ----Code Implementation in `single_reconstruct.py`----
 - Concentration prediction ----Code Implementation in `single_predict.py`----

## Multi-gas detection
 - Data ----Code Implementation in `dataset_multi.py`----
1. Time-Domain Signals in `raw_data/20250401/threeComponentTrainingSet_sliding20250401.npy`
2. Gas Concentration (60 measurements per concentration)
 - Concentration prediction(end-to-end) ----Code Implementation in `multi_predict.py`----

 ## How to install
Code can be installed in Windows operating systems.
Download and unzip SiP_gas_sensing.zip from this repo. 
Create a virtual environment. Run the following command _in the demo directory_:
```sh
python -m venv --prompt "dtcs" .venv
source .venv/bin/activate
```
