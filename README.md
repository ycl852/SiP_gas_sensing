# SiP_gas_sensing
This is the code for the article

## Overview
 These codes were written for a scientific paper 'Laboratory-grade infrared spectroscopy in a centimetre-scale photonic package' and demo of the results. The codes enable reproduction of key results including spectral reconstruction and concentration prediction.

##  Usage notice 
 - Unauthorized use of this code and database for other purpose is prohibited.
 - This code is provided for academic verification purposes only. Commercial use or application beyond the described scope requires explicit permission.
 - The database  for demo, not code, is for demonstrating the operation of the model, and unauthorized or commercial use is prohibited. 
 - For non-commercial purposes or use of the database for research, please contact the author of the paper.

### Software dependencies and operating systems (including version numbers)
- **Operating systems**: Windows 10 (64-bit)
- **Key packages**:
  - python == 3.9.12
  - tensorflow == 2.15.0
  - keras == 3.9.2
  - numpy
  - scikit-learn
  - matplotlib

### Any required non‑standard hardware
- None (runs on any standard desktop/laptop).  

 ## How to install
Code can be installed in Windows operating systems.
1. Download and unzip SiP_gas_sensing.zip from this repo：
```sh
cd /path/to/pythonSiP
```

2. Create a virtual environment. Run the following command:
```sh
python -m venv .venv
```

3. Activate virtual environment：
```sh
source .venv/Scripts/activate
# Note: The command above works in Git Bash or WSL.
```

4. Verify environment：
```sh
which python
which pip
# Expected output should point to .venv directory.
```

5.  Update pip in virtual environment：
```sh
python -m ensurepip --upgrade
```

6. Install dependencies：
```sh
pip install tensorflow==2.15.0
pip install numpy scikit-learn matplotlib
pip install keras==3.9.2
```

7. Verify installation：
```sh
python -c "import tensorflow as tf; print(f'TensorFlow {tf.__version__} installed')"
python -c "import keras; print(f'Keras {keras.__version__} installed')"
```

8. Typical install time：
```sh
~15-30 minutes of full install (depending on download speed and disk performance).
```

9. Demo：
After completing the installation, run the prediction script:
```sh
python single_reconstruct.py
```
 Expected output:
 ```
 Loading model...
 Processing sample 1/5 ...
 Reconstruction error: 0.0234
 ...
 All predictions saved to output/predictions.csv
 ```

10. Expected run time for demo:
```sh
~20–40 seconds/epoch (depending on CPU and hardware)
```


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
