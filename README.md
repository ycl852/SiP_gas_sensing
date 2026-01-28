# SiP_gas_sensing
This is the code for the article

## Overview
 These codes were written for a scientific paper 'Laboratory-grade infrared spectroscopy in a centimetre-scale photonic package' and demo of the results. The codes enable reproduction of key results including data preprocessing, spectral reconstruction and concentration prediction.

##  Usage Notice 
 - Unauthorized use of this code and database for other purpose is prohibited.
 - This code is provided for academic verification purposes only. Commercial use or application beyond the described scope requires explicit permission.
 - The database  for demo, not code, is for demonstrating the operation of the model, and unauthorized or commercial use is prohibited. 
 - For non-commercial purposes or use of the database for research, please contact the author of the paper.
 
## System requirements
 - All python codes are recommended using python IDE (PyCharm)
环境的要求

## Raw data
1. Time-Domain Signals in `raw_data/20241216/O_ch4_experimentData_20241216.txt'`
2. Reference Spectra in `raw_data/20241216/commercialSpectrum_20241216.txt"`
3. Gas Concentration (100 measurements per concentration)
lab = [0, 200, 400, 600, 800, 1000,
       1200, 1400, 1600, 1800, 2000,
       2200, 2400, 2600, 2800, 3000]

 ## Dataset construction
Code Implementation in  `dataset_single.py`
<pre><code>
 def create_dataset(data_elect, data_light, lab, test_size1, test_size2, random_state):
    num = np.arange(data_elect.shape[0])
    train_index, a1 = train_test_split(num, test_size=test_size1, random_state=random_state)
    val_index, test_index = train_test_split(a1, test_size=test_size2, random_state=random_state)
    label = np.array(lab).repeat(100)

    train_data, train_target, train_lab = extract_subset(data_elect, data_light, label, train_index)
    val_data, val_target, val_lab = extract_subset(data_elect, data_light, label, val_index)
    test_data, test_target, test_lab = extract_subset(data_elect, data_light, label, test_index)

    return (train_data, train_target, train_lab,
            val_data, val_target, val_lab,
            test_data, test_target, test_lab)
            
def create_dataset_test(test_data, test_target, test_lab):
    num = np.arange(test_data.shape[0])
    test_label = np.array(test_lab).repeat(100)
    np.random.shuffle(num)
 
    return test_data[num], test_target[num], test_label[num]
</code></pre>

 ## Model architecture
1. Spectral Reconstruction Model in `model_cnn_eca_lstm_mlp.py`
2. Concentration Prediction Model in `model_mlp.py`

 ## Training
1. Spectral Reconstruction Training in `train_single_reconstruct.py`
2. Concentration Prediction Training in `train_single_predict.py`

 ## bash语言，运行代码的命令
