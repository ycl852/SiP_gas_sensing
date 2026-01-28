import pickle
import numpy as np
from sklearn.model_selection import train_test_split
from get_data import save_to_pickle3

def extract_subset(input, output1, output2, indices):
    return (
        input[indices],
        output1[indices],
        output2[indices].reshape(-1, 1)
    )

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

with open('raw_data/20241216/O_ch4_experimentData_20241216_slidingAverage.pkl', 'rb') as f:
    data0_elect = pickle.load(f)
with open('raw_data/20241216/commercialSpectrum_20241216_slidingAverage.pkl', 'rb') as f:
    data0_light = pickle.load(f)
print(data0_elect.shape)
print(data0_light.shape)

lab = [0,
       200, 400, 600, 800, 1000,
       1200, 1400, 1600, 1800, 2000,
       2200, 2400, 2600, 2800, 3000]
train, train_target, train_lab, val, val_target, val_lab, test, test_target, test_lab = (
    create_dataset(data0_elect[0:1600], data0_light[0:1600], lab, 0.3, 0.3333, 42))
print(train.shape)
print(val.shape)
print(test.shape)
print(train_target.shape)
print(val_target.shape)
print(test_target.shape)

save_to_pickle3([train, train_target, train_lab], 'result/CH4/20241216/train_CH4.pkl')
save_to_pickle3([val, val_target, val_lab], 'result/CH4/20241216/val_CH4.pkl')
save_to_pickle3([test, test_target, test_lab], 'result/CH4/20241216/test_CH4.pkl')

