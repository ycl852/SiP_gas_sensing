import numpy as np
from sklearn.model_selection import train_test_split
from get_data import save_to_pickle2

def extract_subset(input, output, indices):
    return (
        input[indices],
        output[indices].reshape(-1, 3))

def create_dataset(data_elect, label, test_size1, test_size2, random_state):
    num = np.arange(data_elect.shape[0])
    train_index, a1 = train_test_split(num, test_size=test_size1, random_state=random_state)
    val_index, test_index = train_test_split(a1, test_size=test_size2, random_state=random_state)

    train_data, train_lab = extract_subset(data_elect, label, train_index)
    val_data, val_lab = extract_subset(data_elect, label, val_index)
    test_data, test_lab = extract_subset(data_elect, label, test_index)

    return (train_data, train_lab,
            val_data, val_lab,
            test_data, test_lab)

def create_dataset_test(test_data, test_lab):
    num = np.arange(test_data.shape[0])
    test_label = np.array(test_lab).repeat(100)
    np.random.shuffle(num)

    return test_data[num], test_label[num]

data0_elect = np.load('raw_data/20250401/threeComponentTrainingSet_sliding20250401.npy')
print(data0_elect.shape)

lab_CH4 = [0, 500, 1000, 1500, 2000]
lab_CO = [0, 500, 1000, 1500, 2000]
lab_NH3 = [0, 500, 1000, 1500, 2000]

train_label = []
for i in range(5):
    a = lab_CH4[i]
    for j in range(5):
        b = lab_CO[j]
        for n in range(5):
            c = lab_NH3[n]
            lab = [a, b, c]
            train_label.append(lab)
train_label = np.array(train_label).repeat([60]*125, axis=0)
print(train_label.shape)

data = data0_elect
train, train_lab, val, val_lab, test, test_lab = create_dataset(data[0:7500], train_label, 0.3, 0.3333, 42)
print(train.shape)
print(train_lab.shape)
print(val.shape)
print(val_lab.shape)
print(test.shape)
print(test_lab.shape)
save_to_pickle2([train, train_lab], 'result/CH4_CO_NH3/20250401/train_slid.pkl')
save_to_pickle2([val, val_lab], 'result/CH4_CO_NH3/20250401/val_slid.pkl')
save_to_pickle2([test, test_lab], 'result/CH4_CO_NH3/20250401/test_slid.pkl')

