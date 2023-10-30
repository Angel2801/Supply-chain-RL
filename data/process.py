import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split

def divideData(dataset):
    dataset = np.array(dataset)
    # scaler = StandardScaler()
    # scaler.fit(dataset)
    # normDataset = scaler.transform(dataset)
    X_train, X_test = train_test_split(dataset, test_size=0.2, random_state=42)
    return X_train, X_test
    