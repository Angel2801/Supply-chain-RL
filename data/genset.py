import numpy as np
import random

def generateDataSet():
    dataset = []
    for _ in range(20001):
        seed = random.randint(80,2000)
        np.random.seed(seed)
        factory = np.random.randint(1,16,dtype=np.int32)
        warehouse = np.random.randint(1,6,dtype=np.int32)
        product = np.random.randint(1,6,dtype=np.int32)
        avail = np.random.randint(800,2001,dtype=np.int32)
        dispatched = np.random.randint(600,1001,dtype=np.int32)
        demand = np.random.randint(1,avail+dispatched,dtype=np.int32)
        productionCost = np.random.randint(200,601,dtype=np.int32)
        transportCost = np.random.randint(10000,20001,dtype=np.int32)
        dataset.append([factory, warehouse, product, avail, dispatched, demand, productionCost, transportCost])  
    print(type(dataset))
    return dataset