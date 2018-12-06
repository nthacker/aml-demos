
import pandas as pd
import numpy as np

def get_data():
    
    data = pd.read_csv("https://devintersection.blob.core.windows.net/data/simulated-data.csv")
    
    X = data.iloc[:,4:72]
    Y = np.empty(data.shape[0], dtype=object)
    Y[:] = data.iloc[:,0].tolist()

    return { "X" : X, "y" : Y }
