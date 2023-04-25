import pandas as pd
import glob 
from datetime import datetime
import pickle

"""df = pd.read_csv("./datasets/Mackey/mackey.csv")
data = [df['value'].values]
with open("formatedMackey.pickle", "wb") as f:
    pickle.dump(data, f)
"""


df = pd.read_csv("./datasets/Mackey/mgdata.txt", sep = " ", names=["index", "value"])
data = [df['value'].values]
with open("formatedMackey2.pickle", "wb") as f:
    pickle.dump(data, f)
