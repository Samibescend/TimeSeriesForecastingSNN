import pandas as pd
import glob 
from datetime import datetime
import pickle

dataset = []

dfTab = []
alldf = glob.glob("./Stocks/*.csv")


for path in alldf:
    df = pd.read_csv(path)
    for i in range(len(df.Date)):
        df.Date[i] = datetime.strptime(df.Date[i], "%Y-%m-%d")
        df.Date[i] = df.Date[i].year

    for i in range(2006, 2017):
        dfTemp = df[df.Date == i]
        tab = dfTemp.Close.values[:300]
        dataset.append(tab)

with open("formatedStocks.pickle", "wb") as f:
    pickle.dump(dataset, f)
