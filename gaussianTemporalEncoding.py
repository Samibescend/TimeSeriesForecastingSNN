import pickle
import numpy as np
from statistics import NormalDist
#with open("./datasets/formatedStocks.pickle", "rb") as f:
with open("./datasets/formatedMackey2.pickle", "rb") as f:
    dataset = pickle.load(f)

def encoding(dataset, ymax, ymin, m, sca):
    data = dataset[0]
    spikesTab = []
    for i in range(m):
        spikes = []
        for j in range(len(data)):
            spikes.append(0)
        spikesTab.append(spikes)
    centers = []
    deviation = sca * ((ymax - ymin) / (m - 2))
    for i in range(m):
        centers.append(ymin + (i) * ((ymax - ymin) / (m - 2)))
    for i in range(m):
        for j in range(len(data)):
            if data[j] <= centers[i]:
                proba = NormalDist(centers[i], deviation).cdf(data[j])
                if (proba >= 0.3):

                    index = j + 10 - round(proba * 10) * 2
                    if index < len(data):
                        spikesTab[i][index] = 1
            else:
                tmpvalue = centers[i] - (data[j] - centers[i])
                proba = NormalDist(centers[i], deviation).cdf(tmpvalue)
                if (proba >= 0.3):
                    index = j + 10 - round(proba * 10) * 2
                    if index < len(data):
                        spikesTab[i][index] = 1   
    print(spikesTab[0][:100])
    print(spikesTab[90][:100])
    return spikesTab


def encodingOneValue(data, ymax, ymin, m, sca):
    centers = []
    spikes = []
    deviation = sca * ((ymax - ymin) / (m - 2))
    for i in range(m):
        centers.append(ymin + (i) * ((ymax - ymin) / (m - 2)))
    for i in range(len(centers)):
        if data <= centers[i]:
            if (NormalDist(centers[i], deviation).cdf(data) >= 0.2):
                spikes[i].append(1)
            else:
                spikes[i].append(0)
        else:
            tmpvalue = centers[i] - (data - centers[i])
            if (NormalDist(centers[i], deviation).cdf(tmpvalue) >= 0.2):
                    spikes[i].append(1)
            else:
                spikes[i].append(0)    
    return spikes



"""
results = (encoding(dataset, 1000, 0, 30, 1))
print(len(results[0]))
with open("./datasets/gaussianEncodedStocks.pickle", "wb") as f:
    pickle.dump(results, f)
"""

nbInputs = 100
results = (encoding(dataset, 2, 0, nbInputs, 1))
print(len(results[0]))
with open("./datasets/gaussianEncodedMackey2.pickle", "wb") as f:
    pickle.dump(results, f)
