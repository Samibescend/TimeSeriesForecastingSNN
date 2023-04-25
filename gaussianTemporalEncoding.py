import pickle
import numpy as np
from statistics import NormalDist
#with open("./datasets/formatedStocks.pickle", "rb") as f:
with open("./datasets/formatedMackey2.pickle", "rb") as f:
    dataset = pickle.load(f)

def encoding(dataset, ymax, ymin, m, sca):
    spikesTab = []
    centers = []
    deviation = sca * ((ymax - ymin) / (m - 2))
    for i in range(m):
        centers.append(ymin + (i) * ((ymax - ymin) / (m - 2)))
    for data in dataset:
        spikes = []
        for i in range(m):
            spikes.append([])

        for i in range(m):
            for value in data:
                if value <= centers[i]:
                    if (NormalDist(centers[i], deviation).cdf(value) >= 0.2):
                        #print(value)
                        #print(NormalDist(centers[i], deviation))
                        #print(NormalDist(centers[i], deviation).cdf(value))
                        #print(i)
                        spikes[i].append(1)
                    else:
                        spikes[i].append(0)
                else:
                    tmpvalue = centers[i] - (value - centers[i])
                    if (NormalDist(centers[i], deviation).cdf(tmpvalue) >= 0.2):
                        #print(value)
                        #print(NormalDist(centers[i], deviation))
                        #print(NormalDist(centers[i], deviation).cdf(value))
                        #print(i)
                        spikes[i].append(1)
                    else:
                        spikes[i].append(0)
        spikesTab.append(spikes)
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
