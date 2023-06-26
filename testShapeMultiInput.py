import torch
import matplotlib.pyplot as plt

from bindsnet         import encoding
from bindsnet.network import Network, nodes, topology, monitors
from bindsnet.pipeline import BasePipeline
from bindsnet.learning import Hebbian, PostPre
from custom.forceNetworkMulti import forceNetworkMulti
import numpy as np
import pickle


#Parameters
mfFeed = 1
mfInp = 2
k = 15
time = 2000
nbInput = 100

max = 2
min = 0


network = forceNetworkMulti(dt=1)  # Instantiates network.

X = nodes.Input(nbInput, traces=True)  # Input layer.
Y = nodes.LIFNodes(2000, traces=True,  tc_decay = 100, tc_trace = 5)  # Layer of LIF neurons. 
Z = nodes.LIFNodes(1200)  # Layer of LIF neurons.
F = nodes.Input(k * nbInput,  traces=True)  # Input layer.


tabTensor = torch.full((X.n, Y.n), 0.01)
C = topology.Connection(source=X, target=Y, w=torch.bernoulli(tabTensor) * torch.rand((X.n, Y.n)) * mfInp) 
tabTensor = torch.full((X.n, Y.n), 0.01)
w=torch.bernoulli(tabTensor) * torch.rand((X.n, Y.n)) * mfFeed
wFeed = torch.cat((w,w,w,w,w, w,w,w,w,w, w,w,w,w,w), 0)
print(wFeed)
C6 = topology.Connection(source=F, target=Y, w=wFeed)  # Connection from X to Y.


tabTensor = torch.full((Y.n, Y.n), 0.01)

C2 = topology.Connection(source=Y, target=Y, w=torch.bernoulli(tabTensor)* torch.rand((Y.n, Y.n)))  # Connection from X to Y.
tabTensor = torch.full((Y.n, Z.n), 0.5)
C3 = topology.Connection(source=Y, target=Z, w=torch.bernoulli(tabTensor)* torch.rand((Y.n, Z.n)))
tabTensor = torch.full((Z.n, Y.n), 0.5)
C4 = topology.Connection(source=Z, target=Y, w= -torch.bernoulli(tabTensor) * torch.rand((Z.n, Y.n))) 
tabTensor = torch.full((Z.n, Z.n), 0.01)
C5 = topology.Connection(source=Z, target=Z, w= -  torch.bernoulli(tabTensor) * torch.rand((Z.n, Z.n)))  # Connection from X to Y.


# Spike monitor objects.
M1 = monitors.Monitor(obj=X, state_vars=['s'])
M3 = monitors.Monitor(obj=F, state_vars=['s'])
M2 = monitors.Monitor(obj=Y, state_vars=['s', "v"])
M4 = monitors.Monitor(obj=Z, state_vars=['s', "v"])


# Add everything to the network object.
network.add_layer(layer=X, name='X')
network.add_layer(layer=F, name='F')
network.add_layer(layer=Y, name='Y')
network.add_layer(layer=Z, name='Z')
network.add_connection(connection=C2, source='Y', target='Y')
network.add_connection(connection=C, source='X', target='Y')
network.add_connection(connection=C3, source='Y', target='Z')
network.add_connection(connection=C4, source='Z', target='Y')
network.add_connection(connection=C5, source='Z', target='Z')
network.add_connection(connection=C6, source='F', target='Y')
#network.add_connection(connection=C7, source='X', target='Z')
#network.add_connection(connection=C8, source='F', target='Z')


network.add_monitor(monitor=M1, name='X')
network.add_monitor(monitor=M2, name='Y')
network.add_monitor(monitor=M4, name='Z')






# Create Poisson-distributed spike train inputs.
#data = 15 * torch.rand(30)  # Generate random Poisson rates for 100 input neurons.
#train = encoding.poisson(datum=data, time=251)  # Encode input as 5000ms Poisson spike trains.
with open("./datasets/gaussianEncodedMackey.pickle", "rb") as f:
    dataset = pickle.load(f)

with open("./datasets/formatedMackey.pickle", "rb") as f:
    decoDataset = pickle.load(f)

print(dataset[90][:100])
print(dataset[0][:100])

enco = torch.Tensor(dataset)
data = []
print(enco.shape)
for i in range(time):
    tmpdata = []
    for j in range(X.n):
        tmpdata.append(enco[j][i].item())
    data.append((tmpdata))

data = torch.Tensor(data)
print(len(data))



void = torch.zeros(time , k*nbInput)
print(void.shape)
# Simulate network on generated spike trains.
inputs = {'X' : data, 'F' : void}  # Create inputs mapping.
network.run(inputs=inputs, time=time, progress_bar = True, deco = decoDataset[0], min = min, max = max, k = k, nbInput = nbInput)  # Run network simulation.
# Plot spikes of input and output layers.
spikes = {'X' : M1.get('s'), 'Y' : M2.get('s'), 'Z' : M4.get('s')}
spikesY = spikes['Y']
spikesY = spikesY.reshape(time, Y.n)
count = 0
for line in spikesY:
    for value in line:
        if value == 1:
            count += 1
print("count = "  + str(count))
spikesY = spikes['Z']
spikesY = spikesY.reshape(time, Z.n)
count = 0
for line in spikesY:
    for value in line:
        if value == 1:
            count += 1
print("count = "  + str(count))


fig, axes = plt.subplots(3, 1, figsize=(12, 7))
for i, layer in enumerate(spikes):
    if layer == "X":
        spikes[layer]  = enco
    elif layer == "Y":
        spikes[layer]  = np.reshape(spikes[layer], (Y.n, time))
    elif layer == "Z":
        spikes[layer]  = np.reshape(spikes[layer], (Z.n, time))
    axes[i].matshow(spikes[layer], cmap='binary')
    axes[i].set_title('%s spikes' % layer)
    axes[i].set_xlabel('Time'); axes[i].set_ylabel('Index of neuron')
    axes[i].set_aspect('auto')

plt.tight_layout()
plt.savefig("shapeTest.jpg")

