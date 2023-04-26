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
mfFeed = 2
mfInp = 0.5
k = 15
time = 3000
nbInput = 100
max = 2
min = 0

dt = 1


nbExci = 2000
tc_decay_exci = 100
tc_trace_exci = 5

nbInhi = 1200


InputToExciRate = 0.01
FeedToExciRate = 0.01
ExciToInhiRate = 0.2
ExciToExciRate = 0.01
InhiToInhiRate = 0.01 
InhiToExciRate = 0.2

network = forceNetworkMulti(dt=dt)  # Instantiates network.

InputLayer = nodes.Input(nbInput, traces=True)  # Input layer.
ExciLayer = nodes.LIFNodes(nbExci, traces=True,  tc_decay = tc_decay_exci, tc_trace = tc_trace_exci)  # Layer of LIF neurons. 
InhiLayer = nodes.LIFNodes(nbInhi)  # Layer of LIF neurons.
FeedLayer = nodes.Input(k * nbInput,  traces=True)  # Input layer.


tabTensor = torch.full((InputLayer.n, ExciLayer.n), InputToExciRate)
InputToExciConnection = topology.Connection(source=InputLayer, target=ExciLayer, w=torch.bernoulli(tabTensor) * torch.rand((X.n, ExciLayer.n)) * mfInp) 


tabTensor = torch.full((InputLayer.n, ExciLayer.n), FeedToExciRate)
w=torch.bernoulli(tabTensor) * torch.rand((InputLayer.n, ExciLayer.n)) * mfFeed
wFeed =torch.bernoulli(tabTensor) * torch.rand((InputLayer.n, ExciLayer.n)) * mfFeed
for i in range(k - 1):
    wFeed = torch.cat((wFeed,w), 0)
FeedToExciConnection = topology.Connection(source=FeedLayer, target=ExciLayer, w=wFeed)  # Connection from X to Y.

tabTensor = torch.full((ExciLayer.n, ExciLayer.n), ExciToExciRate)
ExciToExciConnection = topology.Connection(source=ExciLayer, target=ExciLayer, w=torch.bernoulli(tabTensor)* torch.rand((ExciLayer.n, ExciLayer.n)))  # Connection from X to Y.


tabTensor = torch.full((ExciLayer.n, InhiLayer.n), ExciToInhiRate)
ExciToInhiConnection = topology.Connection(source=ExciLayer, target=InhiLayer, w=torch.bernoulli(tabTensor)* torch.rand((ExciLayer.n, InhiLayer.n)))


tabTensor = torch.full((InhiLayer.n, ExciLayer.n), InhiToExciRate)
InhiToExciConnection = topology.Connection(source=InhiLayer, target=ExciLayer, w= -torch.bernoulli(tabTensor) * torch.rand((InhiLayer.n, ExciLayer.n))) 


tabTensor = torch.full((InhiLayer.n, InhiLayer.n), InhiToInhiRate)
InhiToInhiConnection = topology.Connection(source=InhiLayer, target=InhiLayer, w= -  torch.bernoulli(tabTensor) * torch.rand((InhiLayer.n, InhiLayer.n)))  # Connection from X to Y.


InputMonitor = monitors.Monitor(obj=InputLayer, state_vars=['s'])
FeedMonitor = monitors.Monitor(obj=FeedLayer, state_vars=['s'])
ExciMonitor = monitors.Monitor(obj=ExciLayer, state_vars=['s', "v"])
InhiMonitor = monitors.Monitor(obj=InhiLayer, state_vars=['s', "v"])


# Add everything to the network object.
network.add_layer(layer=InputLayer, name='InputLayer')
network.add_layer(layer=FeedLayer, name='FeedLayer')
network.add_layer(layer=ExciLayer, name='ExciLayer')
network.add_layer(layer=InhiLayer, name='InhiLayer')
network.add_connection(connection=ExciToExciConnection, source='ExciLayer', target='ExciLayer')
network.add_connection(connection=InputToExciConnection, source='InputLayer', target='ExciLayer')
network.add_connection(connection=ExciToInhiConnection, source='ExciLayer', target='InhiLayer')
network.add_connection(connection=InhiToExciConnection, source='InhiLayer', target='ExciLayer')
network.add_connection(connection=InhiToInhiConnection, source='InhiLayer', target='InhiLayer')
network.add_connection(connection=FeedToExciConnection, source='FeedLayer', target='ExciLayer')

network.add_monitor(monitor=InputMonitor, name='InputMonitor')
network.add_monitor(monitor=ExciMonitor, name='ExciMonitor')
network.add_monitor(monitor=InhiMonitor, name='InhiMonitor')

with open("./datasets/gaussianEncodedMackey.pickle", "rb") as f:
    dataset = pickle.load(f)
with open("./datasets/formatedMackey.pickle", "rb") as f:
    decoDataset = pickle.load(f)

enco = torch.Tensor(dataset[0])
data = []
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
spikes = {'X' : InputMonitor.get('s'), 'Y' : ExciMonitor.get('s'), 'Z' : InhiMonitor.get('s')}
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

