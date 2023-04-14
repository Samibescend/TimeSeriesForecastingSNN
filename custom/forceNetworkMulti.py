from bindsnet.network import Network
from typing import Dict, Iterable, Optional, Type
import torch
import numpy as np
import random
from statistics import NormalDist
import numpy.random as random
import matplotlib.pyplot as plt


def encodingOneValue(data, ymax, ymin, m, sca):
    centers = []
    spikes = []
    deviation = sca * ((ymax - ymin) / (m - 2))
    for i in range(m):
        centers.append(ymin + (i) * ((ymax - ymin) / (m - 2)))
    for i in range(len(centers)):
        if data <= centers[i]:
            if (NormalDist(centers[i], deviation).cdf(data) >= 0.2):
                spikes.append(1)
            else:
                spikes.append(0)
        else:
            tmpvalue = centers[i] - (data - centers[i])
            if (NormalDist(centers[i], deviation).cdf(tmpvalue) >= 0.2):
                    spikes.append(1)
            else:
                spikes.append(0)    
    return spikes

class forceNetworkMulti(Network):
    

    def run(
        self, inputs: Dict[str, torch.Tensor], time: int, one_step=False, **kwargs
    ) -> None:
        # language=rst
        # Check input type
        assert type(inputs) == dict, (
            "'inputs' must be a dict of names of layers "
            + f"(str) and relevant input tensors. Got {type(inputs).__name__} instead."
        )
        # Parse keyword arguments.
        clamps = kwargs.get("clamp", {})
        unclamps = kwargs.get("unclamp", {})
        masks = kwargs.get("masks", {})
        injects_v = kwargs.get("injects_v", {})
        decoData = kwargs.get("deco", {})
        min = kwargs.get("min", {})
        max = kwargs.get("max", {})
        k = kwargs.get("k",  {})
        print(k)
        plotValues = []

        nbNeur = self.layers["Y"].n
        traces = np.zeros((time, nbNeur))
        tmpT = 0 #Pour le calcul de la moyenne des errors
        incr = 0.5
        eta = 0.2
        maxWOut = max/(0.1*nbNeur)
        minWOut = min/(0.1*nbNeur)
        traceDecay = 5
        wOut = torch.ones(nbNeur) 
        #print(inputs["F"])
         
        #inputs["F"] = (torch.zeros(30))
        #print(inputs["F"][0])
        baseTrace = np.ones(nbNeur)
        totalError = 0
        #ReadOutWeightsMatrix : number of output neuron * number of reservoir neuron
        # ou peut etre Time * number of output channels attached to the reservoir

        

        # Compute reward.
        if self.reward_fn is not None:
            kwargs["reward"] = self.reward_fn.compute(**kwargs)

        # Dynamic setting of batch size.
        if inputs != {}:
            for key in inputs:
                # goal shape is [time, batch, n_0, ...]
                if len(inputs[key].size()) == 1:
                    # current shape is [n_0, ...]
                    # unsqueeze twice to make [1, 1, n_0, ...]
                    inputs[key] = inputs[key].unsqueeze(0).unsqueeze(0)
                elif len(inputs[key].size()) == 2:
                    # current shape is [time, n_0, ...]
                    # unsqueeze dim 1 so that we have
                    # [time, 1, n_0, ...]
                    inputs[key] = inputs[key].unsqueeze(1)

            for key in inputs:
                # batch dimension is 1, grab this and use for batch size
                if inputs[key].size(1) != self.batch_size:
                    self.batch_size = inputs[key].size(1)

                    for l in self.layers:
                        self.layers[l].set_batch_size(self.batch_size)

                    for m in self.monitors:
                        self.monitors[m].reset_state_variables()

                break

        # Effective number of timesteps.
        timesteps = int(time / self.dt)

        # Simulate network activity for `time` timesteps.
        for t in range(timesteps):

            # Get input to all layers (synchronous mode).
            print(t)
            current_inputs = {}
            if not one_step:
                current_inputs.update(self._get_inputs())

            for l in self.layers:
                # Update each layer of nodes.
                if l in inputs:
                    if l in current_inputs:
                        current_inputs[l] += inputs[l][t]
                    else:
                        current_inputs[l] = inputs[l][t]

                if one_step:
                    # Get input to this layer (one-step mode).
                    current_inputs.update(self._get_inputs(layers=[l]))

                # Inject voltage to neurons.
                inject_v = injects_v.get(l, None)
                if inject_v is not None:
                    if inject_v.ndimension() == 1:
                        self.layers[l].v += inject_v
                    else:
                        self.layers[l].v += inject_v[t]

                if l in current_inputs:
                    self.layers[l].forward(x=current_inputs[l])
                else:
                    self.layers[l].forward(
                        x=torch.zeros(
                            self.layers[l].s.shape, device=self.layers[l].s.device
                        )
                    )

                # Clamp neurons to spike.
                clamp = clamps.get(l, None)
                if clamp is not None:
                    if clamp.ndimension() == 1:
                        self.layers[l].s[:, clamp] = 1
                    else:
                        self.layers[l].s[:, clamp[t]] = 1

                # Clamp neurons not to spike.
                unclamp = unclamps.get(l, None)
                if unclamp is not None:
                    if unclamp.ndimension() == 1:
                        self.layers[l].s[:, unclamp] = 0
                    else:
                        self.layers[l].s[:, unclamp[t]] = 0

            # Run synapse updates.
            if "a_minus" in kwargs:
                A_Minus = kwargs["a_minus"]
                kwargs.pop("a_minus")
                if isinstance(A_Minus, dict):
                    A_MD = True
                else:
                    A_MD = False
            else:
                A_Minus = None

            if "a_plus" in kwargs:
                A_Plus = kwargs["a_plus"]
                kwargs.pop("a_plus")
                if isinstance(A_Plus, dict):
                    A_PD = True
                else:
                    A_PD = False
            else:
                A_Plus = None

            for c in self.connections:
                if A_Minus != None and ((isinstance(A_Minus, float)) or (c in A_Minus)):
                    if A_MD:
                        kwargs["a_minus"] = A_Minus[c]
                    else:
                        kwargs["a_minus"] = A_Minus

                if A_Plus != None and ((isinstance(A_Plus, float)) or (c in A_Plus)):
                    if A_PD:
                        kwargs["a_plus"] = A_Plus[c]
                    else:
                        kwargs["a_plus"] = A_Plus

                self.connections[c].update(
                    mask=masks.get(c, None), learning=self.learning, **kwargs
                )

            # # Get input to all layers.
            #current_inputs.update(self._get_inputs())
            # Record state variables of interest.
            for m in self.monitors:
                self.monitors[m].record()
            #print(self.layers["Y"].x[0])
            if t != 0:
                i = 0
                for s in self.monitors["Y"].recording["s"][t][0][0]:
                    if baseTrace[i] == 1:
                        traces[t][i] = 0
                        if s.item() == True:
                            baseTrace[i] = t
                            traces[t][i] = traces[t][i]  + incr
                            #traces[t][i] =  incr

                    else: 
                        traces[t][i] = np.exp((-1 *((t - baseTrace[i])) / traceDecay))
                        if s.item() == True:
                            print("incr")
                            baseTrace[i] = t
                            traces[t][i] = traces[t][i]  + incr
                            #traces[t][i] =  incr
                    i += 1
                # print(traces[t])

            if t == 0:
                wOut = ((max - min) * torch.rand(1, nbNeur) + min ) * wOut
                QT = torch.eye(nbNeur)/eta
                pred  = random.uniform(min, max)
                plotValues.append(pred)
            else:
                QT = QT - ((QT @ torch.Tensor(traces[t]) @ torch.t(torch.Tensor(traces[t])) * QT ) / (1 + torch.t(torch.Tensor(traces[t]) @ QT @ torch.Tensor(traces[t]))))
                #print(wOut)
                #print(traces[t])
                pred = (wOut) @ traces[t]
                plotValues.append(pred[0])
              
            error = float((decoData[t] - pred ))#torch.mean(wOut)
            totalError += (error ** 2)
            tmpT += 1
            print("Prévision = " + str(pred)) 
            print("MSE = " + str(error ** 2))
            print("Value = " + str(decoData[t]))
            print("Moyenne Error = " + str(totalError/(tmpT)))
            #print(traces[t])
            if t != 0:
               wOut = wOut - error *  QT @ torch.t(torch.Tensor(traces[t]))
            if t % 1000 == 0:
                totalError = 0
                tmpT = 0
            print(inputs["F"].shape)
            for i in range(k):
                if t - i < 0:
                    tmp = torch.zeros(30)
                else:
                    tmp = torch.Tensor(encodingOneValue(decoData[t-i], max, min, 30, 1))
                for j in range(30):
                    inputs["F"][t][0][j + i * 30] = tmp[j]

        # Re-normalize connections.
        for c in self.connections:
            self.connections[c].normalize()
        t = np.arange(0.0, time, 1)
        s = plotValues
        fig, ax = plt.subplots()
        ax.plot(t, s)
        fig.savefig("plotPred.png")

    
