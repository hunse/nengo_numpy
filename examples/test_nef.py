
import numpy as np
import numpy.random as npr

import matplotlib
import matplotlib.pyplot as plt

import nengo_numpy as nengo

##################################################
### make inputs

model = nengo.Model('test')
net = model.network

t_final = 3.0
dt_sample = 0.01

t = model.dt*np.arange(t_final/model.dt)
x = np.linspace(-1, 1, len(t))

##################################################
### make model

input = net.make_input('input', dict(zip(t,x)))

neurons = net.make_ensemble('neurons', 100)

probe1 = net.make_probe('probe1', 1, dt=dt_sample)
probe2 = net.make_probe('probe2', 1, dt=dt_sample)

net.connect(input, neurons, filter=0.005)

net.connect(input, probe1, filter=0.03)
net.connect(neurons, probe2, filter=0.03)

model.build()

##################################################
### run model and plot results

model.run(t_final)

ins = probe1.data
outs = probe2.data

t = dt_sample*np.arange(len(outs))

plt.figure(1)
plt.clf()

plt.plot(t, ins)
plt.plot(t, outs)
plt.legend(['input', 'output'], loc=2)

plt.show()
