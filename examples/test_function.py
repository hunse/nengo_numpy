"""
Test whether NEF computation of a function works
"""

import itertools

import numpy as np
import numpy.random as npr

import matplotlib
import matplotlib.pyplot as plt

import nengo_numpy as nengo

##################################################
### make inputs

model = nengo.Model('test')
net = model.network

t_final = 5.0
dt_sample = 0.01

t = model.dt*np.arange(t_final/model.dt)
x = np.linspace(-1, 1, len(t))

def func1(x):
    return -np.abs(x)

def func2(x):
    return x**2

# def func2(x):
#     return np.sign(x)

##################################################
### make model

input = net.make_input('input', dict(zip(t,x)))

neurons = net.make_ensemble('neurons', 100)

probe0 = net.make_probe('probe0', 1, dt=dt_sample)
probe1 = net.make_probe('probe1', 1, dt=dt_sample)
probe2 = net.make_probe('probe2', 1, dt=dt_sample)

net.connect(input, neurons, filter=0.005)

net.connect(input, probe0, filter=0.03)
net.connect(neurons, probe1, function=func1, filter=0.05)
net.connect(neurons, probe2, function=func2, filter=0.05)

model.build()

##################################################
### run model and plot results

model.run(t_final)

ins = probe0.data
outs1 = probe1.data
outs2 = probe2.data

t = dt_sample*np.arange(len(outs1))

plt.figure(1)
plt.clf()

plt.plot(t, ins, 'k')

colors = itertools.cycle(matplotlib.rcParams['axes.color_cycle'])
for func, out in zip([func1, func2], [outs1, outs2]):
    color = colors.next()
    plt.plot(t, func(ins), '--', color=color)
    plt.plot(t, out, color=color)

# plt.legend(['input', 'output'], loc=4)
plt.show()
