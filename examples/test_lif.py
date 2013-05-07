"""
Basic test of the LIF neuron model
"""

import numpy as np
import numpy.random as npr

import matplotlib
import matplotlib.pyplot as plt

import nengo_numpy as nengo

################################################################################
### set up neuron model and inputs

alpha = 0.3
bias = 1.15
tau_rc = 0.02
tau_ref = 0.002

neuron_model = dict(
    type=nengo.neurons.LIFNeuronModel, tau_rc=tau_rc, tau_ref=tau_ref)

x = np.linspace(-1, 1, 10001)
# x = np.linspace(0.5, 1, 10001)

jm1 = np.maximum(alpha*x + bias - 1, 0)
y = np.zeros_like(x)
y[jm1 > 0] = 1. / (tau_ref + tau_rc*np.log1p(1./jm1[jm1 > 0]))

dt_sample = 0.01
t_final = 1.0

################################################################################
### make the model

model = nengo.Model('test')

net = model.network
input = net.make_input('input', x)
neurons = net.make_ensemble('neurons', len(x), neuron_model=neuron_model)
neurons.neurons.alpha = alpha
neurons.neurons.bias = bias

probe = net.make_probe('probe', len(x), dt=dt_sample)
probe_spikes = net.make_probe('probe2', len(x), dt=model.dt)
net.connect_neurons(input, neurons, filter=0.001)
net.connect_neurons(neurons, probe, filter=0.1)
net.connect_neurons(neurons, probe_spikes, filter=0)

model.build()

################################################################################
### run the model and plot results

model.run(t_final)

t_show = 0.5

out = probe.data
t_out = dt_sample*np.arange(len(out))

spikes = probe_spikes.data
t = model.dt*np.arange(len(spikes))

plt.figure(1)
plt.clf()
plt.plot(x, y, '--')
plt.plot(x, out[t_out > t_show].mean(0))
plt.legend(['analytic rate', 'probe measurement'], loc='upper left')


isis = [np.diff(t[s > 0]) for s in spikes.T]
rates1 = np.zeros_like(x)
rates2 = np.zeros_like(x)
for i, isi in enumerate(isis):
    if len(isi) > 0:
        rates1[i] = 1. / isi.mean()
        rates2[i] = (1. / isi).mean()

plt.figure(2)
plt.clf()
plt.plot(x, y, '--')
plt.plot(x, rates1)
plt.plot(x, rates2)
plt.legend(['analytic rate', 'ISI estimate 1', 'ISI estimate 2'], 
           loc='upper left')

plt.show()
