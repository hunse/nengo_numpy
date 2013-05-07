"""
Test circular de-convolution (i.e. correlation) with a fixed vector
"""

import itertools

import numpy as np
import numpy.random as npr

import matplotlib
import matplotlib.pyplot as plt

import nengo_numpy as nengo

def norm(x, axis=-1):
    return np.sqrt((x**2).sum(axis))

def convolve(a, b, invert_a=False, invert_b=False):
    A = np.fft.fft(a, axis=-1)
    B = np.fft.fft(b, axis=-1)
    if invert_a:
        A = A.conj()
    if invert_b:
        B = B.conj()
    C = A * B
    c = np.fft.ifft(C, axis=-1)
    return c.real

def deconvolve(a, b):
    return convolve(a, b, invert_a=True)

################################################################################
### make inputs and find deconvolution matrix

n = 5
# n_per_d = 50
n_per_d = 100
radius = 1
a = np.random.normal(size=n, scale=1./np.sqrt(n))
b = np.random.normal(size=n, scale=1./np.sqrt(n))
d = deconvolve(a, b)

N = np.array([np.roll(b, -i) for i in xrange(n)]) # deconvolution matrix

dd = np.dot(a, N.T)
assert np.allclose(d, dd) # sanity check

################################################################################
### make model

dt_sample = 0.01
t_final = 1.0

model = nengo.Model('test')
net = model.network

inputs = []
bound = []
decoded = []
probes = []
for i in xrange(n):
    inputs.append(net.make_input('input%d' % i, a[i]))
    decoded.append(net.make_ensemble('decoded%d' % i, n_per_d, radius=radius))
    probes.append(net.make_probe('probe%d' % i, 1, dt=dt_sample))

net.connect(inputs, decoded, transform=N.T, filter=0.005)
net.connect(decoded, probes, filter=0.05)

model.build()

################################################################################
### run model and plot results

model.run(t_final)

outs = np.array([probe.data.flatten() for probe in probes])
t = np.linspace(0, t_final, t_final/dt_sample + 1)

plt.figure(1)
plt.clf()

colors = itertools.cycle(matplotlib.rcParams['axes.color_cycle'])
for dd, out in zip(d, outs):
    color = colors.next()
    plt.plot(t, dd*np.ones_like(t), '--', color=color)
    plt.plot(t, out, color=color)

plt.show()
