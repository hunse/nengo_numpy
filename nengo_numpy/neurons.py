
import numpy as np
import numpy.random

def dist_or_list(arg, size):
    if isinstance(arg, tuple):
        ### distribution, generate random uniforms
        return numpy.random.uniform(size=size, low=arg[0], high=arg[1])
    else:
        ### list or list-like, make sure it's an array
        return np.asarray(arg)


class LIFNeuronModel(object):

    dt = 0.001

    def __init__(self, n_neurons, mode, tau_rc=0.02, tau_ref=0.002):

        self._cache = {}

        self.size = n_neurons
        self.mode = mode
        assert mode in ['spiking', 'bigmodel'],\
            "Non-spiking neurons not yet implemented"

        self.alpha = None
        self.bias = None

        self.tau_rc = tau_rc
        self.tau_ref = tau_ref

        self.v = None
        self.w = None

        self.t = 0

    def rate_intercept_to_alpha_bias(self, max_rate, intercept, radius):
        """Compute the alpha and bias needed to get the given max_rate
        and intercept values.

        Returns gain (alpha) and offset (j_bias) values of neurons.

        :param float array max_rates: maximum firing rates of neurons
        :param float array intercepts: x-intercepts of neurons
        """

        max_rates = dist_or_list(max_rate, self.size)
        intercepts = dist_or_list(intercept, self.size)

        x1 = intercepts
        x2 = radius
        z1 = 1.
        z2 = 1. / (1 - np.exp((self.tau_ref - 1./max_rates)/self.tau_rc))
        alpha = (z1 - z2) / (x1 - x2)
        bias = z1 - alpha * x1

        return alpha, bias

    def set_rate_intercept(self, max_rate, intercept, radius):
        self.alpha, self.bias = self.rate_intercept_to_alpha_bias(
            max_rate=max_rate, intercept=intercept, radius=radius)

    def rate(self, inputs):
        """Analytically compute the firing rates for constant input values."""
        j = np.maximum(self.alpha*inputs + self.bias - 1, 0.0)
        r = 1. / (self.tau_ref + self.tau_rc*np.log1p(1./j))
        return r

    def reset(self):
        self._cache.clear()
        self.t = 0
        if self.v is not None: self.v[:] = 0
        if self.w is not None: self.w[:] = 0

    def reset_input(self):
        self.input[:] = 0

    def build(self, dtype):
        self.dtype = dtype
        self.v = np.zeros(self.size, dtype=dtype)
        self.w = np.zeros(self.size, dtype=dtype)
        self.input = np.zeros(self.size, dtype=dtype)
        self.spikes = np.zeros(self.size, dtype=dtype)

        self.alpha = np.zeros(self.size, dtype=dtype)
        self.bias = np.zeros(self.size, dtype=dtype)

        self.tau_rc = np.zeros(self.size, dtype=dtype)
        self.tau_ref = np.zeros(self.size, dtype=dtype)

    def link_model(self, model, i):
        assert isinstance(model, LIFNeuronModel), "Model must be the same type"
        n = model.size

        ### copy over parameters from sub-model
        if model.v is not None: self.v[i:i+n] = model.v
        if model.w is not None: self.w[i:i+n] = model.w

        self.alpha[i:i+n] = model.alpha
        self.bias[i:i+n] = model.bias
        self.tau_rc[i:i+n] = model.tau_rc
        self.tau_ref[i:i+n] = model.tau_ref

        ### link parameters back to sub-model
        model.input = self.input[i:i+n]
        model.spikes = self.spikes[i:i+n]
        model.v = self.v[i:i+n]
        model.w = self.w[i:i+n]

    def run(self, t_end):

        if 'dt/tau_rc' not in self._cache:
            self._cache['dt/tau_rc'] = self.dt / self.tau_rc
        if 'tau_ref/dt' not in self._cache:
            self._cache['tau_ref/dt'] = self.tau_ref / self.dt

        self.spikes[:] = 0

        while self.t < t_end - 0.5*self.dt:
            self.t += self.dt

            ### Euler's method
            dV = self._cache['dt/tau_rc'] * (self.alpha*self.input + self.bias - self.v)

            ### increase the voltage, ignore values below 0
            self.v[:] = np.maximum(self.v + dV, 0)

            if 0:
                ### basic method: no overshoot approximation
                ### handle refractory period
                self.w -= self.dt
                # self.v[self.w > 1e-6] = 0
                self.v *= (self.w < 1e-6)

                ### determine which neurons spike
                spike = self.v > 1
                self.spikes += spike

                self.w[spike] = self.tau_ref[spike] + self.dt

            elif 0:
                ### Terry's method: overshoot approximation in seconds
                ### handle refractory period
                self.w -= self.dt
                self.v *= (1. - self.w/self.dt).clip(0.0, 1.0)

                ### determine which neurons spike
                spike = self.v > 1
                self.spikes += spike

                ### linearly approximate time since neuron crossed spike threshold
                overshoot = (self.v - 1) * (self.dt / dV)
                self.w[spike] = (self.tau_ref - overshoot + 1.5*self.dt)[spike]
                ### EH: adding 1.5*dt seems to have empirical benefit (matches analytic curve better)

            else:
                ### Eric's method: overshoot approximation in dt units
                ### handle refractory period
                self.w -= 1.
                self.v *= (1. - self.w).clip(0.0, 1.0)

                ### determine which neurons spike
                spike = self.v > 1
                self.spikes += spike

                ### linearly approximate time since neuron crossed spike threshold
                overshoot = (self.v - 1) / dV
                self.w[spike] = (self._cache['tau_ref/dt'] - overshoot + 1.5)[spike]
                ### EH: adding 1.5 seems to have empirical benefit (matches analytic curve better)

                # self.v *= (1 - spike)



    # def run_theano(self, t_end):
    #     import theano
    #     import theano.tensor as T

    #     # tau_rc =
