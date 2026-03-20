from brian2 import *
import numpy as np
import matplotlib.pyplot as plt

start_scope()

# Parametrar
N = 100
tau_value = 20*ms
duration = 1000*ms
defaultclock.dt = 1*ms

# Enkel extern input
I0 = 0.5

# Rate-modell
eqs = '''
dr/dt = (-r + tanh(x)) / tau : 1
x : 1
tau : second
Iext : 1
'''

G = NeuronGroup(N, eqs, method='euler')

# Initialvärden
G.r = '0.1 * rand()'
G.tau = tau_value
G.Iext = I0

# Viktmatris
W = 0.1 * np.random.randn(N, N)

# Manuell uppdatering av recurrent input
@network_operation(dt=defaultclock.dt)
def update_input():
    G.x = W @ G.r[:] + G.Iext[:]

# Monitor
M = StateMonitor(G, 'r', record=True)

run(duration)

# Plot några neuroner
plt.figure(figsize=(8,4))
for i in range(5):
    plt.plot(M.t/ms, M.r[i], label=f'Neuron {i}')
plt.xlabel('Tid (ms)')
plt.ylabel('Aktivitet r')
plt.title('Homogent rate-nätverk')
plt.show()