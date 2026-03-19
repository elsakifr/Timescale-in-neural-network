from brian2 import *
import matplotlib.pyplot as plt
import numpy as np

start_scope()

# -------------------
# Settings
# -------------------
defaultclock.dt = 0.1*ms
duration = 1000*ms

N = 100

# -------------------
# Model parameters
# -------------------
tau = 20*ms
w_rec = 0.15
I_ext = 0.5

# -------------------
# Rate-based equations
# -------------------
eqs = '''
dr/dt = (-r + tanh(total_input))/tau : 1
total_input : 1
'''

G = NeuronGroup(N, eqs, method='euler')
G.r = '0.1*rand()'
G.total_input = I_ext

# recurrent weight matrix
W = np.random.normal(0, w_rec/np.sqrt(N), size=(N, N))
np.fill_diagonal(W, 0)

# Brian2 Synapses object
S = Synapses(G, G, model='w : 1', on_pre='')
S.connect()
S.w = W.flatten()

# update total input manually each timestep
@network_operation(dt=defaultclock.dt)
def update_input():
    G.total_input = I_ext + np.dot(W, G.r)

# -------------------
# Monitors
# -------------------
M = StateMonitor(G, 'r', record=True)

# -------------------
# Run
# -------------------
run(duration)

# -------------------
# Plot
# -------------------
plt.figure(figsize=(10, 6))
for i in range(5):
    plt.plot(M.t/ms, M.r[i], label=f'Neuron {i}')
plt.xlabel('Time (ms)')
plt.ylabel('Rate activity')
plt.title('Rate-based neural activity')
plt.legend()
plt.show()