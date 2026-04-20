from brian2 import *
import matplotlib.pyplot as plt
import numpy as np

# -------------------
# Helper functions
# -------------------
def autocorrelation(x):
    x = np.asarray(x)
    x = x - np.mean(x)
    corr = np.correlate(x, x, mode='full')
    corr = corr[corr.size // 2:]
    corr = corr / corr[0]
    return corr

def estimate_timescale(corr, lags, threshold=0.8):
    below = np.where(corr < threshold)[0]
    if len(below) == 0:
        return np.nan
    return lags[below[0]]

# -------------------
# Baseline settings
# -------------------
num_runs = 20
network_sizes = [50, 100, 200, 500]
duration = 3000 * ms
dt = 0.1 * ms

w_rec = 0.6
baseline_input = 0.0
pulse_amplitude = 0.8
pulse_start = 500 * ms
pulse_end = 1000 * ms
tau_baseline = 20 * ms

# -------------------
# Storage for different N
# -------------------
results_mean_activity = {}
results_std_activity = {}
results_mean_corr = {}
results_std_corr = {}
results_timescale = {}   # <-- NYTT

first_run_t = None
lags = None

# -------------------
# Loop over different network sizes
# -------------------
for N in network_sizes:
    all_corr = []
    all_mean_activity = []

    for run_id in range(num_runs):
        start_scope()

        # Reproducible but different random network each run
        seed(run_id)
        np.random.seed(run_id)

        defaultclock.dt = dt

        # -------------------
        # Rate-based equations
        # -------------------
        eqs = '''
        dr/dt = (-r + tanh(total_input))/tau_i : 1
        total_input : 1
        tau_i : second
        '''

        G = NeuronGroup(N, eqs, method='euler')
        G.r = '0.05 * rand()'
        G.tau_i = tau_baseline
        G.total_input = baseline_input

        # recurrent weight matrix
        W = np.random.normal(0, w_rec / np.sqrt(N), size=(N, N))
        np.fill_diagonal(W, 0)

        @network_operation(dt=defaultclock.dt)
        def update_input():
            if pulse_start <= defaultclock.t < pulse_end:
                input_signal = pulse_amplitude
            else:
                input_signal = baseline_input

            G.total_input = input_signal + np.dot(W, G.r)

        # monitor
        M = StateMonitor(G, 'r', record=True)

        # run simulation
        run(duration)

        # compute mean activity
        mean_activity = np.asarray(np.mean(M.r, axis=0))

        # compute autocorrelation
        corr = np.asarray(autocorrelation(mean_activity))

        if first_run_t is None:
            first_run_t = M.t / ms
            lags = np.arange(len(corr)) * float(defaultclock.dt / ms)

        all_mean_activity.append(mean_activity)
        all_corr.append(corr[:1000])

    # -------------------
    # Convert to arrays
    # -------------------
    all_mean_activity = np.array(all_mean_activity)
    all_corr = np.array(all_corr)

    # Mean and std across runs
    results_mean_activity[N] = np.mean(all_mean_activity, axis=0)
    results_std_activity[N] = np.std(all_mean_activity, axis=0)

    results_mean_corr[N] = np.mean(all_corr, axis=0)
    results_std_corr[N] = np.std(all_corr, axis=0)

    # -------------------
    # Estimate timescale for this N
    # -------------------
    results_timescale[N] = estimate_timescale(results_mean_corr[N], lags[:1000], threshold=0.8)

# -------------------
# Plot 1: mean network activity for different network sizes
# -------------------
plt.figure(figsize=(10, 6))
for N in network_sizes:
    plt.plot(first_run_t, results_mean_activity[N], label=f'N = {N}')
plt.axvspan(pulse_start / ms, pulse_end / ms, alpha=0.2, label='Input pulse')
plt.xlabel('Time (ms)')
plt.ylabel('Mean activity')
plt.title('Effect of network size: mean population activity')
plt.legend()
plt.tight_layout()

# -------------------
# Plot 2: autocorrelation for different network sizes
# -------------------
plt.figure(figsize=(10, 6))
for N in network_sizes:
    plt.plot(lags[:1000], results_mean_corr[N], label=f'N = {N}')
plt.xlabel('Lag (ms)')
plt.ylabel('Autocorrelation')
plt.title('Effect of network size: autocorrelation of mean population activity')
plt.legend()
plt.tight_layout()

# -------------------
# Plot 3: estimated timescale vs network size
# -------------------
plt.figure(figsize=(8, 5))
Ns = list(results_timescale.keys())
taus = [results_timescale[N] for N in Ns]

plt.plot(Ns, taus, 'o-')
plt.xlabel('Network size (N)')
plt.ylabel('Estimated timescale (ms)')
plt.title('Effect of network size: estimated intrinsic timescale')
plt.tight_layout()

plt.show()

# Optional: print values in terminal
print("Estimated timescales:")
for N in network_sizes:
    print(f"N = {N}: {results_timescale[N]:.2f} ms")