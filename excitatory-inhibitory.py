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
ei_ratios = [0.8, 0.7, 0.6, 0.5]   # fraction excitatory
N = 100
duration = 3000 * ms
dt = 0.1 * ms

baseline_input = 0.0
pulse_amplitude = 0.8
pulse_start = 500 * ms
pulse_end = 1000 * ms
tau_baseline = 20 * ms

w_exc = 0.6
w_inh = 0.6

# -------------------
# Storage
# -------------------
results_mean_activity = {}
results_std_activity = {}
results_mean_corr = {}
results_std_corr = {}
results_timescale = {}
results_time = {}
results_lags = {}

# -------------------
# Loop over different E/I balances
# -------------------
for ratio in ei_ratios:
    all_corr = []
    all_mean_activity = []
    current_time = None
    current_lags = None
    expected_len = None

    for run_id in range(num_runs):
        start_scope()

        seed(run_id)
        np.random.seed(run_id)
        defaultclock.dt = dt

        eqs = '''
        dr/dt = (-r + tanh(total_input))/tau_i : 1
        total_input : 1
        tau_i : second
        '''

        G = NeuronGroup(N, eqs, method='euler')
        G.r = '0.05 * rand()'
        G.tau_i = tau_baseline
        G.total_input = baseline_input

        # -------------------
        # Define E/I populations
        # -------------------
        num_exc = int(ratio * N)
        num_inh = N - num_exc

        neuron_types = np.ones(N)      # +1 = excitatory
        neuron_types[num_exc:] = -1    # -1 = inhibitory
        np.random.shuffle(neuron_types)

        # -------------------
        # Weight matrix
        # Sign depends on presynaptic neuron (columns)
        # -------------------
        W_abs = np.abs(np.random.normal(0, 1/np.sqrt(N), size=(N, N)))

        W = np.zeros((N, N))
        for j in range(N):
            if neuron_types[j] == 1:
                W[:, j] = w_exc * W_abs[:, j]
            else:
                W[:, j] = -w_inh * W_abs[:, j]

        np.fill_diagonal(W, 0)

        @network_operation(dt=defaultclock.dt)
        def update_input():
            if pulse_start <= defaultclock.t < pulse_end:
                input_signal = pulse_amplitude
            else:
                input_signal = baseline_input

            G.total_input = input_signal + np.dot(W, G.r)

        M = StateMonitor(G, 'r', record=True)

        run(duration)

        mean_activity = np.asarray(np.mean(M.r, axis=0))
        corr = np.asarray(autocorrelation(mean_activity))

        if current_time is None:
            current_time = np.asarray(M.t / ms)
            current_lags = np.arange(len(corr)) * float(defaultclock.dt / ms)
            expected_len = len(mean_activity)

        if len(mean_activity) != expected_len:
            print(f"Skipping run {run_id} for ratio={ratio}: mean_activity length mismatch")
            continue

        if len(corr) < 1000:
            print(f"Skipping run {run_id} for ratio={ratio}: autocorrelation too short")
            continue

        all_mean_activity.append(mean_activity.copy())
        all_corr.append(corr[:1000].copy())

    if len(all_mean_activity) == 0:
        print(f"No valid runs for ratio={ratio}")
        continue

    all_mean_activity = np.vstack(all_mean_activity)
    all_corr = np.vstack(all_corr)

    results_mean_activity[ratio] = np.mean(all_mean_activity, axis=0)
    results_std_activity[ratio] = np.std(all_mean_activity, axis=0)

    results_mean_corr[ratio] = np.mean(all_corr, axis=0)
    results_std_corr[ratio] = np.std(all_corr, axis=0)

    results_time[ratio] = current_time
    results_lags[ratio] = current_lags[:1000]

    results_timescale[ratio] = estimate_timescale(
        results_mean_corr[ratio],
        results_lags[ratio],
        threshold=0.8
    )

# -------------------
# Plot 1: mean activity
# -------------------
plt.figure(figsize=(10, 6))
for ratio in results_mean_activity.keys():
    plt.plot(results_time[ratio], results_mean_activity[ratio],
             label=f'E/I = {int(ratio*100)}/{int((1-ratio)*100)}')
plt.axvspan(pulse_start / ms, pulse_end / ms, alpha=0.2, label='Input pulse')
plt.xlabel('Time (ms)')
plt.ylabel('Mean activity')
plt.title('Effect of excitatory-inhibitory balance: mean population activity')
plt.legend()
plt.tight_layout()

# -------------------
# Plot 2: autocorrelation
# -------------------
plt.figure(figsize=(10, 6))
for ratio in results_mean_corr.keys():
    plt.plot(results_lags[ratio], results_mean_corr[ratio],
             label=f'E/I = {int(ratio*100)}/{int((1-ratio)*100)}')
plt.xlabel('Lag (ms)')
plt.ylabel('Autocorrelation')
plt.title('Effect of excitatory-inhibitory balance: autocorrelation of mean population activity')
plt.legend()
plt.tight_layout()

# -------------------
# Plot 3: estimated timescale
# -------------------
plt.figure(figsize=(8, 5))
ratios = list(results_timescale.keys())
taus = [results_timescale[r] for r in ratios]

plt.plot(ratios, taus, 'o-')
plt.xlabel('Excitatory fraction')
plt.ylabel('Estimated timescale (ms)')
plt.title('Effect of excitatory-inhibitory balance: estimated intrinsic timescale')
plt.tight_layout()

plt.show()

print("Estimated timescales:")
for ratio in ratios:
    print(f"E/I = {int(ratio*100)}/{int((1-ratio)*100)}: {results_timescale[ratio]:.2f} ms")