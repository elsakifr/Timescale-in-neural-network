from brian2 import *
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

# -------------------
# Helper function
# -------------------
def autocorrelation(x):
    x = np.asarray(x)
    x = x - np.mean(x)
    corr = np.correlate(x, x, mode='full')
    corr = corr[corr.size // 2:]
    corr = corr / corr[0]
    return corr

# -------------------
# Baseline settings
# -------------------
num_runs = 20
N = 100
duration = 3000 * ms
dt = 0.1 * ms

w_rec = 0.6
baseline_input = 0.0
pulse_amplitude = 0.8
pulse_start = 500 * ms
pulse_end = 1000 * ms
tau_baseline = 20 * ms

# -------------------
# Storage
# -------------------
all_corr = []
all_mean_activity = []

first_run_t = None
first_run_r = None
lags = None

# -------------------
# Run multiple baseline simulations
# -------------------
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
    mean_activity = np.mean(M.r, axis=0)

    # compute autocorrelation
    corr = autocorrelation(mean_activity)

    # save representative run (first run) for single-neuron plot
    if run_id == 0:
        first_run_t = M.t / ms
        first_run_r = np.array(M.r)
        lags = np.arange(len(corr)) * float(defaultclock.dt / ms)

    # store results from each run
    all_mean_activity.append(mean_activity)
    all_corr.append(corr[:1000])

# -------------------
# Convert to arrays
# -------------------
all_mean_activity = np.array(all_mean_activity)
all_corr = np.array(all_corr)

# Mean and std across runs
mean_of_mean_activity = np.mean(all_mean_activity, axis=0)
std_of_mean_activity = np.std(all_mean_activity, axis=0)

mean_corr = np.mean(all_corr, axis=0)
std_corr = np.std(all_corr, axis=0)


def exp_decay(t, tau):
    return np.exp(-t / tau)

fit_max_lag = 100

corr_curve = mean_corr
lag_curve = lags[:len(corr_curve)]

mask = (lag_curve > 0) & (lag_curve <= fit_max_lag)

x_fit = lag_curve[mask]
y_fit = corr_curve[mask]

popt, _ = curve_fit(
    exp_decay,
    x_fit,
    y_fit,
    p0=(20.0,),
    bounds=([1], [1000])
)

tau_eff_baseline = popt[0]

print(
    f"Baseline intrinsic timescale = {tau_eff_baseline:.2f} ms"
)

# -------------------
# Plot 1: representative single-neuron activity
# -------------------
plt.figure(figsize=(10, 6))
for i in range(5):
    plt.plot(first_run_t, first_run_r[i], label=f'Neuron {i}')
plt.axvspan(pulse_start / ms, pulse_end / ms, alpha=0.2, label='Input pulse')
plt.xlabel('Time (ms)')
plt.ylabel('Rate activity')
plt.title('Baseline model: representative single-neuron activity')
plt.legend()
plt.tight_layout()

# -------------------
# Plot 2: mean network activity across runs
# -------------------
plt.figure(figsize=(10, 6))
plt.plot(first_run_t, mean_of_mean_activity, label='Mean across runs')
plt.fill_between(
    first_run_t,
    mean_of_mean_activity - std_of_mean_activity,
    mean_of_mean_activity + std_of_mean_activity,
    alpha=0.3,
    label='±1 std'
)
plt.axvspan(pulse_start / ms, pulse_end / ms, alpha=0.2, label='Input pulse')
plt.xlabel('Time (ms)')
plt.ylabel('Mean activity')
plt.title('Baseline model: mean network activity across runs')
plt.legend()
plt.tight_layout()

# -------------------
# Plot 3: autocorrelation across runs
# -------------------
plt.figure(figsize=(10, 6))
plt.plot(lags[:1000], mean_corr, label='Mean autocorrelation')
plt.fill_between(
    lags[:1000],
    mean_corr - std_corr,
    mean_corr + std_corr,
    alpha=0.3,
    label='±1 std'
)
plt.xlabel('Lag (ms)')
plt.ylabel('Autocorrelation')
plt.title('Baseline model: autocorrelation across runs')
plt.legend()
plt.tight_layout()

plt.show()