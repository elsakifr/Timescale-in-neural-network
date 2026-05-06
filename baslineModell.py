from brian2 import *

from scipy.optimize import curve_fit

import matplotlib.pyplot as plt

import numpy as np

prefs.codegen.target = "numpy"

# -------------------

# Helper functions

# -------------------

def autocorrelation(x):

    x = np.asarray(x)

    x = x - np.mean(x)

    corr = np.correlate(x, x, mode='full')

    corr = corr[corr.size // 2:]

    if corr[0] == 0:

        return np.full_like(corr, np.nan)

    return corr / corr[0]

def exp_decay(t, A, tau):

    return A * np.exp(-t / tau)

def estimate_timescale_from_decay(mean_activity, time_ms, pulse_end_ms):

    post_mask = time_ms >= pulse_end_ms

    t_post = time_ms[post_mask] - pulse_end_ms

    y_post = mean_activity[post_mask]

    if len(y_post) < 20:

        return np.nan

    tail_len = max(10, len(y_post) // 5)

    baseline_est = np.mean(y_post[-tail_len:])

    y_decay = y_post - baseline_est

    fit_mask = y_decay > 0.02

    t_fit = t_post[fit_mask]

    y_fit = y_decay[fit_mask]

    if len(y_fit) < 20:

        return np.nan

    if y_fit[0] <= y_fit[-1] * 1.2:

        return np.nan

    try:

        popt, _ = curve_fit(

            exp_decay,

            t_fit,

            y_fit,

            p0=(y_fit[0], 100.0),

            bounds=([0, 1e-6], [np.inf, np.inf]),

            maxfev=10000

        )

        return popt[1]

    except RuntimeError:

        return np.nan

# -------------------

# Simulation settings

# Same as E/I test

# -------------------

num_runs = 35

N = 100

duration = 4000 * ms

dt = 0.1 * ms

baseline_input = 0.0

pulse_amplitude = 0.8

pulse_start = 500 * ms

pulse_end = 1000 * ms

tau_baseline = 20 * ms

# Case 3 log-normal parameters

w_exc = 0.6

mu = -np.log(np.sqrt(N))

sigma = 0.8

noise_sigma = 0.05

# Baseline E/I ratio

ratio = 0.5

num_exc = int(round(ratio * N))

num_inh = N - num_exc

# -------------------

# Storage

# -------------------

all_mean_activity = []

all_corr = []

first_run_t = None

first_run_r = None

lags = None

representative_W = None

representative_neuron_types = None

# -------------------

# Run baseline simulations

# -------------------

for run_id in range(num_runs):

    start_scope()

    current_seed = 1000 * run_id + int(ratio * 1000)

    seed(current_seed)

    np.random.seed(current_seed)

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

    # -------------------

    # Define E/I neuron types

    # +1 = excitatory

    # -1 = inhibitory

    # -------------------

    neuron_types = np.ones(N)

    neuron_types[num_exc:] = -1

    np.random.shuffle(neuron_types)

    exc_cols = np.where(neuron_types == 1)[0]

    inh_cols = np.where(neuron_types == -1)[0]

    # -------------------

    # Case 3 connectivity:

    # full connectivity, log-normal magnitudes,

    # inhibitory columns rescaled so sum(W) ≈ 0

    # -------------------

    W = np.random.lognormal(mean=mu, sigma=sigma, size=(N, N))

    # Remove self-connections before balancing

    np.fill_diagonal(W, 0)

    # Excitatory columns: positive

    W[:, exc_cols] = w_exc * W[:, exc_cols]

    # Balance inhibition using actual sums

    exc_sum = np.sum(W[:, exc_cols])

    inh_sum = np.sum(W[:, inh_cols])

    ei_scale = exc_sum / inh_sum

    # Inhibitory columns: negative and scaled

    W[:, inh_cols] = -W[:, inh_cols] * ei_scale

    if run_id == 0:

        representative_W = W.copy()

        representative_neuron_types = neuron_types.copy()

        row_sums = np.sum(W, axis=1)

        col_sums = np.sum(W, axis=0)

        print(f"Baseline E/I = {num_exc}/{num_inh}")

        print(f"ei_scale = {ei_scale:.6f}")

        print(f"Total W sum: {np.sum(W):.10f}")

        print(f"Mean row sum: {np.mean(row_sums):.10f}")

        print(f"Std row sum: {np.std(row_sums):.10f}")

        print(f"Mean col sum: {np.mean(col_sums):.10f}")

        print(f"Std col sum: {np.std(col_sums):.10f}")

        print(f"Min excitatory weight: {np.min(W[:, exc_cols]):.6f}")

        print(f"Max inhibitory weight: {np.max(W[:, inh_cols]):.6f}")

    # -------------------

    # External input + recurrent input + noise

    # -------------------

    @network_operation(dt=defaultclock.dt)

    def update_input():

        if pulse_start <= defaultclock.t < pulse_end:

            input_signal = pulse_amplitude

        else:

            input_signal = baseline_input

        noise = noise_sigma * np.random.randn(N)

        G.total_input = input_signal + np.dot(W, G.r) + noise

    # -------------------

    # Monitor and run

    # -------------------

    M = StateMonitor(G, 'r', record=True)

    run(duration)

    mean_activity = np.asarray(np.mean(M.r, axis=0))

    corr = np.asarray(autocorrelation(mean_activity))

    if run_id == 0:

        first_run_t = np.asarray(M.t / ms)

        first_run_r = np.asarray(M.r)

        lags = np.arange(len(corr)) * float(defaultclock.dt / ms)

    all_mean_activity.append(mean_activity.copy())

    all_corr.append(corr[:1000].copy())

# -------------------

# Convert to arrays

# -------------------

all_mean_activity = np.vstack(all_mean_activity)

all_corr = np.vstack(all_corr)

mean_of_mean_activity = np.mean(all_mean_activity, axis=0)

std_of_mean_activity = np.std(all_mean_activity, axis=0)

mean_corr = np.mean(all_corr, axis=0)

std_corr = np.std(all_corr, axis=0)

lags = lags[:1000]

# -------------------

# Estimate baseline timescale

# Same method as E/I test

# -------------------

tau_eff_baseline = estimate_timescale_from_decay(

    mean_of_mean_activity,

    first_run_t,

    pulse_end / ms

)

# -------------------

# Print result

# -------------------

post_mask = first_run_t >= pulse_end / ms

peak = np.max(mean_of_mean_activity[post_mask])

baseline = np.mean(mean_of_mean_activity[-100:])

print("\nEstimated baseline timescale:")

if np.isnan(tau_eff_baseline):

    print(f"Baseline E/I = {num_exc}/{num_inh}: peak={peak:.3f} → baseline={baseline:.3f}, no clear decay")

else:

    print(f"Baseline E/I = {num_exc}/{num_inh}: peak={peak:.3f} → baseline={baseline:.3f}, τ={tau_eff_baseline:.2f} ms")

# -------------------

# Plot 1: representative single-neuron activity

# -------------------

plt.figure(figsize=(10, 6))

for i in range(5):

    plt.plot(first_run_t, first_run_r[i], label=f'Neuron {i}')

plt.axvspan(pulse_start / ms, pulse_end / ms, alpha=0.2, label='Input pulse')

plt.xlabel('Time (ms)')

plt.ylabel('Rate activity')

plt.title('Baseline Case 3: representative single-neuron activity')

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

plt.title('Baseline Case 3: mean network activity across runs')

plt.legend()

plt.tight_layout()

# -------------------

# Plot 3: autocorrelation across runs

# -------------------

plt.figure(figsize=(10, 6))

plt.plot(lags, mean_corr, label='Mean autocorrelation')

plt.fill_between(

    lags,

    mean_corr - std_corr,

    mean_corr + std_corr,

    alpha=0.3,

    label='±1 std'

)

plt.xlabel('Lag (ms)')

plt.ylabel('Autocorrelation')

plt.title('Baseline Case 3: autocorrelation across runs')

plt.legend()

plt.tight_layout()

# -------------------

# Plot 4: connectivity matrix

# -------------------

plt.figure(figsize=(7, 6))

absmax = np.max(np.abs(representative_W))

plt.imshow(

    representative_W,

    cmap='bwr',

    aspect='auto',

    vmin=-absmax,

    vmax=absmax

)

plt.colorbar(label='Connection strength')

plt.xlabel('Presynaptic neuron j')

plt.ylabel('Postsynaptic neuron i')

plt.title('Baseline Case 3: log-normal balanced connectivity matrix')

plt.tight_layout()

# -------------------

# Plot 5: sorted connectivity matrix

# -------------------

sorted_idx = np.argsort(representative_neuron_types)

W_sorted = representative_W[sorted_idx][:, sorted_idx]

plt.figure(figsize=(7, 6))

plt.pcolor(W_sorted, cmap="viridis")

plt.colorbar(label="Connection strength")

plt.xlabel("Presynaptic neuron j")

plt.ylabel("Postsynaptic neuron i")

plt.title("Baseline Case 3: sorted connectivity matrix")

plt.tight_layout()

plt.show()