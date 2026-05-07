from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

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
# Same as baseline / E/I Case 3
# -------------------
num_runs = 35
network_sizes = [50, 100, 200, 500]

duration = 4000 * ms
dt = 0.1 * ms

baseline_input = 0.0
pulse_amplitude = 0.8
pulse_start = 500 * ms
pulse_end = 1000 * ms
tau_baseline = 20 * ms

# Case 3 settings
w_exc = 0.6
sigma = 0.8
noise_sigma = 0.05

# Baseline E/I ratio: 50/50
ratio = 0.5

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

representative_W = {}
representative_neuron_types = {}

# -------------------
# Loop over network sizes
# -------------------
for N in network_sizes:
    print(f"\nRunning network size N = {N}")

    all_mean_activity = []
    all_corr = []

    num_exc = int(round(ratio * N))
    num_inh = N - num_exc

    # Important: same log-normal scaling idea as baseline
    mu = -np.log(np.sqrt(N))

    current_time = None
    current_lags = None

    for run_id in range(num_runs):
        start_scope()

        current_seed = 1000 * run_id + N
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
        # -------------------
        neuron_types = np.ones(N)
        neuron_types[num_exc:] = -1
        np.random.shuffle(neuron_types)

        exc_cols = np.where(neuron_types == 1)[0]
        inh_cols = np.where(neuron_types == -1)[0]

        # -------------------
        # Case 3 connectivity:
        # full connectivity, log-normal magnitudes,
        # inhibition rescaled so total W sum ≈ 0
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
            representative_W[N] = W.copy()
            representative_neuron_types[N] = neuron_types.copy()

            row_sums = np.sum(W, axis=1)
            col_sums = np.sum(W, axis=0)

            print(f"E/I = {num_exc}/{num_inh}")
            print(f"mu = {mu:.6f}")
            print(f"ei_scale = {ei_scale:.6f}")
            print(f"Total W sum: {np.sum(W):.10f}")
            print(f"Mean row sum: {np.mean(row_sums):.10f}")
            print(f"Std row sum: {np.std(row_sums):.10f}")
            print(f"Mean col sum: {np.mean(col_sums):.10f}")
            print(f"Std col sum: {np.std(col_sums):.10f}")

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
            current_time = np.asarray(M.t / ms)
            current_lags = np.arange(len(corr)) * float(defaultclock.dt / ms)

        all_mean_activity.append(mean_activity.copy())
        all_corr.append(corr[:1000].copy())

    # -------------------
    # Average across runs
    # -------------------
    all_mean_activity = np.vstack(all_mean_activity)
    all_corr = np.vstack(all_corr)

    results_mean_activity[N] = np.mean(all_mean_activity, axis=0)
    results_std_activity[N] = np.std(all_mean_activity, axis=0)

    results_mean_corr[N] = np.mean(all_corr, axis=0)
    results_std_corr[N] = np.std(all_corr, axis=0)

    results_time[N] = current_time
    results_lags[N] = current_lags[:1000]

    results_timescale[N] = estimate_timescale_from_decay(
        results_mean_activity[N],
        results_time[N],
        pulse_end / ms
    )


# -------------------
# Plot 1: mean population activity
# -------------------
plt.figure(figsize=(10, 6))

for N in network_sizes:
    mean_curve = results_mean_activity[N]
    std_curve = results_std_activity[N]

    plt.plot(
        results_time[N],
        mean_curve,
        label=f'N = {N}'
    )

    plt.fill_between(
        results_time[N],
        mean_curve - std_curve,
        mean_curve + std_curve,
        alpha=0.15
    )

plt.axvspan(pulse_start / ms, pulse_end / ms, alpha=0.2, label='Input pulse')
plt.xlabel('Time (ms)')
plt.ylabel('Mean activity')
plt.title('Effect of network size: mean population activity')
plt.legend()
plt.tight_layout()


# -------------------
# Plot 2: autocorrelation
# -------------------
plt.figure(figsize=(10, 6))

for N in network_sizes:
    mean_corr = results_mean_corr[N]
    std_corr = results_std_corr[N]

    plt.plot(
        results_lags[N],
        mean_corr,
        label=f'N = {N}'
    )

    plt.fill_between(
        results_lags[N],
        mean_corr - std_corr,
        mean_corr + std_corr,
        alpha=0.15
    )

plt.xlabel('Lag (ms)')
plt.ylabel('Autocorrelation')
plt.title('Effect of network size: autocorrelation')
plt.legend()
plt.tight_layout()


# -------------------
# Plot 3: estimated post-pulse timescale
# -------------------
plt.figure(figsize=(8, 5))

valid_N = []
valid_tau = []
first_nan = True

for N in network_sizes:
    tau = results_timescale[N]

    if np.isnan(tau):
        if first_nan:
            plt.scatter(N, 0, marker='x', s=100, label='Undefined timescale')
            first_nan = False
        else:
            plt.scatter(N, 0, marker='x', s=100)

        plt.text(N, 5, 'undefined', ha='center')
    else:
        valid_N.append(N)
        valid_tau.append(tau)

if len(valid_N) > 0:
    plt.plot(valid_N, valid_tau, 'o-', label='Estimated timescale')

plt.xlabel('Network size (N)')
plt.ylabel('Estimated timescale (ms)')
plt.title('Effect of network size: estimated post-pulse time constant')
plt.legend()
plt.tight_layout()


# -------------------
# Plot 4: representative connectivity matrices
# -------------------
plt.figure(figsize=(12, 10))
plt.suptitle("Representative Case 3 connectivity matrices for different network sizes", fontsize=16)

# Use common color scale
absmax = max(np.max(np.abs(representative_W[N])) for N in network_sizes)

for ii, N in enumerate(network_sizes):
    W_plot = representative_W[N]

    plt.subplot(2, 2, ii + 1)
    plt.imshow(
        W_plot,
        cmap='bwr',
        aspect='auto',
        vmin=-absmax,
        vmax=absmax
    )
    plt.colorbar(label='Connection strength')
    plt.title(f'N = {N}')
    plt.xlabel('Presynaptic neuron j')
    plt.ylabel('Postsynaptic neuron i')

plt.tight_layout()


# -------------------
# Plot 5: sorted connectivity matrices
# -------------------
plt.figure(figsize=(12, 10))
plt.suptitle("Sorted Case 3 connectivity matrices for different network sizes", fontsize=16)

for ii, N in enumerate(network_sizes):
    W_plot = representative_W[N]
    neuron_types = representative_neuron_types[N]

    sorted_idx = np.argsort(neuron_types)
    W_sorted = W_plot[sorted_idx][:, sorted_idx]

    plt.subplot(2, 2, ii + 1)
    plt.imshow(
        W_sorted,
        cmap='bwr',
        aspect='auto',
        vmin=-absmax,
        vmax=absmax
    )
    plt.colorbar(label='Connection strength')
    plt.title(f'N = {N}')
    plt.xlabel('Presynaptic neuron j')
    plt.ylabel('Postsynaptic neuron i')

plt.tight_layout()


# -------------------
# Print results
# -------------------
print("\nEstimated post-pulse timescales:")
for N in network_sizes:
    activity = results_mean_activity[N]
    time = results_time[N]

    post_mask = time >= pulse_end / ms
    peak = np.max(activity[post_mask])
    baseline = np.mean(activity[-100:])

    tau = results_timescale[N]

    if np.isnan(tau):
        print(f"N = {N}: peak={peak:.3f} → baseline={baseline:.3f}, no clear decay")
    else:
        print(f"N = {N}: peak={peak:.3f} → baseline={baseline:.3f}, τ={tau:.2f} ms")

plt.show()