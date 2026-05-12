from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

prefs.codegen.target = "numpy"

# -------------------
# Helper functions
# -------------------
def autocorrelation(x, max_lag=None, unbiased=False):
    x = np.asarray(x)
    x = x - np.mean(x)

    corr = np.correlate(x, x, mode="full")
    corr = corr[corr.size // 2:]

    if unbiased:
        n = len(x)
        overlap = np.arange(n, 0, -1)
        corr = corr / overlap

    if corr[0] == 0:
        return np.full_like(corr, np.nan)

    corr = corr / corr[0]

    if max_lag is not None:
        corr = corr[:max_lag]

    return corr


def find_zero_crossing(corr, lags):
    crossing_idx = np.where(corr <= 0)[0]

    if len(crossing_idx) == 0:
        return np.nan

    idx = crossing_idx[0]

    if idx == 0:
        return lags[0]

    x1, x2 = lags[idx - 1], lags[idx]
    y1, y2 = corr[idx - 1], corr[idx]

    return x1 - y1 * (x2 - x1) / (y2 - y1)


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

w_exc = 0.6
sigma = 0.8
noise_sigma = 0.1

ratio = 0.5

# -------------------
# Storage
# -------------------
results_mean_activity = {}
results_std_activity = {}

results_mean_corr_full = {}
results_std_corr_full = {}
results_lags_full = {}

results_mean_corr_post = {}
results_std_corr_post = {}
results_lags_post = {}

results_zero_full = {}
results_zero_post = {}

results_timescale = {}
results_time = {}

representative_W = {}
representative_neuron_types = {}

# -------------------
# Loop over network sizes
# -------------------
for N in network_sizes:
    print(f"\nRunning network size N = {N}")

    all_mean_activity = []
    all_corr_full = []
    all_corr_post = []

    num_exc = int(round(ratio * N))
    num_inh = N - num_exc
    mu = -np.log(np.sqrt(N))

    current_time = None

    for run_id in range(num_runs):
        start_scope()

        current_seed = 1000 * run_id + N
        seed(current_seed)
        np.random.seed(current_seed)

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

        neuron_types = np.ones(N)
        neuron_types[num_exc:] = -1
        np.random.shuffle(neuron_types)

        exc_cols = np.where(neuron_types == 1)[0]
        inh_cols = np.where(neuron_types == -1)[0]

        W = np.random.lognormal(mean=mu, sigma=sigma, size=(N, N))
        np.fill_diagonal(W, 0)

        W[:, exc_cols] = w_exc * W[:, exc_cols]

        exc_sum = np.sum(W[:, exc_cols])
        inh_sum = np.sum(W[:, inh_cols])
        ei_scale = exc_sum / inh_sum

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

        @network_operation(dt=defaultclock.dt)
        def update_input():
            if pulse_start <= defaultclock.t < pulse_end:
                input_signal = pulse_amplitude
            else:
                input_signal = baseline_input

            noise = noise_sigma * np.random.randn(N)
            G.total_input = input_signal + np.dot(W, G.r) + noise

        M = StateMonitor(G, 'r', record=True)
        run(duration)

        time_ms = np.asarray(M.t / ms)
        mean_activity = np.asarray(np.mean(M.r, axis=0))

        # Full-signal autocorrelation
        corr_full = autocorrelation(mean_activity, unbiased=False)

        # Post-stimulus autocorrelation only
        post_mask = time_ms >= pulse_end / ms
        post_activity = mean_activity[post_mask]
        corr_post = autocorrelation(post_activity, unbiased=False)

        if run_id == 0:
            current_time = time_ms

        all_mean_activity.append(mean_activity.copy())
        all_corr_full.append(corr_full.copy())
        all_corr_post.append(corr_post.copy())

    # -------------------
    # Average across runs
    # -------------------
    all_mean_activity = np.vstack(all_mean_activity)

    min_len_full = min(len(c) for c in all_corr_full)
    min_len_post = min(len(c) for c in all_corr_post)

    all_corr_full = np.vstack([c[:min_len_full] for c in all_corr_full])
    all_corr_post = np.vstack([c[:min_len_post] for c in all_corr_post])

    results_mean_activity[N] = np.mean(all_mean_activity, axis=0)
    results_std_activity[N] = np.std(all_mean_activity, axis=0)

    results_mean_corr_full[N] = np.mean(all_corr_full, axis=0)
    results_std_corr_full[N] = np.std(all_corr_full, axis=0)
    results_lags_full[N] = np.arange(len(results_mean_corr_full[N])) * float(dt / ms)

    results_mean_corr_post[N] = np.mean(all_corr_post, axis=0)
    results_std_corr_post[N] = np.std(all_corr_post, axis=0)
    results_lags_post[N] = np.arange(len(results_mean_corr_post[N])) * float(dt / ms)

    results_zero_full[N] = find_zero_crossing(
        results_mean_corr_full[N],
        results_lags_full[N]
    )

    results_zero_post[N] = find_zero_crossing(
        results_mean_corr_post[N],
        results_lags_post[N]
    )

    results_time[N] = current_time

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
    plt.plot(
        results_time[N],
        results_mean_activity[N],
        label=f'N = {N}'
    )

    plt.fill_between(
        results_time[N],
        results_mean_activity[N] - results_std_activity[N],
        results_mean_activity[N] + results_std_activity[N],
        alpha=0.15
    )

plt.axvspan(pulse_start / ms, pulse_end / ms, alpha=0.2, label='Input pulse')
plt.xlabel('Time (ms)')
plt.ylabel('Mean activity')
plt.title('Effect of network size: mean population activity')
plt.legend()
plt.tight_layout()

# -------------------
# Plot 2: post-stimulus autocorrelation
# -------------------
plt.figure(figsize=(10, 6))

for N in network_sizes:
    plt.plot(
        results_lags_post[N],
        results_mean_corr_post[N],
        label=f'N = {N}'
    )

    plt.fill_between(
        results_lags_post[N],
        results_mean_corr_post[N] - results_std_corr_post[N],
        results_mean_corr_post[N] + results_std_corr_post[N],
        alpha=0.15
    )

plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlim(0, 1000)
plt.xlabel('Lag after stimulus offset (ms)')
plt.ylabel('Autocorrelation')
plt.title('Effect of network size: post-stimulus autocorrelation')
plt.legend()
plt.tight_layout()

# -------------------
# Plot 3: zoomed post-stimulus autocorrelation
# -------------------
plt.figure(figsize=(10, 6))

for N in network_sizes:
    plt.plot(
        results_lags_post[N],
        results_mean_corr_post[N],
        label=f'N = {N}'
    )

plt.axhline(0, color='black', linestyle='--', linewidth=1)
plt.xlim(0, 150)
plt.xlabel('Lag after stimulus offset (ms)')
plt.ylabel('Autocorrelation')
plt.title('Effect of network size: post-stimulus autocorrelation zoom')
plt.legend()
plt.tight_layout()

# -------------------
# Plot 4: estimated post-pulse timescale
# -------------------
plt.figure(figsize=(8, 5))

valid_N = []
valid_tau = []

for N in network_sizes:
    tau = results_timescale[N]

    if np.isnan(tau):
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

print("\nAutocorrelation zero-crossing diagnostics:")
for N in network_sizes:
    z_full = results_zero_full[N]
    z_post = results_zero_post[N]

    if np.isnan(z_full):
        full_text = "no zero crossing"
    else:
        full_text = f"{z_full:.2f} ms"

    if np.isnan(z_post):
        post_text = "no zero crossing"
    else:
        post_text = f"{z_post:.2f} ms"

    print(f"N = {N}: full-signal zero crossing = {full_text}, post-stimulus zero crossing = {post_text}")

plt.show()