from brian2 import *
from scipy.optimize import curve_fit
import matplotlib.pyplot as plt
import numpy as np

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
            maxfev=10000,
        )
        return popt[1]

    except RuntimeError:
        return np.nan


# -------------------
# Simulation settings
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

w_exc = 0.6
mu = -np.log(np.sqrt(N))
sigma = 0.8

# noise level
#noise_sigma = 0.05
noise_sigma = 0.1
# noise_sigma = 0.25

ratio = 0.5
num_exc = int(round(ratio * N))
num_inh = N - num_exc

# -------------------
# Storage
# -------------------

all_mean_activity = []

all_corr_full = []
all_corr_post = []

first_run_t = None
first_run_r = None

representative_W = None
representative_neuron_types = None


# -------------------
# Run simulations
# -------------------

for run_id in range(num_runs):
    start_scope()

    current_seed = 1000 * run_id + int(ratio * 1000)
    seed(current_seed)
    np.random.seed(current_seed)

    defaultclock.dt = dt

    eqs = '''
    dr/dt = (-r + tanh(total_input))/tau_i : 1
    total_input : 1
    tau_i : second
    '''

    G = NeuronGroup(N, eqs, method="euler")
    G.r = "0.05 * rand()"
    G.tau_i = tau_baseline
    G.total_input = baseline_input

    neuron_types = np.ones(N)
    neuron_types[num_exc:] = -1
    np.random.shuffle(neuron_types)

    exc_cols = np.where(neuron_types == 1)[0]
    inh_cols = np.where(neuron_types == -1)[0]

    W = np.random.lognormal(mean=mu, sigma=sigma, size=(N, N))

    # Remove self-connections
    np.fill_diagonal(W, 0)

    # Excitatory columns positive
    W[:, exc_cols] = w_exc * W[:, exc_cols]

    # Inhibitory columns negative and balanced
    exc_sum = np.sum(W[:, exc_cols])
    inh_sum = np.sum(W[:, inh_cols])
    ei_scale = exc_sum / inh_sum

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

    @network_operation(dt=defaultclock.dt)
    def update_input():
        if pulse_start <= defaultclock.t < pulse_end:
            input_signal = pulse_amplitude
        else:
            input_signal = baseline_input

        noise = noise_sigma * np.random.randn(N)
        G.total_input = input_signal + np.dot(W, G.r) + noise

    M = StateMonitor(G, "r", record=True)

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
        first_run_t = time_ms
        first_run_r = np.asarray(M.r)

    all_mean_activity.append(mean_activity.copy())
    all_corr_full.append(corr_full.copy())
    all_corr_post.append(corr_post.copy())


# -------------------
# Convert to arrays
# -------------------

all_mean_activity = np.vstack(all_mean_activity)

# Make all autocorrelation arrays same length
min_len_full = min(len(c) for c in all_corr_full)
min_len_post = min(len(c) for c in all_corr_post)

all_corr_full = np.vstack([c[:min_len_full] for c in all_corr_full])
all_corr_post = np.vstack([c[:min_len_post] for c in all_corr_post])

mean_of_mean_activity = np.mean(all_mean_activity, axis=0)
std_of_mean_activity = np.std(all_mean_activity, axis=0)

mean_corr_full = np.mean(all_corr_full, axis=0)
std_corr_full = np.std(all_corr_full, axis=0)

mean_corr_post = np.mean(all_corr_post, axis=0)
std_corr_post = np.std(all_corr_post, axis=0)

lags_full = np.arange(len(mean_corr_full)) * float(dt / ms)
lags_post = np.arange(len(mean_corr_post)) * float(dt / ms)

# -------------------
# Estimate post-pulse timescale
# -------------------

tau_eff_baseline = estimate_timescale_from_decay(
    mean_of_mean_activity,
    first_run_t,
    pulse_end / ms,
)

post_mask = first_run_t >= pulse_end / ms
peak = np.max(mean_of_mean_activity[post_mask])
baseline = np.mean(mean_of_mean_activity[-100:])

print("\nEstimated baseline timescale:")
if np.isnan(tau_eff_baseline):
    print(f"Baseline E/I = {num_exc}/{num_inh}: peak={peak:.3f} → baseline={baseline:.3f}, no clear decay")
else:
    print(f"Baseline E/I = {num_exc}/{num_inh}: peak={peak:.3f} → baseline={baseline:.3f}, τ={tau_eff_baseline:.2f} ms")


# -------------------
# Autocorrelation diagnostics
# -------------------

zero_full = find_zero_crossing(mean_corr_full, lags_full)
zero_post = find_zero_crossing(mean_corr_post, lags_post)

print("\nAutocorrelation diagnostics:")

if np.isnan(zero_full):
    print("Full-signal autocorrelation does not cross zero.")
else:
    print(f"Full-signal autocorrelation crosses zero at {zero_full:.2f} ms.")

if np.isnan(zero_post):
    print("Post-stimulus autocorrelation does not cross zero.")
else:
    print(f"Post-stimulus autocorrelation crosses zero at {zero_post:.2f} ms.")


# -------------------
# Plot 1: representative single-neuron activity
# -------------------

plt.figure(figsize=(10, 6))

for i in range(5):
    plt.plot(first_run_t, first_run_r[i], label=f"Neuron {i}")

plt.axvspan(pulse_start / ms, pulse_end / ms, alpha=0.2, label="Input pulse")
plt.xlabel("Time (ms)")
plt.ylabel("Rate activity")
plt.title("Baseline: representative single-neuron activity")
plt.legend()
plt.tight_layout()


# -------------------
# Plot 2: mean population activity
# -------------------

plt.figure(figsize=(10, 6))

plt.plot(first_run_t, mean_of_mean_activity, label="Mean across runs")

plt.fill_between(
    first_run_t,
    mean_of_mean_activity - std_of_mean_activity,
    mean_of_mean_activity + std_of_mean_activity,
    alpha=0.3,
    label="±1 std",
)

plt.axvspan(pulse_start / ms, pulse_end / ms, alpha=0.2, label="Input pulse")
plt.xlabel("Time (ms)")
plt.ylabel("Mean activity")
plt.title("Baseline: mean network activity across runs")
plt.legend()
plt.tight_layout()


# -------------------
# Plot 3: full-signal autocorrelation
# -------------------

plt.figure(figsize=(10, 6))

plt.plot(lags_full, mean_corr_full, label="Full-signal autocorrelation")

plt.fill_between(
    lags_full,
    mean_corr_full - std_corr_full,
    mean_corr_full + std_corr_full,
    alpha=0.3,
    label="±1 std",
)

plt.axhline(0, color="black", linestyle="--", linewidth=1, label="Zero line")

if not np.isnan(zero_full):
    plt.axvline(
        zero_full,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Zero crossing ≈ {zero_full:.1f} ms",
    )
    plt.scatter(zero_full, 0, color="red", zorder=5)

plt.xlabel("Lag (ms)")
plt.ylabel("Autocorrelation")
plt.title("Baseline: full-signal autocorrelation")
plt.legend()
plt.tight_layout()


# -------------------
# Plot 4: post-stimulus autocorrelation
# -------------------

plt.figure(figsize=(10, 6))

plt.plot(lags_post, mean_corr_post, label="Post-stimulus autocorrelation")

plt.fill_between(
    lags_post,
    mean_corr_post - std_corr_post,
    mean_corr_post + std_corr_post,
    alpha=0.3,
    label="±1 std",
)

plt.axhline(0, color="black", linestyle="--", linewidth=1, label="Zero line")

if not np.isnan(zero_post):
    plt.axvline(
        zero_post,
        color="red",
        linestyle="--",
        linewidth=1.5,
        label=f"Zero crossing ≈ {zero_post:.1f} ms",
    )
    plt.scatter(zero_post, 0, color="red", zorder=5)

plt.xlim(0, 1000)
plt.xlabel("Lag (ms)")
plt.ylabel("Autocorrelation")
plt.title("Baseline: post-stimulus autocorrelation")
plt.legend()
plt.tight_layout()


# -------------------
# Plot 5: zoomed post-stimulus autocorrelation
# -------------------

plt.figure(figsize=(10, 6))

plt.plot(lags_post, mean_corr_post, label="Post-stimulus autocorrelation")

plt.fill_between(
    lags_post,
    mean_corr_post - std_corr_post,
    mean_corr_post + std_corr_post,
    alpha=0.3,
    label="±1 std",
)

plt.axhline(0, color="black", linestyle="--", linewidth=1)
plt.xlim(0, 150)
plt.xlabel("Lag (ms)")
plt.ylabel("Autocorrelation")
plt.title("Baseline: post-stimulus autocorrelation zoom")
plt.legend()
plt.tight_layout()


# -------------------
# Plot 6: connectivity matrix
# -------------------

plt.figure(figsize=(7, 6))

absmax = np.max(np.abs(representative_W))

plt.imshow(
    representative_W,
    cmap="bwr",
    aspect="auto",
    vmin=-absmax,
    vmax=absmax,
)

plt.colorbar(label="Connection strength")
plt.xlabel("Presynaptic neuron j")
plt.ylabel("Postsynaptic neuron i")
plt.title("Baseline: log-normal balanced connectivity matrix")
plt.tight_layout()


# -------------------
# Plot 7: sorted connectivity matrix
# -------------------

sorted_idx = np.argsort(representative_neuron_types)
W_sorted = representative_W[sorted_idx][:, sorted_idx]

plt.figure(figsize=(7, 6))

plt.pcolor(W_sorted, cmap="viridis")
plt.colorbar(label="Connection strength")
plt.xlabel("Presynaptic neuron j")
plt.ylabel("Postsynaptic neuron i")
plt.title("Baseline: sorted connectivity matrix")
plt.tight_layout()

plt.show()