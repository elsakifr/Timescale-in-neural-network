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
    corr = corr / corr[0]
    return corr

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
ei_ratios = [0.8, 0.7, 0.6, 0.5]
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

noise_sigma = 0.05

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

ei_labels = {}

representative_W = {}
representative_neuron_types = {}

# -------------------
# Loop over E/I ratios
# -------------------
for ratio in ei_ratios:
    all_mean_activity = []
    all_corr = []

    current_time = None
    current_lags = None
    expected_len = None

    num_exc = int(round(ratio * N))
    num_inh = N - num_exc

    ei_labels[ratio] = f"{num_exc}/{num_inh}"

    print(f"\nRunning E/I = {num_exc}/{num_inh}")

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
        # Balanced connectivity matrix W
        # Case 3-inspired:
        # full connectivity, log-normal magnitudes,
        # inhibitory columns rescaled so sum(W) ≈ 0
        # -------------------
        W = np.random.lognormal(mean=mu, sigma=sigma, size=(N, N))

        # remove self-connections BEFORE balancing
        np.fill_diagonal(W, 0)

        # excitatory columns: positive
        W[:, exc_cols] = w_exc * W[:, exc_cols]

        # compute total excitation and inhibition magnitudes
        exc_sum = np.sum(W[:, exc_cols])
        inh_sum = np.sum(W[:, inh_cols])

        # scale inhibition so total recurrent weight is balanced
        ei_scale = exc_sum / inh_sum

        # inhibitory columns: negative and scaled
        W[:, inh_cols] = -W[:, inh_cols] * ei_scale

        if run_id == 0:
            representative_W[ratio] = W.copy()
            representative_neuron_types[ratio] = neuron_types.copy()

            print(f"Total W sum: {np.sum(W):.10f}")
            print(f"Mean row sum: {np.mean(np.sum(W, axis=1)):.10f}")
            print(f"Mean col sum: {np.mean(np.sum(W, axis=0)):.10f}")

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

        if current_time is None:
            current_time = np.asarray(M.t / ms)
            current_lags = np.arange(len(corr)) * float(defaultclock.dt / ms)
            expected_len = len(mean_activity)

        if len(mean_activity) != expected_len:
            print(f"Skipping run {run_id} for ratio={ratio}: length mismatch")
            continue

        if len(corr) < 1000:
            print(f"Skipping run {run_id} for ratio={ratio}: autocorrelation too short")
            continue

        all_mean_activity.append(mean_activity.copy())
        all_corr.append(corr[:1000].copy())

    all_mean_activity = np.vstack(all_mean_activity)
    all_corr = np.vstack(all_corr)

    results_mean_activity[ratio] = np.mean(all_mean_activity, axis=0)
    results_std_activity[ratio] = np.std(all_mean_activity, axis=0)

    results_mean_corr[ratio] = np.mean(all_corr, axis=0)
    results_std_corr[ratio] = np.std(all_corr, axis=0)

    results_time[ratio] = current_time
    results_lags[ratio] = current_lags[:1000]

    results_timescale[ratio] = estimate_timescale_from_decay(
        results_mean_activity[ratio],
        results_time[ratio],
        pulse_end / ms
    )

# -------------------
# Plot 1: mean population activity
# -------------------
plt.figure(figsize=(10, 6))

for ratio in ei_ratios:
    mean_curve = results_mean_activity[ratio]
    std_curve = results_std_activity[ratio]

    plt.plot(
        results_time[ratio],
        mean_curve,
        label=f'E/I = {ei_labels[ratio]}'
    )

    '''
    plt.fill_between(
        results_time[ratio],
        mean_curve - std_curve,
        mean_curve + std_curve,
        alpha=0.15
    )'''

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

for ratio in ei_ratios:
    mean_corr = results_mean_corr[ratio]
    std_corr = results_std_corr[ratio]

    plt.plot(
        results_lags[ratio],
        mean_corr,
        label=f'E/I = {ei_labels[ratio]}'
    )
    '''
    plt.fill_between(
        results_lags[ratio],
        mean_corr - std_corr,
        mean_corr + std_corr,
        alpha=0.15
    )'''

plt.xlabel('Lag (ms)')
plt.ylabel('Autocorrelation')
plt.title('Effect of excitatory-inhibitory balance: autocorrelation')
plt.legend()
plt.tight_layout()

# -------------------
# Plot 3: estimated timescale
# -------------------
plt.figure(figsize=(8, 5))

valid_x = []
valid_y = []
first_nan = True

for r in ei_ratios:
    tau = results_timescale[r]

    if np.isnan(tau):
        if first_nan:
            plt.scatter(r, 0, marker='x', s=100, label='Undefined timescale')
            first_nan = False
        else:
            plt.scatter(r, 0, marker='x', s=100)

        plt.text(r, 5, 'undefined', ha='center')
    else:
        valid_x.append(r)
        valid_y.append(tau)

valid_pairs = sorted(zip(valid_x, valid_y))
x_sorted = [p[0] for p in valid_pairs]
y_sorted = [p[1] for p in valid_pairs]

plt.plot(x_sorted, y_sorted, 'o-', label='Estimated timescale')
plt.xlabel('Excitatory fraction')
plt.ylabel('Estimated timescale (ms)')
plt.title('Effect of excitatory-inhibitory balance: estimated intrinsic timescale')
plt.legend()
plt.tight_layout()

# -------------------
# Plot 4: connectivity matrices
# -------------------
for ratio in ei_ratios:
    W_plot = representative_W[ratio]

    plt.figure(figsize=(6, 6))
    plt.imshow(W_plot, cmap='bwr', aspect='auto')
    plt.colorbar(label='Connection strength')
    plt.xlabel('Presynaptic neuron j')
    plt.ylabel('Postsynaptic neuron i')
    plt.title(f'Connectivity matrix W (E/I = {ei_labels[ratio]})')
    plt.tight_layout()

# -------------------
# Plot 5: sorted connectivity matrices
# -------------------
# -------------------
# Plot 5: all sorted connectivity matrices in one figure
# Arvind-style subplot layout
# -------------------
plt.figure(figsize=(12, 10))
plt.suptitle("Sorted connectivity matrices for different E/I balances", fontsize=16)

for ii, ratio in enumerate(ei_ratios):
    W_plot = representative_W[ratio]
    neuron_types = representative_neuron_types[ratio]

    # Sort so inhibitory neurons come first, excitatory neurons second
    sorted_idx = np.argsort(neuron_types)
    W_sorted = W_plot[sorted_idx][:, sorted_idx]

    plt.subplot(2, 2, ii + 1)
    plt.pcolor(W_sorted, cmap="viridis")
    plt.colorbar(label="Connection strength")

    plt.title(f"E/I = {ei_labels[ratio]}")
    plt.xlabel("Presynaptic neuron j")
    plt.ylabel("Postsynaptic neuron i")

plt.tight_layout()

# -------------------
# Print results
# -------------------
print("\nEstimated timescales:")
for r in ei_ratios:

    activity = results_mean_activity[r]
    time = results_time[r]

    # hitta peak efter input
    mask = time >= pulse_end/ms
    peak = np.max(activity[mask])
    baseline = np.mean(activity[-100:])

    value = results_timescale[r]

    if np.isnan(value):
        print(f"E/I = {ei_labels[r]}: peak={peak:.3f} → baseline={baseline:.3f}, no clear decay")
    else:
        print(f"E/I = {ei_labels[r]}: peak={peak:.3f} → baseline={baseline:.3f}, τ={value:.2f} ms")


plt.show()