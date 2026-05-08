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


def estimate_post_pulse_timescale(mean_activity, time_ms, pulse_end_ms):
    """
    Estimate timescale from decay of mean activity after the input pulse is removed.
    Baseline is estimated from the late post-pulse activity.
    Fit is performed only on the early post-pulse decay window.
    """

    mean_activity = np.asarray(mean_activity)
    time_ms = np.asarray(time_ms)

    # use full post-pulse segment to estimate baseline
    post_mask = time_ms >= pulse_end_ms
    y_post = mean_activity[post_mask]

    if len(y_post) < 20:
        return np.nan

    # baseline = mean of last 20% of post-pulse activity
    tail_len = max(10, len(y_post) // 5)
    baseline = np.mean(y_post[-tail_len:])

    # fit only early post-pulse decay
    fit_window_ms = 300
    fit_time_mask = (time_ms >= pulse_end_ms) & (time_ms <= pulse_end_ms + fit_window_ms)

    t_fit_all = time_ms[fit_time_mask] - pulse_end_ms
    y_fit_all = mean_activity[fit_time_mask] - baseline

    # keep only positive values above baseline
    fit_mask = y_fit_all > 0.02

    t_fit = t_fit_all[fit_mask]
    y_fit = y_fit_all[fit_mask]

    if len(y_fit) < 20:
        return np.nan

    # require a clear decay
    if y_fit[0] <= y_fit[-1] * 1.2:
        return np.nan

    try:
        popt, _ = curve_fit(
            exp_decay,
            t_fit,
            y_fit,
            p0=(y_fit[0], 100.0),
            bounds=([0, 1e-6], [np.inf, 1000]),
            maxfev=10000
        )

        tau_fit = popt[1]

        # reject fits that hit upper bound
        if tau_fit >= 999:
            return np.nan

        return tau_fit

    except RuntimeError:
        return np.nan


# -------------------
# General settings
# -------------------

num_runs = 20
N = 100

duration = 3000 * ms
dt = 0.1 * ms

baseline_input = 0.0
pulse_amplitude = 0.8
pulse_start = 500 * ms
pulse_end = 1000 * ms

tau_baseline = 20 * ms

# Balanced E/I connectivity settings
exc_fraction = 0.5
num_exc = int(round(exc_fraction * N))

w_exc = 0.6
mu = -np.log(np.sqrt(N))
sigma = 0.8

# Network divided into groups
num_groups = 4
group_size = N // num_groups

# Structured connectivity strengths
structure_values = [0.0, 0.02, 0.05, 0.08]

# -------------------
# Storage
# -------------------

results_mean_activity = {}
results_std_activity = {}

results_mean_corr = {}
results_std_corr = {}

timescale_results = {}

representative_t = None
representative_r = {}
lags = None

representative_W = {}
representative_neuron_types = {}

# -------------------
# Simulations
# -------------------

for structure_strength in structure_values:

    all_mean_activity = []
    all_corr = []

    print(f"\nRunning structured connectivity strength = {structure_strength}")

    for run_id in range(num_runs):

        start_scope()

        current_seed = 1000 * run_id + int(structure_strength * 10000)
        seed(current_seed)
        np.random.seed(current_seed)

        defaultclock.dt = dt

        # -------------------
        # Model equations
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
        # Balanced E/I connectivity matrix
        # -------------------

        neuron_types = np.ones(N)
        neuron_types[num_exc:] = -1
        np.random.shuffle(neuron_types)

        exc_cols = np.where(neuron_types == 1)[0]
        inh_cols = np.where(neuron_types == -1)[0]

        # base log-normal connection magnitudes
        W = np.random.lognormal(
            mean=mu,
            sigma=sigma,
            size=(N, N)
        )

        # remove self-connections before balancing
        np.fill_diagonal(W, 0)

        # -------------------
        # Add structured connectivity inside groups
        # -------------------
        # This multiplicatively strengthens within-group magnitudes.
        # structure_strength = 0.0 gives the unstructured baseline.
        for g in range(num_groups):
            start_idx = g * group_size
            end_idx = (g + 1) * group_size

            W[start_idx:end_idx, start_idx:end_idx] *= (1.0 + structure_strength)

        # excitatory columns positive
        W[:, exc_cols] = w_exc * W[:, exc_cols]

        # balance inhibitory columns so total recurrent weight sum ≈ 0
        exc_sum = np.sum(W[:, exc_cols])
        inh_sum = np.sum(W[:, inh_cols])
        ei_scale = exc_sum / inh_sum

        W[:, inh_cols] = -W[:, inh_cols] * ei_scale

        if run_id == 0:
            representative_W[structure_strength] = W.copy()
            representative_neuron_types[structure_strength] = neuron_types.copy()

            print(f"Total W sum: {np.sum(W):.10f}")
            print(f"Mean row sum: {np.mean(np.sum(W, axis=1)):.10f}")
            print(f"Mean col sum: {np.mean(np.sum(W, axis=0)):.10f}")
            print(f"Std of W: {np.std(W):.6f}")
            print(f"Min W: {np.min(W):.6f}")
            print(f"Max W: {np.max(W):.6f}")

        # -------------------
        # Input update each timestep
        # -------------------
        @network_operation(dt=defaultclock.dt)
        def update_input():

            if pulse_start <= defaultclock.t < pulse_end:
                input_signal = pulse_amplitude
            else:
                input_signal = baseline_input

            G.total_input = input_signal + np.dot(W, G.r)

        # -------------------
        # Monitor and run
        # -------------------
        M = StateMonitor(G, 'r', record=True)

        run(duration)

        mean_activity = np.mean(M.r, axis=0)
        corr = autocorrelation(mean_activity)

        all_mean_activity.append(mean_activity)
        all_corr.append(corr[:1000])

        # Save one representative run
        if run_id == 0:

            representative_r[structure_strength] = np.asarray(M.r)

            if representative_t is None:
                representative_t = np.asarray(M.t / ms)

            if lags is None:
                lags = np.arange(len(corr)) * float(defaultclock.dt / ms)

    # -------------------
    # Average across runs
    # -------------------
    all_mean_activity = np.array(all_mean_activity)
    all_corr = np.array(all_corr)

    results_mean_activity[structure_strength] = np.mean(all_mean_activity, axis=0)
    results_std_activity[structure_strength] = np.std(all_mean_activity, axis=0)

    results_mean_corr[structure_strength] = np.mean(all_corr, axis=0)
    results_std_corr[structure_strength] = np.std(all_corr, axis=0)


# -------------------
# Estimate post-pulse timescale
# -------------------

print("\nEstimated post-pulse timescales:")

for structure_strength in structure_values:

    tau_eff = estimate_post_pulse_timescale(
        results_mean_activity[structure_strength],
        representative_t,
        pulse_end / ms
    )

    timescale_results[structure_strength] = tau_eff

    if np.isnan(tau_eff):
        print(
            f"structure strength = {structure_strength} -> post-pulse timescale = undefined"
        )
    else:
        print(
            f"structure strength = {structure_strength} -> post-pulse timescale = {tau_eff:.2f} ms"
        )


# -------------------
# Plots
# -------------------

# 1. Representative single-neuron activity
plt.figure(figsize=(10, 6))

structure_example = 0.05

for i in range(5):
    plt.plot(
        representative_t,
        representative_r[structure_example][i],
        label=f'Neuron {i}'
    )

plt.axvspan(
    pulse_start / ms,
    pulse_end / ms,
    alpha=0.2,
    label='Input pulse'
)

plt.xlabel('Time (ms)')
plt.ylabel('Rate activity')
plt.title(
    f'Structured connectivity: representative single-neuron activity '
    f'(structure = {structure_example})'
)
plt.legend()
plt.tight_layout()


# 2. Mean network activity across runs
plt.figure(figsize=(10, 6))

for structure_strength in structure_values:

    mean_curve = results_mean_activity[structure_strength]

    plt.plot(
        representative_t,
        mean_curve,
        label=f'structure = {structure_strength}'
    )

plt.axvspan(
    pulse_start / ms,
    pulse_end / ms,
    alpha=0.2,
    label='Input pulse'
)

plt.xlabel('Time (ms)')
plt.ylabel('Mean activity')
plt.title('Structured connectivity: mean network activity')
plt.legend()
plt.tight_layout()


# 3. Autocorrelation across runs
plt.figure(figsize=(10, 6))

for structure_strength in structure_values:

    mean_corr_curve = results_mean_corr[structure_strength]

    plt.plot(
        lags[:1000],
        mean_corr_curve,
        label=f'structure = {structure_strength}'
    )

plt.xlabel('Lag (ms)')
plt.ylabel('Autocorrelation')
plt.title('Structured connectivity: autocorrelation')
plt.legend()
plt.tight_layout()


# 4. Post-pulse timescale vs structured connectivity
plt.figure(figsize=(8, 5))

x_vals = []
y_vals = []
first_nan = True

for structure_strength in structure_values:

    tau_eff = timescale_results[structure_strength]

    if np.isnan(tau_eff):
        if first_nan:
            plt.scatter(
                structure_strength,
                0,
                marker='x',
                s=100,
                label='undefined'
            )
            first_nan = False
        else:
            plt.scatter(
                structure_strength,
                0,
                marker='x',
                s=100
            )

        plt.text(structure_strength, 5, 'undefined', ha='center')

    else:
        x_vals.append(structure_strength)
        y_vals.append(tau_eff)

plt.plot(
    x_vals,
    y_vals,
    marker='o',
    label='estimated timescale'
)

plt.xlabel('Structure strength')
plt.ylabel('Post-pulse timescale (ms)')
plt.title('Post-pulse timescale vs structured connectivity')
plt.legend()
plt.tight_layout()


# 5. Sorted connectivity matrices
plt.figure(figsize=(12, 10))
plt.suptitle('Structured balanced connectivity matrices', fontsize=16)

for ii, structure_strength in enumerate(structure_values):

    W_plot = representative_W[structure_strength]
    neuron_types = representative_neuron_types[structure_strength]

    sorted_idx = np.argsort(neuron_types)
    W_sorted = W_plot[sorted_idx][:, sorted_idx]

    plt.subplot(2, 2, ii + 1)

    plt.imshow(
        W_sorted,
        cmap='bwr',
        aspect='auto'
    )

    plt.colorbar(label='Connection strength')
    plt.title(f'structure = {structure_strength}')
    plt.xlabel('Neuron index sorted by type')
    plt.ylabel('Neuron index sorted by type')

plt.tight_layout()

plt.show()