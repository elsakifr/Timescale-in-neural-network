# from brian2 import *
# import matplotlib.pyplot as plt
# import numpy as np
# from scipy.optimize import curve_fit

# prefs.codegen.target = "numpy"

# # -------------------
# # Helper functions
# # -------------------

# def autocorrelation(x):
#     x = np.asarray(x)
#     x = x - np.mean(x)
#     corr = np.correlate(x, x, mode='full')
#     corr = corr[corr.size // 2:]

#     if corr[0] == 0:
#         return np.full_like(corr, np.nan)

#     corr = corr / corr[0]
#     return corr


# def exp_decay(t, tau):
#     return np.exp(-t / tau)


# def exp_decay_with_amplitude(t, A, tau):
#     return A * np.exp(-t / tau)


# def estimate_post_pulse_timescale(mean_activity, time_ms, pulse_end_ms):
#     """
#     Estimate timescale from the decay of mean activity after the input pulse is removed.
#     Baseline is estimated from the late post-pulse activity.
#     Fit is performed only on the early post-pulse decay window.
#     """

#     mean_activity = np.asarray(mean_activity)
#     time_ms = np.asarray(time_ms)

#     # Full post-pulse segment, used to estimate final baseline
#     post_mask = time_ms >= pulse_end_ms
#     y_post = mean_activity[post_mask]

#     if len(y_post) < 20:
#         return np.nan

#     # Baseline = average of last 20% of post-pulse activity
#     tail_len = max(10, len(y_post) // 5)
#     baseline = np.mean(y_post[-tail_len:])

#     # Fit only early decay after pulse
#     fit_window_ms = 300
#     fit_time_mask = (time_ms >= pulse_end_ms) & (time_ms <= pulse_end_ms + fit_window_ms)

#     t_fit_all = time_ms[fit_time_mask] - pulse_end_ms
#     y_fit_all = mean_activity[fit_time_mask] - baseline

#     # Keep only positive values above baseline
#     fit_mask = y_fit_all > 0.02

#     t_fit = t_fit_all[fit_mask]
#     y_fit = y_fit_all[fit_mask]

#     if len(y_fit) < 20:
#         return np.nan

#     # Require actual decay
#     if y_fit[0] <= y_fit[-1] * 1.2:
#         return np.nan

#     try:
#         popt, _ = curve_fit(
#             exp_decay_with_amplitude,
#             t_fit,
#             y_fit,
#             p0=(y_fit[0], 100.0),
#             bounds=([0, 1e-6], [np.inf, 1000]),
#             maxfev=10000
#         )

#         tau_fit = popt[1]

#         # If fit hits upper bound, treat as undefined
#         if tau_fit >= 999:
#             return np.nan

#         return tau_fit

#     except RuntimeError:
#         return np.nan


# # -------------------
# # General settings
# # -------------------

# num_runs = 20
# N = 100

# duration = 3000 * ms
# dt = 0.1 * ms

# baseline_input = 0.0
# pulse_amplitude = 0.8
# pulse_start = 500 * ms
# pulse_end = 1000 * ms

# tau_mean = 20 * ms

# # different heterogeneity levels
# tau_std_values = [0, 2, 5, 8]

# # balanced E/I connectivity settings
# exc_fraction = 0.5
# w_exc = 0.6
# mu = -np.log(np.sqrt(N))
# sigma = 0.8

# # -------------------
# # storage
# # -------------------

# results_mean_activity = {}
# results_mean_corr = {}
# timescale_results = {}

# representative_t = None
# representative_r = {}
# representative_tau = {}

# lags = None

# representative_W = None
# representative_neuron_types = None

# # -------------------
# # simulations
# # -------------------

# for tau_std_ms in tau_std_values:

#     all_mean_activity = []
#     all_corr = []

#     print(f"\nRunning tau std = {tau_std_ms} ms")

#     for run_id in range(num_runs):

#         start_scope()

#         current_seed = 1000 * run_id + int(tau_std_ms * 100)
#         seed(current_seed)
#         np.random.seed(current_seed)

#         defaultclock.dt = dt

#         # model equations
#         eqs = '''
#         dr/dt = (-r + tanh(total_input))/tau_i : 1
#         total_input : 1
#         tau_i : second
#         '''

#         G = NeuronGroup(N, eqs, method='euler')

#         G.r = '0.05 * rand()'
#         G.total_input = baseline_input

#         # heterogeneous tau
#         if tau_std_ms == 0:

#             tau_values = np.ones(N) * float(tau_mean / ms)

#         else:

#             tau_values = np.random.normal(
#                 loc=float(tau_mean / ms),
#                 scale=tau_std_ms,
#                 size=N
#             )

#             tau_values = np.clip(tau_values, 1.0, None)

#         G.tau_i = tau_values * ms

#         # -------------------
#         # balanced E/I connectivity
#         # -------------------

#         num_exc = int(round(exc_fraction * N))

#         neuron_types = np.ones(N)
#         neuron_types[num_exc:] = -1
#         np.random.shuffle(neuron_types)

#         exc_cols = np.where(neuron_types == 1)[0]
#         inh_cols = np.where(neuron_types == -1)[0]

#         W = np.random.lognormal(mean=mu, sigma=sigma, size=(N, N))
#         np.fill_diagonal(W, 0)

#         # excitatory columns positive
#         W[:, exc_cols] = w_exc * W[:, exc_cols]

#         # balance inhibitory columns so total W sum is close to zero
#         exc_sum = np.sum(W[:, exc_cols])
#         inh_sum = np.sum(W[:, inh_cols])
#         ei_scale = exc_sum / inh_sum

#         W[:, inh_cols] = -W[:, inh_cols] * ei_scale

#         if run_id == 0:
#             print(f"Total W sum: {np.sum(W):.10f}")
#             print(f"Mean row sum: {np.mean(np.sum(W, axis=1)):.10f}")
#             print(f"Mean col sum: {np.mean(np.sum(W, axis=0)):.10f}")

#             if representative_W is None:
#                 representative_W = W.copy()
#                 representative_neuron_types = neuron_types.copy()

#         # input update each timestep
#         @network_operation(dt=defaultclock.dt)
#         def update_input():

#             if pulse_start <= defaultclock.t < pulse_end:
#                 input_signal = pulse_amplitude
#             else:
#                 input_signal = baseline_input

#             G.total_input = input_signal + np.dot(W, G.r)

#         # monitor
#         M = StateMonitor(G, 'r', record=True)

#         run(duration)

#         mean_activity = np.mean(M.r, axis=0)

#         corr = autocorrelation(mean_activity)

#         all_mean_activity.append(mean_activity)
#         all_corr.append(corr[:1000])

#         # save one example run
#         if run_id == 0:

#             if representative_t is None:
#                 representative_t = M.t / ms

#             representative_r[tau_std_ms] = np.array(M.r)
#             representative_tau[tau_std_ms] = tau_values

#             if lags is None:
#                 lags = np.arange(len(corr)) * float(defaultclock.dt/ms)

#     # average across runs
#     all_mean_activity = np.array(all_mean_activity)
#     all_corr = np.array(all_corr)

#     results_mean_activity[tau_std_ms] = np.mean(all_mean_activity, axis=0)
#     results_mean_corr[tau_std_ms] = np.mean(all_corr, axis=0)


# # -------------------
# # compute post-pulse decay timescale
# # -------------------

# for tau_std_ms in tau_std_values:

#     tau_eff = estimate_post_pulse_timescale(
#         results_mean_activity[tau_std_ms],
#         np.asarray(representative_t),
#         pulse_end / ms
#     )

#     timescale_results[tau_std_ms] = tau_eff

#     if np.isnan(tau_eff):
#         print(
#             f"tau std = {tau_std_ms} ms -> post-pulse timescale = undefined"
#         )
#     else:
#         print(
#             f"tau std = {tau_std_ms} ms -> post-pulse timescale = {tau_eff:.2f} ms"
#         )


# # -------------------
# # plots
# # -------------------

# # 1 single neurons
# plt.figure(figsize=(10,6))

# tau_example = 5

# for i in range(5):

#     plt.plot(
#         representative_t,
#         representative_r[tau_example][i],
#         label=f'neuron {i}'
#     )

# plt.axvspan(pulse_start/ms, pulse_end/ms, alpha=0.2)

# plt.xlabel('time (ms)')
# plt.ylabel('rate')

# plt.title(
#     f'single neuron activity (tau std = {tau_example} ms)'
# )

# plt.legend()
# plt.tight_layout()

# # 2 mean activity
# plt.figure(figsize=(10,6))

# for tau_std_ms in tau_std_values:

#     plt.plot(
#         representative_t,
#         results_mean_activity[tau_std_ms],
#         label=f'tau std = {tau_std_ms}'
#     )

# plt.axvspan(pulse_start/ms, pulse_end/ms, alpha=0.2)

# plt.xlabel('time (ms)')
# plt.ylabel('mean activity')

# plt.title('mean network activity')

# plt.legend()
# plt.tight_layout()

# # 3 autocorrelation
# plt.figure(figsize=(10,6))

# for tau_std_ms in tau_std_values:

#     plt.plot(
#         lags[:1000],
#         results_mean_corr[tau_std_ms],
#         label=f'tau std = {tau_std_ms}'
#     )

# plt.xlabel('lag (ms)')
# plt.ylabel('autocorrelation')

# plt.title('autocorrelation')

# plt.legend()
# plt.tight_layout()

# # 4 tau distribution
# plt.figure(figsize=(10,6))

# for tau_std_ms in tau_std_values:

#     plt.hist(
#         representative_tau[tau_std_ms],
#         bins=15,
#         alpha=0.5,
#         label=f'tau std = {tau_std_ms}'
#     )

# plt.xlabel('tau_i (ms)')
# plt.ylabel('count')

# plt.title('distribution of tau_i')

# plt.legend()
# plt.tight_layout()

# # 5 post-pulse timescale vs heterogeneity
# plt.figure(figsize=(8,5))

# x_vals = []
# y_vals = []

# for tau_std_ms in tau_std_values:
#     tau_eff = timescale_results[tau_std_ms]

#     if np.isnan(tau_eff):
#         plt.scatter(tau_std_ms, 0, marker='x', s=100)
#         plt.text(tau_std_ms, 5, 'undefined', ha='center')
#     else:
#         x_vals.append(tau_std_ms)
#         y_vals.append(tau_eff)

# plt.plot(x_vals, y_vals, marker='o')

# plt.xlabel('tau std (ms)')
# plt.ylabel('post-pulse timescale (ms)')

# plt.title('post-pulse timescale vs tau heterogeneity')

# plt.tight_layout()

# # 6 sorted connectivity matrix
# if representative_W is not None:

#     sorted_idx = np.argsort(representative_neuron_types)
#     W_sorted = representative_W[sorted_idx][:, sorted_idx]

#     plt.figure(figsize=(6,6))
#     plt.imshow(W_sorted, cmap='bwr', aspect='auto')
#     plt.colorbar(label='connection strength')
#     plt.xlabel('neuron index sorted by type')
#     plt.ylabel('neuron index sorted by type')
#     plt.title('balanced sorted connectivity matrix W')
#     plt.tight_layout()

# plt.show()







from brian2 import *
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit
from scipy.signal import correlate

prefs.codegen.target = "numpy"

# -------------------
# Helper functions
# -------------------

def autocorrelation(x):
    x = np.asarray(x)
    x = x - np.mean(x)

    corr = correlate(x, x, mode='full', method='fft')
    corr = corr[corr.size // 2:]

    if corr[0] == 0:
        return np.full_like(corr, np.nan)

    return corr / corr[0]


def exp_decay_with_amplitude(t, A, tau):
    return A * np.exp(-t / tau)


def first_zero_crossing(corr, lags):
    idx = np.where(corr <= 0)[0]

    if len(idx) == 0:
        return np.nan

    return lags[idx[0]]


def first_baseline_crossing(mean_activity, time_ms, pulse_end_ms):
    mean_activity = np.asarray(mean_activity)
    time_ms = np.asarray(time_ms)

    post_mask = time_ms >= pulse_end_ms
    t_post = time_ms[post_mask] - pulse_end_ms
    y_post = mean_activity[post_mask]

    if len(y_post) < 20:
        return np.nan

    tail_len = max(10, len(y_post) // 5)
    baseline = np.mean(y_post[-tail_len:])

    y_shifted = y_post - baseline

    crossing_idx = np.where(y_shifted <= 0)[0]

    if len(crossing_idx) == 0:
        return np.nan

    return t_post[crossing_idx[0]]


def estimate_post_pulse_timescale(mean_activity, time_ms, pulse_end_ms):
    """
    Estimate timescale from the first post-pulse relaxation phase.
    The fit starts after pulse offset and stops when activity first
    reaches/crosses the estimated baseline.
    """

    mean_activity = np.asarray(mean_activity)
    time_ms = np.asarray(time_ms)

    post_mask = time_ms >= pulse_end_ms
    t_post = time_ms[post_mask] - pulse_end_ms
    y_post = mean_activity[post_mask]

    if len(y_post) < 20:
        return np.nan

    # Baseline = last 20% of post-pulse activity
    tail_len = max(10, len(y_post) // 5)
    baseline = np.mean(y_post[-tail_len:])

    y_decay = y_post - baseline

    # Start from first positive point after pulse end
    positive_idx = np.where(y_decay > 0)[0]

    if len(positive_idx) < 20:
        return np.nan

    start_idx = positive_idx[0]

    # Stop at first baseline crossing after start
    crossing_candidates = np.where(y_decay[start_idx:] <= 0)[0]

    if len(crossing_candidates) > 0:
        end_idx = start_idx + crossing_candidates[0]
    else:
        max_fit_ms = 300
        end_idx = np.searchsorted(t_post, max_fit_ms)

    t_fit = t_post[start_idx:end_idx]
    y_fit = y_decay[start_idx:end_idx]

    # Use only clearly positive values
    fit_mask = y_fit > 0.02
    t_fit = t_fit[fit_mask]
    y_fit = y_fit[fit_mask]

    if len(y_fit) < 20:
        return np.nan

    if y_fit[0] <= y_fit[-1] * 1.2:
        return np.nan

    try:
        popt, _ = curve_fit(
            exp_decay_with_amplitude,
            t_fit,
            y_fit,
            p0=(y_fit[0], 20.0),
            bounds=([0, 1e-6], [np.inf, 1000]),
            maxfev=10000
        )

        tau_fit = popt[1]

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

num_steps = int(duration / dt)
post_pulse_steps = int((duration - pulse_end) / dt)

tau_mean = 20 * ms

# Different heterogeneity levels
tau_std_values = [0, 2, 5, 8]

# Balanced E/I connectivity settings
exc_fraction = 0.5
num_exc = int(round(exc_fraction * N))
num_inh = N - num_exc

w_exc = 0.6
mu = -np.log(np.sqrt(N))
sigma = 0.8

# Background noise
noise_sigma = 0.05

# -------------------
# Storage
# -------------------

results_mean_activity = {}
results_std_activity = {}

results_mean_corr = {}
results_std_corr = {}

timescale_results = {}
baseline_crossing_results = {}
autocorr_zero_crossing_results = {}

representative_t = None
representative_r = {}
representative_tau = {}

representative_W = None
representative_neuron_types = None

lags = None

# -------------------
# Simulations
# -------------------

for tau_std_ms in tau_std_values:

    all_mean_activity = np.zeros((num_runs, num_steps))
    all_corr = np.zeros((num_runs, post_pulse_steps))

    print(f"\nRunning tau std = {tau_std_ms} ms")

    for run_id in range(num_runs):

        start_scope()

        current_seed = 1000 * run_id + int(tau_std_ms * 100)
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
        G.total_input = baseline_input

        # -------------------
        # Heterogeneous tau
        # -------------------
        if tau_std_ms == 0:
            tau_values = np.ones(N) * float(tau_mean / ms)
        else:
            tau_values = np.random.normal(
                loc=float(tau_mean / ms),
                scale=tau_std_ms,
                size=N
            )

            tau_values = np.clip(tau_values, 1.0, None)

        G.tau_i = tau_values * ms

        # -------------------
        # Balanced E/I connectivity
        # -------------------
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

        # Balance inhibitory columns so total W sum is close to zero
        exc_sum = np.sum(W[:, exc_cols])
        inh_sum = np.sum(W[:, inh_cols])
        ei_scale = exc_sum / inh_sum

        W[:, inh_cols] = -W[:, inh_cols] * ei_scale

        if run_id == 0:
            row_sums = np.sum(W, axis=1)
            col_sums = np.sum(W, axis=0)

            print(f"E/I = {num_exc}/{num_inh}")
            print(f"ei_scale = {ei_scale:.6f}")
            print(f"Total W sum: {np.sum(W):.10f}")
            print(f"Mean row sum: {np.mean(row_sums):.10f}")
            print(f"Std row sum: {np.std(row_sums):.10f}")
            print(f"Mean col sum: {np.mean(col_sums):.10f}")
            print(f"Std col sum: {np.std(col_sums):.10f}")
            print(f"Tau mean: {np.mean(tau_values):.3f} ms")
            print(f"Tau std: {np.std(tau_values):.3f} ms")
            print(f"Min tau: {np.min(tau_values):.3f} ms")
            print(f"Max tau: {np.max(tau_values):.3f} ms")

            if representative_W is None:
                representative_W = W.copy()
                representative_neuron_types = neuron_types.copy()

        # -------------------
        # Input update
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

        # Autocorrelation only after pulse end
        pulse_end_idx = int((pulse_end / ms) / (dt / ms))
        post_pulse_activity = mean_activity[pulse_end_idx:]

        corr = np.asarray(autocorrelation(post_pulse_activity))

        all_mean_activity[run_id, :] = mean_activity
        all_corr[run_id, :] = corr

        # Save one example run
        if run_id == 0:

            if representative_t is None:
                representative_t = np.asarray(M.t / ms)

            representative_r[tau_std_ms] = np.asarray(M.r)
            representative_tau[tau_std_ms] = tau_values.copy()

            if lags is None:
                lags = np.arange(len(corr)) * float(dt / ms)

    # -------------------
    # Average across runs
    # -------------------
    results_mean_activity[tau_std_ms] = np.mean(all_mean_activity, axis=0)
    results_std_activity[tau_std_ms] = np.std(all_mean_activity, axis=0)

    results_mean_corr[tau_std_ms] = np.mean(all_corr, axis=0)
    results_std_corr[tau_std_ms] = np.std(all_corr, axis=0)


# -------------------
# Compute post-pulse decay timescale and crossings
# -------------------

print("\nEstimated post-pulse timescales:")

for tau_std_ms in tau_std_values:

    tau_eff = estimate_post_pulse_timescale(
        results_mean_activity[tau_std_ms],
        representative_t,
        pulse_end / ms
    )

    crossing_time = first_baseline_crossing(
        results_mean_activity[tau_std_ms],
        representative_t,
        pulse_end / ms
    )

    zero_crossing = first_zero_crossing(
        results_mean_corr[tau_std_ms],
        lags
    )

    timescale_results[tau_std_ms] = tau_eff
    baseline_crossing_results[tau_std_ms] = crossing_time
    autocorr_zero_crossing_results[tau_std_ms] = zero_crossing

    if np.isnan(tau_eff):
        tau_text = "undefined"
    else:
        tau_text = f"{tau_eff:.2f} ms"

    if np.isnan(crossing_time):
        crossing_text = "no baseline crossing"
    else:
        crossing_text = f"{crossing_time:.2f} ms"

    if np.isnan(zero_crossing):
        zero_text = "no zero crossing"
    else:
        zero_text = f"{zero_crossing:.2f} ms"

    print(
        f"tau std = {tau_std_ms} ms -> post-pulse timescale = {tau_text}, "
        f"baseline crossing = {crossing_text}, "
        f"post-pulse autocorr zero crossing = {zero_text}"
    )


# -------------------
# Plots
# -------------------

# 1. Single neurons
plt.figure(figsize=(10, 6))

tau_example = 5

for i in range(5):
    plt.plot(
        representative_t,
        representative_r[tau_example][i],
        label=f'neuron {i}'
    )

plt.axvspan(
    pulse_start / ms,
    pulse_end / ms,
    alpha=0.2,
    label='input pulse'
)

plt.xlabel('time (ms)')
plt.ylabel('rate')
plt.title(f'single neuron activity (tau std = {tau_example} ms)')
plt.legend()
plt.tight_layout()


# 2. Mean activity
plt.figure(figsize=(10, 6))

for tau_std_ms in tau_std_values:
    plt.plot(
        representative_t,
        results_mean_activity[tau_std_ms],
        label=f'tau std = {tau_std_ms}'
    )

plt.axvspan(
    pulse_start / ms,
    pulse_end / ms,
    alpha=0.2,
    label='input pulse'
)

plt.xlabel('time (ms)')
plt.ylabel('mean activity')
plt.title('mean network activity')
plt.legend()
plt.tight_layout()


# 3. Zoomed post-pulse mean activity
plt.figure(figsize=(10, 6))

zoom_start = pulse_end / ms - 50
zoom_end = pulse_end / ms + 500
zoom_mask = (representative_t >= zoom_start) & (representative_t <= zoom_end)

for tau_std_ms in tau_std_values:
    m = results_mean_activity[tau_std_ms]

    plt.plot(
        representative_t[zoom_mask],
        m[zoom_mask],
        label=f'tau std = {tau_std_ms}'
    )

plt.axvline(
    pulse_end / ms,
    linestyle='--',
    label='pulse end'
)

plt.axhline(
    0,
    linestyle=':',
    label='zero activity'
)

plt.xlabel('time (ms)')
plt.ylabel('mean activity')
plt.title('zoomed post-pulse mean activity')
plt.legend()
plt.tight_layout()


# 4. Post-pulse autocorrelation zoom
plt.figure(figsize=(10, 6))

max_lag_zoom = 600
lag_mask = lags <= max_lag_zoom

for tau_std_ms in tau_std_values:
    m = results_mean_corr[tau_std_ms]

    plt.plot(
        lags[lag_mask],
        m[lag_mask],
        label=f'tau std = {tau_std_ms}'
    )

    zero_crossing = autocorr_zero_crossing_results[tau_std_ms]

    if not np.isnan(zero_crossing) and zero_crossing <= max_lag_zoom:
        plt.axvline(
            zero_crossing,
            linestyle='--',
            alpha=0.4
        )

plt.axhline(0, linestyle=':')

plt.xlabel('lag after pulse end (ms)')
plt.ylabel('autocorrelation')
plt.title('post-pulse autocorrelation zoom')
plt.legend()
plt.tight_layout()


# 5. Tau distribution as line plot
plt.figure(figsize=(10, 6))

x_tau = np.linspace(0, 45, 500)

for tau_std_ms in tau_std_values:

    tau_vals = representative_tau[tau_std_ms]

    if tau_std_ms == 0:
        plt.axvline(
            float(tau_mean / ms),
            linestyle='--',
            label='tau std = 0'
        )
    else:
        mean_tau = np.mean(tau_vals)
        std_tau = np.std(tau_vals)

        density = (
            1 / (std_tau * np.sqrt(2 * np.pi))
            * np.exp(-0.5 * ((x_tau - mean_tau) / std_tau) ** 2)
        )

        plt.plot(
            x_tau,
            density,
            label=f'tau std = {tau_std_ms}'
        )

plt.xlabel('tau_i (ms)')
plt.ylabel('density')
plt.title('distribution of tau_i')
plt.legend()
plt.tight_layout()


# 6. Fast vs slow neuron groups
plt.figure(figsize=(10, 6))

tau_example = 8

r_example = representative_r[tau_example]
tau_vals = representative_tau[tau_example]

sorted_idx = np.argsort(tau_vals)

n_group = N // 4

fast_idx = sorted_idx[:n_group]
slow_idx = sorted_idx[-n_group:]

fast_mean = np.mean(r_example[fast_idx], axis=0)
slow_mean = np.mean(r_example[slow_idx], axis=0)

plt.plot(
    representative_t,
    fast_mean,
    label='fastest 25% neurons'
)

plt.plot(
    representative_t,
    slow_mean,
    label='slowest 25% neurons'
)

plt.axvspan(
    pulse_start / ms,
    pulse_end / ms,
    alpha=0.2,
    label='input pulse'
)

plt.xlabel('time (ms)')
plt.ylabel('mean activity')
plt.title('fast vs slow neuron activity groups')
plt.legend()
plt.tight_layout()


# 7. Zoomed fast vs slow groups after pulse
plt.figure(figsize=(10, 6))

plt.plot(
    representative_t[zoom_mask],
    fast_mean[zoom_mask],
    label='fastest 25% neurons'
)

plt.plot(
    representative_t[zoom_mask],
    slow_mean[zoom_mask],
    label='slowest 25% neurons'
)

plt.axvline(
    pulse_end / ms,
    linestyle='--',
    label='pulse end'
)

plt.axhline(
    0,
    linestyle=':',
    label='zero activity'
)

plt.xlabel('time (ms)')
plt.ylabel('mean activity')
plt.title('zoomed fast vs slow neuron groups after pulse')
plt.legend()
plt.tight_layout()


# 8. Post-pulse timescale vs tau heterogeneity
plt.figure(figsize=(8, 5))

x_vals = []
y_vals = []

for tau_std_ms in tau_std_values:
    tau_eff = timescale_results[tau_std_ms]

    if np.isnan(tau_eff):
        plt.scatter(tau_std_ms, 0, marker='x', s=100)
        plt.text(tau_std_ms, 5, 'undefined', ha='center')
    else:
        x_vals.append(tau_std_ms)
        y_vals.append(tau_eff)

plt.plot(x_vals, y_vals, marker='o')

plt.xlabel('tau std (ms)')
plt.ylabel('post-pulse timescale (ms)')
plt.title('post-pulse timescale vs tau heterogeneity')
plt.tight_layout()


# 9. Baseline crossing time
plt.figure(figsize=(8, 5))

x_cross = []
y_cross = []

for tau_std_ms in tau_std_values:
    crossing_time = baseline_crossing_results[tau_std_ms]

    if not np.isnan(crossing_time):
        x_cross.append(tau_std_ms)
        y_cross.append(crossing_time)

plt.plot(
    x_cross,
    y_cross,
    marker='o'
)

plt.xlabel('tau std (ms)')
plt.ylabel('first baseline crossing after pulse end (ms)')
plt.title('baseline crossing time vs tau heterogeneity')
plt.tight_layout()


# 10. Sorted connectivity matrix
if representative_W is not None:

    sorted_idx = np.argsort(representative_neuron_types)
    W_sorted = representative_W[sorted_idx][:, sorted_idx]

    plt.figure(figsize=(6, 6))

    absmax = np.max(np.abs(W_sorted))

    plt.imshow(
        W_sorted,
        cmap='bwr',
        aspect='auto',
        vmin=-absmax,
        vmax=absmax
    )

    plt.colorbar(label='connection strength')
    plt.xlabel('neuron index sorted by type')
    plt.ylabel('neuron index sorted by type')
    plt.title('balanced sorted connectivity matrix W')
    plt.tight_layout()

plt.show()