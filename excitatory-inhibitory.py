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
    corr = np.correlate(x, x, mode="full")
    corr = corr[corr.size // 2:]

    if corr[0] == 0:
        return np.full_like(corr, np.nan)

    return corr / corr[0]


def exp_decay(t, A, tau):
    return A * np.exp(-t / tau)


def estimate_timescale_from_return_to_baseline(
    mean_activity,
    time_ms,
    pulse_start_ms,
    pulse_end_ms,
    min_amplitude=0.02,
    return_fraction=0.25,
):
    time_ms = np.asarray(time_ms)
    y = np.asarray(mean_activity)

    baseline_mask = (time_ms >= pulse_start_ms - 200) & (time_ms < pulse_start_ms)
    if np.sum(baseline_mask) < 10:
        return np.nan

    baseline = np.mean(y[baseline_mask])

    post_mask = time_ms >= pulse_end_ms
    t_post = time_ms[post_mask] - pulse_end_ms
    y_post = y[post_mask]

    response = y_post - baseline

    if len(response) < 50:
        return np.nan

    initial_window = (t_post >= 0) & (t_post <= 50)
    if np.sum(initial_window) < 10:
        return np.nan

    A0 = np.mean(response[initial_window])

    if A0 <= min_amplitude:
        return np.nan

    tail_len = max(20, len(response) // 5)
    final_response = np.mean(response[-tail_len:])

    if abs(final_response) > return_fraction * abs(A0):
        return np.nan

    fit_mask = response > 0.05 * A0
    t_fit = t_post[fit_mask]
    y_fit = response[fit_mask]

    if len(y_fit) < 30:
        return np.nan

    try:
        popt, _ = curve_fit(
            exp_decay,
            t_fit,
            y_fit,
            p0=(A0, 100.0),
            bounds=([0, 1e-6], [np.inf, np.inf]),
            maxfev=10000,
        )

        tau = popt[1]

        if tau > t_post[-1]:
            return np.nan

        return tau

    except RuntimeError:
        return np.nan


# -------------------
# Connectivity builders
# -------------------
def build_connectivity(case_name, N, exc_percent, w_exc, con_prob=0.2, log_sigma=0.05):
    NE = int(round(N * exc_percent / 100))
    NI = N - NE

    if NI == 0:
        raise ValueError("NI cannot be zero.")

    neuron_types = np.ones(N)
    neuron_types[NE:] = -1

    con_mat = np.zeros((N, N))

    # -------------------
    # Case 1: sparse random connections + rescaling
    # -------------------
    if case_name == "case1_sparse_rescaled":
        for jj in range(N):
            rid = np.random.uniform(0, 1, N)
            tgt_ids = np.where(rid < con_prob)[0]
            con_mat[tgt_ids, jj] = 1

        np.fill_diagonal(con_mat, 0)

        con_mat[:, :NE] *= w_exc

        exc_sum = np.sum(con_mat[:, :NE])
        inh_sum = np.sum(con_mat[:, NE:])

        if inh_sum == 0:
            return con_mat, neuron_types, np.nan

        ei_scale = exc_sum / inh_sum
        con_mat[:, NE:] = -con_mat[:, NE:] * ei_scale

    # -------------------
    # Case 2: sparse fixed outdegree
    # -------------------
    elif case_name == "case2_fixed_outdegree":
        no_con = int(round(con_prob * N))
        w_inh = -w_exc * (NE / NI)

        for jj in range(N):
            possible_targets = np.array([i for i in range(N) if i != jj])
            tgt_ids = np.random.choice(possible_targets, size=no_con, replace=False)
            con_mat[tgt_ids, jj] = 1

        con_mat[:, :NE] *= w_exc
        con_mat[:, NE:] *= w_inh

    # -------------------
    # Case 3: fully connected log-normal + rescaling
    # -------------------
    elif case_name == "case3_lognormal_rescaled":
        con_mat = np.random.lognormal(mean=0.0, sigma=log_sigma, size=(N, N))

        # Important scaling for large N
        con_mat = con_mat / np.sqrt(N)

        np.fill_diagonal(con_mat, 0)

        con_mat[:, :NE] *= w_exc

        exc_sum = np.sum(con_mat[:, :NE])
        inh_sum = np.sum(con_mat[:, NE:])

        if inh_sum == 0:
            return con_mat, neuron_types, np.nan

        ei_scale = exc_sum / inh_sum
        con_mat[:, NE:] = -con_mat[:, NE:] * ei_scale

    else:
        raise ValueError(f"Unknown case name: {case_name}")

    return con_mat, neuron_types, np.sum(con_mat)


# -------------------
# Simulation settings
# -------------------
num_runs = 10
N = 500
exc_frac = [50, 60, 70, 80, 90]

duration = 4000 * ms
dt = 0.1 * ms

baseline_input = 0.0
pulse_amplitude = 0.8
pulse_start = 500 * ms
pulse_end = 1000 * ms
tau_baseline = 20 * ms

# Adjusted parameters
w_exc = 0.05
con_prob = 0.2
log_sigma = 0.05

# Keep noise off first; add later only if needed
noise_sigma = 0.0

cases_to_run = [
    "case1_sparse_rescaled",
    "case2_fixed_outdegree",
    "case3_lognormal_rescaled",
]


# -------------------
# Main experiment
# -------------------
for case_name in cases_to_run:
    print("\n" + "=" * 70)
    print(f"Running connectivity case: {case_name}")
    print("=" * 70)

    results_mean_activity = {}
    results_std_activity = {}
    results_mean_corr = {}
    results_std_corr = {}
    results_timescale = {}
    results_time = {}
    results_lags = {}
    ei_labels = {}
    representative_W = {}

    for exc_percent in exc_frac:
        all_mean_activity = []
        all_corr = []

        NE = int(round(N * exc_percent / 100))
        NI = N - NE
        ei_labels[exc_percent] = f"{NE}/{NI}"

        print(f"\nRunning E/I = {NE}/{NI}")

        for run_id in range(num_runs):
            start_scope()

            current_seed = 10000 * run_id + exc_percent
            seed(current_seed)
            np.random.seed(current_seed)
            defaultclock.dt = dt

            W, neuron_types, total_sum = build_connectivity(
                case_name=case_name,
                N=N,
                exc_percent=exc_percent,
                w_exc=w_exc,
                con_prob=con_prob,
                log_sigma=log_sigma,
            )

            if run_id == 0:
                representative_W[exc_percent] = W.copy()

                row_sums = np.sum(W, axis=1)
                col_sums = np.sum(W, axis=0)

                print(f"Total W sum: {np.sum(W):.10f}")
                print(f"Mean row sum: {np.mean(row_sums):.10f}")
                print(f"Std row sum: {np.std(row_sums):.10f}")
                print(f"Mean col sum: {np.mean(col_sums):.10f}")
                print(f"Std col sum: {np.std(col_sums):.10f}")
                print(f"Max inhibitory weight: {np.max(W[:, NE:]):.6f}")
                print(f"Min excitatory weight: {np.min(W[:, :NE]):.6f}")

            eqs = '''
            dr/dt = (-r + tanh(total_input))/tau_i : 1
            total_input : 1
            tau_i : second
            '''

            G = NeuronGroup(N, eqs, method="euler")
            G.r = "0.05 * rand()"
            G.tau_i = tau_baseline
            G.total_input = baseline_input

            @network_operation(dt=defaultclock.dt)
            def update_input():
                if pulse_start <= defaultclock.t < pulse_end:
                    input_signal = pulse_amplitude
                else:
                    input_signal = baseline_input

                if noise_sigma > 0:
                    noise = noise_sigma * np.random.randn(N)
                else:
                    noise = 0

                G.total_input = input_signal + np.dot(W, G.r) + noise

            M = StateMonitor(G, "r", record=True)
            run(duration)

            mean_activity = np.asarray(np.mean(M.r, axis=0))
            corr = np.asarray(autocorrelation(mean_activity))

            all_mean_activity.append(mean_activity.copy())
            all_corr.append(corr[:1000].copy())

            if run_id == 0:
                time_ms = np.asarray(M.t / ms)
                lags_ms = np.arange(len(corr)) * float(defaultclock.dt / ms)

        all_mean_activity = np.vstack(all_mean_activity)
        all_corr = np.vstack(all_corr)

        results_mean_activity[exc_percent] = np.mean(all_mean_activity, axis=0)
        results_std_activity[exc_percent] = np.std(all_mean_activity, axis=0)

        results_mean_corr[exc_percent] = np.mean(all_corr, axis=0)
        results_std_corr[exc_percent] = np.std(all_corr, axis=0)

        results_time[exc_percent] = time_ms
        results_lags[exc_percent] = lags_ms[:1000]

        results_timescale[exc_percent] = estimate_timescale_from_return_to_baseline(
            results_mean_activity[exc_percent],
            results_time[exc_percent],
            pulse_start / ms,
            pulse_end / ms,
        )

    # -------------------
    # Plot 1: mean activity
    # -------------------
    plt.figure(figsize=(10, 6))
    for exc_percent in exc_frac:
        plt.plot(
            results_time[exc_percent],
            results_mean_activity[exc_percent],
            label=f"E/I = {ei_labels[exc_percent]}",
        )

    plt.axvspan(pulse_start / ms, pulse_end / ms, alpha=0.2, label="Input pulse")
    plt.xlabel("Time (ms)")
    plt.ylabel("Mean activity")
    plt.title(f"{case_name}: mean population activity")
    plt.legend()
    plt.tight_layout()

    # -------------------
    # Plot 2: autocorrelation
    # -------------------
    plt.figure(figsize=(10, 6))
    for exc_percent in exc_frac:
        plt.plot(
            results_lags[exc_percent],
            results_mean_corr[exc_percent],
            label=f"E/I = {ei_labels[exc_percent]}",
        )

    plt.xlabel("Lag (ms)")
    plt.ylabel("Autocorrelation")
    plt.title(f"{case_name}: autocorrelation")
    plt.legend()
    plt.tight_layout()

    # -------------------
    # Plot 3: timescale
    # -------------------
    plt.figure(figsize=(8, 5))

    valid_x = []
    valid_y = []
    first_nan = True

    for exc_percent in exc_frac:
        tau = results_timescale[exc_percent]
        x = exc_percent / 100

        if np.isnan(tau):
            if first_nan:
                plt.scatter(x, 0, marker="x", s=100, label="Undefined timescale")
                first_nan = False
            else:
                plt.scatter(x, 0, marker="x", s=100)

            plt.text(x, 5, "undefined", ha="center")
        else:
            valid_x.append(x)
            valid_y.append(tau)

    if len(valid_x) > 0:
        valid_pairs = sorted(zip(valid_x, valid_y))
        x_sorted = [p[0] for p in valid_pairs]
        y_sorted = [p[1] for p in valid_pairs]
        plt.plot(x_sorted, y_sorted, "o-", label="Estimated timescale")

    plt.xlabel("Excitatory fraction")
    plt.ylabel("Estimated timescale (ms)")
    plt.title(f"{case_name}: estimated post-pulse time constant")
    plt.legend()
    plt.tight_layout()

    # -------------------
    # Plot 4: connectivity matrices
    # -------------------
    absmax = max(np.max(np.abs(W)) for W in representative_W.values())

    plt.figure(figsize=(12, 10))
    plt.suptitle(f"{case_name}: representative connectivity matrices", fontsize=16)

    for ii, exc_percent in enumerate(exc_frac):
        W_plot = representative_W[exc_percent]

        plt.subplot(3, 2, ii + 1)
        plt.imshow(W_plot, cmap="bwr", aspect="auto", vmin=-absmax, vmax=absmax)
        plt.colorbar(label="Connection strength")
        plt.title(f"E/I = {ei_labels[exc_percent]}")
        plt.xlabel("Presynaptic neuron j")
        plt.ylabel("Postsynaptic neuron i")

    plt.tight_layout()

    print("\nEstimated post-pulse time constants:")
    for exc_percent in exc_frac:
        tau = results_timescale[exc_percent]
        if np.isnan(tau):
            print(f"E/I = {ei_labels[exc_percent]}: undefined")
        else:
            print(f"E/I = {ei_labels[exc_percent]}: {tau:.2f} ms")

plt.show()