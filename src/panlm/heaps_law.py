import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from numba import njit, prange


@njit
def polyfit_numba(x, y, deg):
    """
    Numba-compatible polynomial fitting using least squares.
    Returns coefficients [slope, intercept] for deg=1
    """
    n = x.shape[0]
    mat = np.zeros((n, deg + 1), dtype=np.float64)
    mat[:, 0] = 1.0
    for d in range(1, deg + 1):
        mat[:, d] = x**d
    coeffs = np.linalg.lstsq(mat, y)[0]
    return coeffs[1], coeffs[0]  # slope, intercept


@njit(parallel=True, fastmath=True)
def ultra_fast_permutations(
    per_strain_arrays, per_strain_lengths, orders, n_perm, S, C, logn
):
    """
    ULTRA-OPTIMIZED for 1000+ genomes with proper Numba type handling.
    """
    # Output arrays
    pan_curves = np.zeros((n_perm, S), dtype=np.int32)
    new_curves = np.zeros((n_perm, S), dtype=np.int32)
    core_curves = np.zeros((n_perm, S), dtype=np.int32)
    shell_curves = np.zeros((n_perm, S), dtype=np.int32)
    cloud_curves = np.zeros((n_perm, S), dtype=np.int32)
    gamma_vals = np.zeros(n_perm, dtype=np.float64)
    k_pan_vals = np.zeros(n_perm, dtype=np.float64)
    alpha_vals = np.zeros(n_perm, dtype=np.float64)
    k_new_vals = np.zeros(n_perm, dtype=np.float64)

    # Process permutations in parallel
    for p in prange(n_perm):
        order = orders[p]

        # Track cluster counts efficiently
        cluster_counts = np.zeros(C, dtype=np.int32)
        seen = np.zeros(C, dtype=np.bool_)
        pan_count = 0

        for i in range(S):
            idx = order[i]
            length = per_strain_lengths[idx]
            new_added = 0

            # Update cluster counts
            for j in range(length):
                cluster_idx = per_strain_arrays[idx, j]
                if not seen[cluster_idx]:
                    seen[cluster_idx] = True
                    new_added += 1
                cluster_counts[cluster_idx] += 1

            pan_count += new_added
            pan_curves[p, i] = pan_count
            new_curves[p, i] = new_added if i > 0 else pan_count

            # Calculate core/shell/cloud only on active clusters
            n_genomes_so_far = i + 1
            core = 0
            shell = 0
            cloud = 0
            threshold_95 = n_genomes_so_far * 0.95
            threshold_15 = n_genomes_so_far * 0.15

            # Only iterate over clusters that have been seen
            for c_idx in range(C):
                if seen[c_idx]:
                    count = min(cluster_counts[c_idx], n_genomes_so_far)
                    if count >= threshold_95:
                        core += 1
                    elif count >= threshold_15:
                        shell += 1
                    else:
                        cloud += 1

            core_curves[p, i] = core
            shell_curves[p, i] = shell
            cloud_curves[p, i] = cloud

        # Fit pangenome model: log(P) = log(k) + gamma * log(n)
        # Convert int32 array to float64 for log operation
        pan_float = pan_curves[p, :].astype(np.float64)
        y = np.log(pan_float + 1e-10)
        slope, intercept = polyfit_numba(logn, y, 1)
        gamma_vals[p] = slope
        k_pan_vals[p] = np.exp(intercept)

        # Fit new genes model: log(Δ) = log(k) - alpha * log(n)
        # Count valid points first
        valid_count = 0
        for idx in range(1, S):
            if new_curves[p, idx] > 0:
                valid_count += 1

        if valid_count >= 3:
            xs = np.zeros(valid_count, dtype=np.float64)
            ys = np.zeros(valid_count, dtype=np.float64)
            k = 0
            for idx in range(1, S):
                if new_curves[p, idx] > 0:
                    xs[k] = logn[idx]
                    # Direct conversion: int32 -> float64
                    ys[k] = np.log(np.float64(new_curves[p, idx]))
                    k += 1

            slope_a, intercept_a = polyfit_numba(xs, ys, 1)
            alpha_vals[p] = -slope_a  # negative because we want decay
            k_new_vals[p] = np.exp(intercept_a)
        else:
            alpha_vals[p] = 0.0
            k_new_vals[p] = 0.0

    return (
        pan_curves,
        new_curves,
        core_curves,
        shell_curves,
        cloud_curves,
        gamma_vals,
        k_pan_vals,
        alpha_vals,
        k_new_vals,
    )


def compute_heaps_law(
    df: pd.DataFrame,
    cluster_col: str = "cluster_id",
    strain_col: str = "strain",
    n_perm: int = 200,
    seed: int = 0,
    ci: tuple = (2.5, 97.5),
    fit_pan: bool = True,
    fit_new: bool = True,
    min_points_new: int = 3,
    return_permutation_params: bool = True,
    subsample_genomes: int = None,
):
    """
    ULTRA-OPTIMIZED for 1000+ genomes.

    NEW PARAMETER:
    --------------
    subsample_genomes : int, optional
        Randomly subsample this many genomes for faster computation.
        Recommended: 100-300 for quick estimates, None for full analysis.

    Expected time for 1000 genomes, 200 permutations: 5-15 minutes
    With subsampling (300 genomes): 1-2 minutes
    """
    import time

    start_time = time.time()

    # Validation
    required = {cluster_col, strain_col}
    missing_cols = required - set(df.columns)
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    # Prepare data
    df = df[[cluster_col, strain_col]].copy()
    df[cluster_col] = df[cluster_col].astype(str)
    df[strain_col] = df[strain_col].astype(str)
    df = df[(df[cluster_col].str.len() > 0) & (df[strain_col].str.len() > 0)]

    if df.empty:
        raise ValueError("No valid rows after filtering.")

    # Subsample genomes if requested
    if subsample_genomes is not None:
        unique_strains = df[strain_col].unique()
        if len(unique_strains) > subsample_genomes:
            rng = np.random.default_rng(seed)
            selected_strains = rng.choice(
                unique_strains, size=subsample_genomes, replace=False
            )
            df = df[df[strain_col].isin(selected_strains)]

    # Map to integers
    clusters, cluster_codes = np.unique(df[cluster_col].values, return_inverse=True)
    C = len(clusters)
    strain_vals = df[strain_col].values
    strains, strain_inv = np.unique(strain_vals, return_inverse=True)
    S = len(strains)

    # Build per-strain arrays efficiently using a dict first
    from collections import defaultdict

    strain_clusters = defaultdict(set)
    for i in range(len(df)):
        strain_clusters[strain_inv[i]].add(cluster_codes[i])

    # Convert to arrays
    max_len = max(len(s) for s in strain_clusters.values())
    per_strain_arrays = np.zeros((S, max_len), dtype=np.int32)
    per_strain_lengths = np.zeros(S, dtype=np.int32)

    for strain_idx, clusters_set in strain_clusters.items():
        clusters_list = list(clusters_set)
        per_strain_lengths[strain_idx] = len(clusters_list)
        per_strain_arrays[strain_idx, : len(clusters_list)] = clusters_list

    # Filter empty strains
    keep = per_strain_lengths > 0
    per_strain_arrays = per_strain_arrays[keep]
    per_strain_lengths = per_strain_lengths[keep]
    strains = strains[keep]
    S = len(strains)

    if S < 2:
        raise ValueError("At least two genomes required.")

    # Generate all permutations
    rng = np.random.default_rng(seed)
    orders = np.zeros((n_perm, S), dtype=np.int32)
    for p in range(n_perm):
        orders[p] = rng.permutation(S)

    # Precompute log values
    n = np.arange(1, S + 1, dtype=np.int64)
    logn = np.log(n.astype(np.float64))

    # Run computation
    comp_start = time.time()
    (
        pan_curves,
        new_curves,
        core_curves,
        shell_curves,
        cloud_curves,
        gamma_vals,
        k_pan_vals,
        alpha_vals,
        k_new_vals,
    ) = ultra_fast_permutations(
        per_strain_arrays, per_strain_lengths, orders, n_perm, S, C, logn
    )

    pan_mean = np.mean(pan_curves, axis=0)
    pan_sd = np.std(pan_curves, axis=0)
    new_mean = np.mean(new_curves, axis=0)
    new_sd = np.std(new_curves, axis=0)
    core_mean = np.mean(core_curves, axis=0)
    core_sd = np.std(core_curves, axis=0)
    shell_mean = np.mean(shell_curves, axis=0)
    shell_sd = np.std(shell_curves, axis=0)
    cloud_mean = np.mean(cloud_curves, axis=0)
    cloud_sd = np.std(cloud_curves, axis=0)

    # Filter valid values
    valid_alpha = alpha_vals[alpha_vals > 0]
    valid_k_new = k_new_vals[alpha_vals > 0]

    elapsed = time.time() - start_time

    result = {
        "n_genomes": S,
        "n_clusters_total": C,
        "n_perm_used": n_perm,
        "pan_mean": pan_mean,
        "pan_sd": pan_sd,
        "new_mean": new_mean,
        "new_sd": new_sd,
        "core_mean": core_mean,
        "core_sd": core_sd,
        "shell_mean": shell_mean,
        "shell_sd": shell_sd,
        "cloud_mean": cloud_mean,
        "cloud_sd": cloud_sd,
        "computation_time_seconds": elapsed,
    }

    if fit_pan:
        result["gamma_mean"] = float(np.mean(gamma_vals))
        result["gamma_ci"] = (
            float(np.percentile(gamma_vals, ci[0])),
            float(np.percentile(gamma_vals, ci[1])),
        )
        result["k_pan_mean"] = float(np.mean(k_pan_vals))
        result["open_by_gamma"] = bool(result["gamma_mean"] > 0.0)

    if fit_new and len(valid_alpha) > 0:
        result["alpha_mean"] = float(np.mean(valid_alpha))
        result["alpha_ci"] = (
            float(np.percentile(valid_alpha, ci[0])),
            float(np.percentile(valid_alpha, ci[1])),
        )
        result["k_new_mean"] = float(np.mean(valid_k_new))
        result["open_by_alpha"] = bool(result["alpha_mean"] < 1.0)

    if return_permutation_params:
        result["per_perm_gamma"] = gamma_vals.tolist()
        result["per_perm_k_pan"] = k_pan_vals.tolist()
        result["per_perm_alpha"] = valid_alpha.tolist()
        result["per_perm_k_new"] = valid_k_new.tolist()

    print(f"\n{'='*70}")
    print("Heaps' Law Analysis")
    print(f"{'='*70}")
    print(f"Genomes: {S:,}, Clusters: {C:,}, Permutations: {n_perm}")

    if "gamma_mean" in result:
        print("\nPangenome characteristics:")
        print(
            f"  - Gamma (γ): {result['gamma_mean']:.4f} (95% CI: {result['gamma_ci'][0]:.4f}-{result['gamma_ci'][1]:.4f})"
        )

    if "alpha_mean" in result:
        print(
            f"  - Alpha (α): {result['alpha_mean']:.4f} (95% CI: {result['alpha_ci'][0]:.4f}-{result['alpha_ci'][1]:.4f})"
        )
        print(f"  - Relationship: α = 1 - γ = {1 - result['gamma_mean']:.4f}")
        print(
            f"  - Status: {'OPEN' if result['open_by_gamma'] else 'CLOSED'} pangenome"
        )

    return result


def plot_heaps_law(
    result: dict,
    figsize: tuple = (6, 4),
    color: str = "#2E86C1",
    marker: str = "-",
    markersize: float = 4,
    alpha: float = 0.7,
    xlabel: str = "Number of genomes",
    ylabel: str = "Number of protein clusters",
    title: str = None,
    show_confidence: bool = True,
    ci_alpha: float = 0.15,
    save_path: str = None,
    dpi: int = 300,
):
    """
    Publication-quality Heaps' law plot with seaborn styling matching reference.
    """
    # Set seaborn theme to match reference
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font="serif",
        font_scale=1.1,
        rc={
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.figsize": figsize,
        },
    )

    n_genomes = result.get("n_genomes", len(result["pan_mean"]))
    pan_mean = result["pan_mean"]
    n_genomes_range = np.arange(1, n_genomes + 1)

    fig, ax = plt.subplots(figsize=figsize)

    if show_confidence and "pan_sd" in result:
        pan_sd = result["pan_sd"]
        ax.fill_between(
            n_genomes_range,
            pan_mean - pan_sd,
            pan_mean + pan_sd,
            alpha=ci_alpha,
            color=color,
            linewidth=0,
            label="±1 SD",
        )

    ax.plot(
        n_genomes_range,
        pan_mean,
        color=color,
        marker=marker,
        linewidth=1.5,
        alpha=alpha,
        label="Pangenome size",
    )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    if title is None and "gamma_mean" in result:
        openness = "Open" if result.get("open_by_gamma", True) else "Closed"
        title = f'Pangenome accumulation (γ={result["gamma_mean"]:.3f}, {openness})'

    if title:
        ax.set_title(title)

    ax.set_xlim(0, n_genomes * 1.02)
    ax.set_ylim(0, max(pan_mean) * 1.05)
    ax.legend(loc="lower right", frameon=True)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, format="pdf", bbox_inches="tight")

    return fig, ax


def plot_heaps_biplot(
    result: dict,
    figsize: tuple = (12, 6),
    colors: dict = None,
    marker: str = "-",
    markersize: float = 4,
    alpha: float = 0.7,
    xlabel: str = "Number of genomes",
    ylabel_left: str = "Number of protein clusters",
    ylabel_right: str = "Proportion of Core/Shell/Cloud (%)",
    title_left: str = None,
    title_right: str = "Core/Shell/Cloud genome evolution",
    show_confidence: bool = True,
    ci_alpha: float = 0.15,
    save_path: str = None,
    dpi: int = 300,
    legend_loc: str = "best",
):
    """
    biplot with A/B labels matching reference style.
    """
    # Set seaborn theme to match reference
    sns.set_theme(
        style="whitegrid",
        context="paper",
        font="serif",
        font_scale=1.1,
        rc={
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 7,
            "figure.figsize": figsize,
        },
    )

    if colors is None:
        colors = {
            "pan": "#2E86C1",
            "core": "#27AE60",
            "shell": "#F39C12",
            "cloud": "#3498DB",
        }

    n_genomes = result.get("n_genomes", len(result["pan_mean"]))
    n_genomes_range = np.arange(1, n_genomes + 1)

    pan_mean = result["pan_mean"]
    core_mean = result.get("core_mean", np.zeros(n_genomes))
    shell_mean = result.get("shell_mean", np.zeros(n_genomes))
    cloud_mean = result.get("cloud_mean", np.zeros(n_genomes))

    fig, axes = plt.subplots(1, 2, figsize=figsize)

    # Left plot (A): Pangenome
    if show_confidence and "pan_sd" in result:
        pan_sd = result["pan_sd"]
        axes[0].fill_between(
            n_genomes_range,
            pan_mean - pan_sd,
            pan_mean + pan_sd,
            alpha=ci_alpha,
            color=colors["pan"],
            linewidth=0,
        )

    axes[0].plot(
        n_genomes_range,
        pan_mean,
        color=colors["pan"],
        marker=marker,
        linewidth=1.5,
        alpha=alpha,
        label="Pangenome",
    )

    axes[0].set_xlabel(xlabel)
    axes[0].set_ylabel(ylabel_left)

    if title_left is None and "gamma_mean" in result:
        openness = "Open" if result.get("open_by_gamma", True) else "Closed"
        title_left = (
            f'Pangenome accumulation (γ={result["gamma_mean"]:.3f}, {openness})'
        )

    axes[0].set_title(title_left)
    axes[0].set_xlim(0, n_genomes * 1.02)
    axes[0].set_ylim(0, max(pan_mean) * 1.05)

    # Add label 'A' just outside the top-left corner
    axes[0].text(
        -0.15,
        1.02,
        "A",
        transform=axes[0].transAxes,
        fontsize=12,
        fontweight="bold",
        ha="right",
        va="bottom",
    )

    total = core_mean + shell_mean + cloud_mean

    total = np.where(total == 0, 1, total)
    core_pct = (core_mean / total) * 100
    shell_pct = (shell_mean / total) * 100
    cloud_pct = (cloud_mean / total) * 100

    if show_confidence:
        # For percentages, we'll show the raw confidence bands (optional, can be removed)
        for _name, mean, sd_key, color in [
            ("core", core_pct, "core_sd", colors["core"]),
            ("shell", shell_pct, "shell_sd", colors["shell"]),
            ("cloud", cloud_pct, "cloud_sd", colors["cloud"]),
        ]:
            if sd_key in result:
                # Calculate percentage SD (approximation)
                raw_sd = result[sd_key]
                pct_sd = (raw_sd / total) * 100
                axes[1].fill_between(
                    n_genomes_range,
                    mean - pct_sd,
                    mean + pct_sd,
                    alpha=ci_alpha,
                    color=color,
                    linewidth=0,
                )

    axes[1].plot(
        n_genomes_range,
        core_pct,
        color=colors["core"],
        marker=marker,
        linewidth=1.5,
        alpha=alpha,
        label="Core",
    )
    axes[1].plot(
        n_genomes_range,
        shell_pct,
        color=colors["shell"],
        marker=marker,
        linewidth=1.5,
        alpha=alpha,
        label="Shell",
    )
    axes[1].plot(
        n_genomes_range,
        cloud_pct,
        color=colors["cloud"],
        marker=marker,
        linewidth=1.5,
        alpha=alpha,
        label="Cloud",
    )

    axes[1].set_xlabel(xlabel)
    axes[1].set_ylabel(ylabel_right)
    axes[1].set_title(title_right)
    axes[1].set_xlim(0, n_genomes * 1.02)
    axes[1].set_ylim(0, 105)  # Set to 105% to give some headroom
    axes[1].legend(loc=legend_loc, frameon=True)

    axes[1].text(
        -0.15,
        1.02,
        "B",
        transform=axes[1].transAxes,
        fontsize=12,
        fontweight="bold",
        ha="right",
        va="bottom",
    )

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=dpi, format="pdf", bbox_inches="tight")
        print(f"Saved plot: {save_path}")

    print(f"{'='*70}\n")
    return fig, axes


if __name__ == "__main__":

    path = "/data/nilar/pan_genome/full_analysis_output/G1000/output_m0915_sd_0005_pca450.csv"
    df = pd.read_csv(path)
    df = df[["strain", "cluster_id"]].copy()

    result_quick = compute_heaps_law(
        df,
        cluster_col="cluster_id",
        strain_col="strain",
        n_perm=1034,
        subsample_genomes=None,
    )

    fig1, ax1 = plot_heaps_law(
        result_quick, show_confidence=True, save_path="heaps_law.pdf", marker=None
    )

    fig2, axes = plot_heaps_biplot(
        result_quick, show_confidence=True, save_path="heaps_biplot.pdf", marker=None
    )

    plt.show()
