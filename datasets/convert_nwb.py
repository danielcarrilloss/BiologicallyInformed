"""
Convert AC / ALM NWB sessions into the folder structure expected by datasets/dataloader.py::TrialDataset 
and infopath/session_stitching.py::build_network (same layout as datasets/datastructure2datasetandvideo_Vahid.py).

Target layout, for each session under `out_root`:
    out_root/
        <session_name>/
            trial_info                  (csv)
            cluster_info                (csv)
            neuron_index_{i}.npy        (spike times in seconds, one file per neuron i)
            jaw_trace/trial_{t}.npy     (lick-rate trace, substitutes for jaw video trace)
        cluster_information             (csv, aggregated across all converted sessions)

Usage:
    python convert_nwb_sessions.py --nwb_dir /path/to/nwbs --out datasets/DataFromNWB_AC --area AC
"""

import os
import h5py
import argparse
import numpy as np
import pandas as pd
from scipy.ndimage import gaussian_filter1d

# --------------------------------------------------------------------------------------
# exc / inh classification
# --------------------------------------------------------------------------------------
# Reuses the convention from datasets/datastructure2datasetandvideo_Vahid.py:
#     excitatory = peak_to_trough_width > WIDTH_THR
# There it was `mat["width"]` (ms). Here the equivalent NWB Units column is
# `peak_to_valley`. CHECK UNITS: pynwb Units tables usually store this in seconds, so we
# convert the old 0.275 ms threshold to seconds. Before trusting this on your data, plot
# the peak_to_valley histogram per session and confirm it is bimodal around the cutoff --
# this threshold was calibrated on a different spike-sorting pipeline and may not transfer.
WIDTH_THR_S = 0.275e-3  # 0.275 ms
OUTPUT_FOLDER = "./NWBData"


def classify_excitatory(peak_to_valley_seconds):
    return np.asarray(peak_to_valley_seconds) > WIDTH_THR_S


# --------------------------------------------------------------------------------------
# trial type mapping -> dataloader.py / session_stitching.py expect
#   trial_type in {"Hit", "Miss", "CR", "FA", "EarlyLick"}
# --------------------------------------------------------------------------------------
TRIAL_TYPE_MAP = {
    "Hit": "Hit",
    "Miss": "Miss",
    "Correct": "CR",  # correct rejection
    "FalseAlarm": "FA",
}

def _decode(x):
    return x.decode() if isinstance(x, bytes) else x


def convert_session(nwb_path, out_root, area, lick_window=(-0.1, 3.0), lick_dt=0.002):
    """Convert a single .nwb file into the target folder structure. Returns session_name."""
    session_name = os.path.splitext(os.path.basename(nwb_path))[0]
    session_path = os.path.join(out_root, session_name)
    os.makedirs(session_path, exist_ok=True)
    os.makedirs(os.path.join(session_path, "jaw_trace"), exist_ok=True)

    with h5py.File(nwb_path, "r", load_namespaces=True) as io:
        nwbfile = io.read()

        # ---------------- units / neurons -----------------------------------------
        units = io["units"]
        n_units = len(units)

        # column name candidates -- adjust once you see the real dataframe
        ptv_col = units["peak_to_valley"]
        cluster_id_col = units["ks_unit_id"]

        cluster_df = pd.DataFrame(
            {
                "neuron_index": np.arange(n_units),
                "area": area,
                "excitatory": classify_excitatory(units[ptv_col].values),
                "depth": units["depth"].values if "depth" in units.columns else np.nan,
                "cluster": np.asarray(cluster_id_col),
                "firing_rate": units["firing_rate"].values,
                "with_video": 0,  # no jaw/tongue/whisker video for these sessions, licks only
            }
        )
        cluster_df.to_csv(os.path.join(session_path, "cluster_info"))

        for i in range(n_units):
            spike_times = np.asarray(units["spike_times"].iloc[i])
            np.save(os.path.join(session_path, f"neuron_index_{i}"), spike_times)

        # ---------------- trials ----------------------------------------------------
        trials_df = None
        if getattr(nwbfile, "trials", None) is not None:
            trials_df = nwbfile.trials.to_dataframe()
        elif "Trials" in nwbfile.acquisition:
            trials_df = nwbfile.acquisition["Trials"].to_dataframe()
        else:
            raise RuntimeError(f"could not locate trials table in {nwb_path}")

        trial_type_raw = [_decode(t) for t in trials_df["trial_type"].values]
        trial_type = [TRIAL_TYPE_MAP.get(t, "Miss") for t in trial_type_raw]
        n_trials = len(trials_df)

        trial_info = pd.DataFrame(
            {
                "trial_number": np.arange(n_trials),
                "reaction_time_piezo": trials_df["response_time"].values,
                "reaction_time_jaw": trials_df["response_time"].values,  # licks are the readout
                "stim": np.zeros(n_trials, dtype=int),  # adjust if sound_ids differentiate stims
                "trial_active": np.ones(n_trials, dtype=int),
                "trial_type": trial_type,
                "trial_onset": trials_df["start_time"].values,
                "jaw_trace": [
                    os.path.join(session_path, "jaw_trace", f"trial_{t}")
                    for t in range(n_trials)
                ],
                "tongue_trace": "",
                "whisker_angle": "",
                "completed_trials": np.ones(n_trials, dtype=int),
                "video_onset": -1,
                "video_offset": 3,
            }
        )
        trial_info.to_csv(os.path.join(session_path, "trial_info"))

        # ---------------- licks -> pseudo jaw_trace ---------------------------------
        licks = io["acquisition"]["Licks"]
        lick_times = np.asarray(
            licks["timestamps"][:] if licks["timestamps"] is not None else licks["timestamps"]
        )
        save_lick_traces(
            lick_times,
            trials_df["start_time"][:],
            session_path,
            window=lick_window,
            timestep=lick_dt,
        )

    return session_name


def save_lick_traces(lick_times, trial_onsets, session_path, window=(-0.1, 3.0), timestep=0.002):
    """Bin lick times into a smoothed rate trace per trial and save under jaw_trace/.
    Downstream code (dataloader.py) resamples this the same way it resamples the real
    jaw trace, so the sampling rate here doesn't need to match exactly -- just be
    finer than the model dt.
    """
    n_bins = int((window[1] - window[0]) / timestep)
    edges = np.arange(window[0], window[1] + timestep, timestep)
    for t, onset in enumerate(trial_onsets):
        rel = lick_times - onset
        rel = rel[(rel >= window[0]) & (rel <= window[1])]
        counts, _ = np.histogram(rel, bins=edges)
        trace = gaussian_filter1d(counts.astype(float), sigma=2)
        np.save(os.path.join(session_path, "jaw_trace", f"trial_{t}"), trace[:n_bins])


def unify_cluster_table(out_root):
    sessions = [
        d
        for d in os.listdir(out_root)
        if os.path.isdir(os.path.join(out_root, d))
    ]
    all_df = []
    for sess in sessions:
        df = pd.read_csv(os.path.join(out_root, sess, "cluster_info"), index_col=0)
        df = df.assign(session=sess, cluster_index=df.index.values)
        all_df.append(df[["session", "area", "excitatory", "firing_rate", "cluster_index", "with_video"]])
    pd.concat(all_df, ignore_index=True).to_csv(os.path.join(out_root, "cluster_information"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--nwb_dir", required=True, help="folder containing .nwb files for one area")
    parser.add_argument("--out", required=True, help="output dataset folder, e.g. datasets/DataFromNWB_AC")
    parser.add_argument("--area", required=True, help="area label to write into cluster_info, e.g. AC or ALM")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    converted = []
    for fname in sorted(os.listdir(args.nwb_dir)):
        if fname.endswith(".nwb"):
            print("converting", fname)
            converted.append(convert_session(os.path.join(args.nwb_dir, fname), args.out, args.area))

    unify_cluster_table(args.out)
    print(f"converted {len(converted)} sessions into {args.out}")

