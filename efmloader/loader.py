import os
import re
import h5py
import numpy as np
import pandas as pd
from typing import List, Optional
from datetime import datetime, date, timedelta, timezone
from scipy.signal import find_peaks

try:
    from tqdm import tqdm
except ImportError:
    tqdm = lambda x, **kwargs: x


def parse_timestamp_from_filename(fname: str) -> Optional[datetime]:
    match = re.search(r"_(\d{8})_(\d{4})\.h5$", fname)
    if not match:
        return None
    date_str, time_str = match.groups()
    try:
        dt = datetime.strptime(date_str + time_str, "%Y%m%d%H%M")
        return dt.replace(tzinfo=timezone.utc)
    except ValueError:
        return None


class EFMDataLoader:
    def __init__(self, folder_path: str, verbose: bool = True):
        self.folder_path = folder_path
        self.verbose = verbose
        self.all_files = self._find_h5_files()
        self.cached_df: Optional[pd.DataFrame] = None
        self.hi_indices: List[int] = []
        self.lo_indices: List[int] = []

    def _find_h5_files(self) -> List[str]:
        files = []
        for root, _, filenames in os.walk(self.folder_path):
            for fname in filenames:
                if fname.endswith(".h5"):
                    files.append(os.path.join(root, fname))
        if self.verbose:
            print(f"✓ Found {len(files)} HDF5 files")
        return sorted(files)

    def _select_files_in_range(self, from_ts: int, to_ts: int) -> List[str]:
        selected = []
        for file_path in self.all_files:
            fname = os.path.basename(file_path)
            file_start_dt = parse_timestamp_from_filename(fname)
            if not file_start_dt:
                continue
            file_start_ts = int(file_start_dt.timestamp() * 1000)
            file_end_ts = file_start_ts + 10 * 60 * 1000  # +10 minutes

            if file_end_ts >= from_ts and file_start_ts <= to_ts:
                selected.append(file_path)

        if self.verbose:
            print(f"✓ Selected {len(selected)} files in range")
        return selected

    def _load_files_in_range(self, from_ts: int, to_ts: int) -> pd.DataFrame:
        selected_files = self._select_files_in_range(from_ts, to_ts)
        results = []

        iterator = tqdm(selected_files, desc="Loading files", unit="file") if self.verbose else selected_files

        for file_path in iterator:
            try:
                with h5py.File(file_path, "r") as f:
                    if "metadata" not in f or "waveform" not in f:
                        continue

                    meta = f["metadata"][()]
                    df_meta = pd.DataFrame(meta)

                    if "timestamp" not in df_meta.columns:
                        continue

                    mask = (df_meta["timestamp"] >= from_ts) & (df_meta["timestamp"] <= to_ts)
                    if not mask.any():
                        continue

                    waveform_data = f["waveform"][()]
                    matching_rows = df_meta[mask].copy()
                    matching_rows["waveform"] = list(waveform_data[mask.to_numpy()])
                    results.append(matching_rows)

                    if self.verbose:
                        tqdm.write(f"✓ {os.path.basename(file_path)}: {mask.sum()} rows in range")

            except Exception as e:
                tqdm.write(f"⚠️ Error processing {file_path}: {e}")

        if results:
            return pd.concat(results, ignore_index=True)
        else:
            if self.verbose:
                tqdm.write("⚠️ No data found in the specified range.")
            return pd.DataFrame()

    def load_range(self, from_dt: datetime, to_dt: datetime) -> pd.DataFrame:
        from_ts = int(from_dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
        to_ts = int(to_dt.replace(tzinfo=timezone.utc).timestamp() * 1000)
        df = self._load_files_in_range(from_ts, to_ts)
        self.cached_df = df
        return df

    def load_day(self, day: date) -> pd.DataFrame:
        from_dt = datetime.combine(day, datetime.min.time(), tzinfo=timezone.utc)
        to_dt = from_dt + timedelta(days=1) - timedelta(milliseconds=1)
        return self.load_range(from_dt, to_dt)

    def load_hour(self, hour: datetime) -> pd.DataFrame:
        from_dt = hour.replace(minute=0, second=0, microsecond=0, tzinfo=timezone.utc)
        to_dt = from_dt + timedelta(hours=1) - timedelta(milliseconds=1)
        return self.load_range(from_dt, to_dt)

    def load_all(self) -> pd.DataFrame:
        return self.load_range(datetime.min.replace(tzinfo=timezone.utc),
                               datetime.max.replace(tzinfo=timezone.utc))

    def get_intensity_timeseries(self) -> pd.DataFrame:
        if self.cached_df is None or self.cached_df.empty:
            raise ValueError("No data loaded.")
        return self.cached_df[["timestamp", "ef"]].copy()

    def set_extraction_pattern(self, hi: List[int], lo: List[int]):
        self.hi_indices = hi
        self.lo_indices = lo
        if self.verbose:
            print(f"✓ Set {len(hi)} Hi and {len(lo)} Lo indices")

    def extract_efm_intensity(self) -> pd.DataFrame:
        if self.cached_df is None or self.cached_df.empty:
            raise ValueError("No data loaded.")
        if not self.hi_indices or not self.lo_indices:
            raise ValueError("Hi and Lo indices not set. Use set_extraction_pattern().")

        efm_values = []
        for wf in self.cached_df["waveform"]:
            hi_vals = np.take(wf, self.hi_indices)
            lo_vals = np.take(wf, self.lo_indices)
            efm = np.mean(hi_vals) - np.mean(lo_vals)
            efm_values.append(efm)

        self.cached_df["efm_intensity"] = efm_values

        if self.verbose:
            print(f"✓ Calculated {len(efm_values)} EFM intensities")

        return self.cached_df[["timestamp", "efm_intensity"]].copy()

    def average_waveform(self, from_dt: datetime, to_dt: datetime) -> np.ndarray:
        df = self.load_range(from_dt, to_dt)
        if df.empty:
            raise ValueError("No data available for averaging.")
        waveforms = np.stack(df["waveform"].to_numpy())
        avg = np.mean(waveforms, axis=0)
        if self.verbose:
            print(f"✓ Computed average waveform from {waveforms.shape[0]} rows")
        return avg


def get_top_hi_lo_indices(waveform: np.ndarray, count: int = 3, distance: int = 5) -> tuple[list[int], list[int]]:
    peaks, _ = find_peaks(waveform, distance=distance)
    troughs, _ = find_peaks(-waveform, distance=distance)

    top_peaks = sorted(peaks, key=lambda x: waveform[x], reverse=True)[:count]
    top_troughs = sorted(troughs, key=lambda x: waveform[x])[:count]

    return sorted(top_peaks), sorted(top_troughs)


def plot_waveform_with_hi_lo_lines(waveform: np.ndarray, hi: list[int], lo: list[int], title: str = "Waveform with Hi/Lo"):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(12, 4))
    plt.plot(waveform, label="Waveform", color="black")

    for x in hi:
        plt.axvline(x=x, color='red', linestyle='--', label='Hi' if x == hi[0] else "")
    for x in lo:
        plt.axvline(x=x, color='blue', linestyle='--', label='Lo' if x == lo[0] else "")

    plt.title(title)
    plt.xlabel("Samples")
    plt.ylabel("Amplitude")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()
