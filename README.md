# EFM Loader

**EFM Loader** is a Python package for loading, processing, and visualizing electric field waveform data from THUNDERMILL stations. It supports batch loading from HDF5 files, waveform averaging, and extraction of EFM intensity using configurable waveform regions.

## Features

- Load waveform data from HDF5 files based on timestamp ranges
- Compute average waveform over a given time interval
- Extract EFM intensity using Hi/Lo sample regions
- Plotting tools for waveform visualization
- CSV comparison plotting (e.g. SEVAN vs EFM)

## Installation

```bash
pip install efmloader
