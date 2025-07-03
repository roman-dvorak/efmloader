from efmloader import EFMDataLoader, get_top_hi_lo_indices, plot_waveform_with_hi_lo_lines
from datetime import datetime

# 1. Create loader instance
loader = EFMDataLoader("/data/THUNDERMILL/waveform/", verbose=True)

# 2. Define time range (UTC)
from_time = datetime(2025, 6, 28, 13, 0)
to_time = datetime(2025, 6, 28, 14, 0)

# 3. Compute average waveform
avg_waveform = loader.average_waveform(from_time, to_time)

# 4. Detect top 3 local maxima and minima
hi, lo = get_top_hi_lo_indices(avg_waveform, count=3)

# 5. Plot waveform with vertical Hi/Lo lines
plot_waveform_with_hi_lo_lines(avg_waveform, hi, lo, title="Average waveform with Hi/Lo markers")
