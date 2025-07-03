from efmloader import EFMDataLoader
from datetime import datetime

# 1. Create loader
loader = EFMDataLoader("/data/THUNDERMILL/waveform/", verbose=True)

# 2. Select time range (UTC)
from_time = datetime(2025, 6, 27, 17, 30)
to_time   = datetime(2025, 6, 27, 23, 30)

# 3. Load data
df = loader.load_range(from_time, to_time)
print(df.head())

# 4. Set Hi/Lo indices (use from previous waveform analysis)
hi = [12, 49, 85]
lo = [30, 66, 102]
loader.set_extraction_pattern(hi, lo)

# 5. Extract EFM intensity
efm_df = loader.extract_efm_intensity()
print(efm_df.head())
