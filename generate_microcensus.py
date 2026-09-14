"""
Generate synthetic_commune_census_microcensus.csv from scratch.

Spec: synthetic_data_gen_prompt.md
Format: commune-level data combining census + microcensus columns
        (matching the existing CSV schema exactly)

All 34 provinces (2026 admin reform) are represented.
1000 rows total, distributed proportionally across 34 provinces.
"""

import numpy as np
import pandas as pd
import math

RANDOM_SEED = 42
N_ROWS = 1000
rng = np.random.default_rng(RANDOM_SEED)

OUTPUT_PATH = r"d:\Coding_prj\Final_prj_ML+DL\data\synthetic_commune_census_microcensus.csv"

# ─────────────────────────────────────────────────────────────────────────────
# 1. PROVINCE DEFINITIONS (34 tỉnh/thành 2026)
# ─────────────────────────────────────────────────────────────────────────────

PROVINCES = {
    # name: (region, pop_density_range, elevation_range, urban_ratio, mnar_rate, rainfall_range, n_communes_approx)
    # === Đồng bằng sông Hồng ===
    "Hà Nội":           ("dong_bang_song_hong",   (800, 8000),  (0,  50),   0.70, 0.05, (1200, 1800), 579),
    "Hải Phòng":        ("dong_bang_song_hong",   (400, 2500),  (0,  30),   0.55, 0.06, (1400, 2000), 223),
    "Hưng Yên":         ("dong_bang_song_hong",   (600, 2000),  (0,  20),   0.45, 0.06, (1200, 1700), 210),
    "Ninh Bình":        ("dong_bang_song_hong",   (300, 1500),  (0,  80),   0.40, 0.07, (1500, 2200), 190),
    "Bắc Ninh":         ("dong_bang_song_hong",   (400, 1500),  (0,  30),   0.50, 0.05, (1300, 1800), 150),
    "Quảng Ninh":       ("dong_bang_song_hong",   (200,  800),  (0, 300),   0.45, 0.08, (1800, 2500), 170),
    # === Miền núi phía Bắc ===
    "Cao Bằng":         ("mien_nui_phia_bac",     (40,  110),  (300, 1500), 0.05, 0.45, (1400, 2000), 149),
    "Lạng Sơn":         ("mien_nui_phia_bac",     (50,  130),  (200, 1200), 0.06, 0.40, (1200, 1800), 200),
    "Thái Nguyên":      ("mien_nui_phia_bac",     (80,  300),  (100,  800), 0.12, 0.30, (1400, 2000), 170),
    "Tuyên Quang":      ("mien_nui_phia_bac",     (35,  100),  (200, 1500), 0.05, 0.48, (1500, 2200), 138),
    "Lào Cai":          ("mien_nui_phia_bac",     (40,  150),  (400, 3143), 0.06, 0.50, (1800, 3000), 152),
    "Phú Thọ":          ("mien_nui_phia_bac",     (200,  700), (50,  600),  0.20, 0.25, (1400, 2000), 250),
    "Lai Châu":         ("mien_nui_phia_bac",     (25,   70),  (600, 2500), 0.04, 0.55, (2000, 3200), 96),
    "Điện Biên":        ("mien_nui_phia_bac",     (30,   90),  (400, 2000), 0.05, 0.50, (1600, 2500), 129),
    "Sơn La":           ("mien_nui_phia_bac",     (40,  100),  (400, 2000), 0.05, 0.48, (1400, 2200), 202),
    # === Bắc Trung Bộ & Duyên hải miền Trung ===
    "Thanh Hóa":        ("duyen_hai_mien_trung",  (180,  600), (0,  800),   0.30, 0.18, (1500, 2500), 520),
    "Nghệ An":          ("duyen_hai_mien_trung",  (100,  400), (0, 1000),   0.25, 0.20, (1400, 2200), 460),
    "Hà Tĩnh":          ("duyen_hai_mien_trung",  (100,  350), (0,  600),   0.25, 0.18, (1800, 2800), 229),
    "Quảng Trị":        ("duyen_hai_mien_trung",  (80,   300), (0,  600),   0.28, 0.20, (2000, 3000), 145),
    "Huế":              ("duyen_hai_mien_trung",  (300, 2200), (0,  400),   0.45, 0.12, (2500, 3500), 133),
    "Đà Nẵng":          ("duyen_hai_mien_trung",  (400, 2000), (0,  500),   0.55, 0.10, (2000, 3000), 105),
    "Quảng Ngãi":       ("duyen_hai_mien_trung",  (80,   400), (0,  800),   0.22, 0.22, (2000, 3200), 165),
    # === Tây Nguyên & Nam Trung Bộ ===
    "Gia Lai":          ("tay_nguyen",            (55,  200),  (400, 1500), 0.15, 0.30, (1600, 2400), 180),
    "Đắk Lắk":         ("tay_nguyen",            (60,  250),  (400, 1500), 0.18, 0.28, (1500, 2200), 200),
    "Khánh Hòa":        ("tay_nguyen",            (150,  800), (0,  800),   0.40, 0.20, (1200, 2000), 135),
    "Lâm Đồng":         ("tay_nguyen",            (80,  300),  (600, 2400), 0.25, 0.25, (1800, 2800), 130),
    # === Đông Nam Bộ ===
    "TP. Hồ Chí Minh":  ("dong_nam_bo",           (1000,14000),(0, 50),    0.85, 0.03, (1400, 2200), 321),
    "Đồng Nai":         ("dong_nam_bo",            (200, 1500), (0, 200),   0.40, 0.08, (1800, 2500), 135),
    "Tây Ninh":         ("dong_nam_bo",            (150,  700), (0, 150),   0.30, 0.10, (1800, 2500), 130),
    # === Đồng bằng sông Cửu Long ===
    "Đồng Tháp":        ("dong_bang_song_cuu_long",(250,  700), (0, 12),    0.28, 0.10, (1800, 2500), 130),
    "Vĩnh Long":        ("dong_bang_song_cuu_long",(300,  800), (0, 10),    0.30, 0.10, (1800, 2400), 107),
    "Cần Thơ":          ("dong_bang_song_cuu_long",(400, 1500), (0, 10),    0.40, 0.08, (1800, 2400), 142),
    "An Giang":         ("dong_bang_song_cuu_long",(280,  750), (0, 12),    0.30, 0.10, (1600, 2200), 138),
    "Cà Mau":           ("dong_bang_song_cuu_long",(150,  400), (0, 10),    0.22, 0.15, (2200, 2800), 100),
}

COMMUNE_TYPES = ["phuong", "thi_tran", "xa", "xa"]  # xa more common
AREA_BY_TYPE = {
    "phuong":   (2,  15),
    "thi_tran": (10, 40),
    "xa":       (20, 80),
}
LAND_USE_TYPES = ["residential", "agricultural", "forest", "mixed"]

# ─────────────────────────────────────────────────────────────────────────────
# 2. DISTRIBUTE ROWS ACROSS 34 PROVINCES
#    Proportional to n_communes_approx, minimum 15 rows per province
# ─────────────────────────────────────────────────────────────────────────────

province_names = list(PROVINCES.keys())
weights = np.array([PROVINCES[p][-1] for p in province_names], dtype=float)
weights = weights / weights.sum()

# Calculate row counts, ensure min 15 per province
raw_counts = np.maximum(15, np.round(weights * N_ROWS).astype(int))
# Adjust to hit exactly N_ROWS total
diff = N_ROWS - raw_counts.sum()
# Add/subtract from largest province proportionally
sorted_idx = np.argsort(raw_counts)[::-1]
for i in range(abs(diff)):
    raw_counts[sorted_idx[i % len(sorted_idx)]] += int(np.sign(diff))

province_row_counts = {p: int(raw_counts[i]) for i, p in enumerate(province_names)}

print(f"Total rows planned: {sum(province_row_counts.values())}")
print(f"Provinces: {len(province_row_counts)}")

# ─────────────────────────────────────────────────────────────────────────────
# 3. GENERATE ROWS
# ─────────────────────────────────────────────────────────────────────────────

rows = []
commune_counter = 0

for province, n_rows in province_row_counts.items():
    (region, density_range, elev_range, urban_ratio,
     mnar_rate, rainfall_range, n_communes_approx) = PROVINCES[province]

    for _ in range(n_rows):
        commune_counter += 1
        commune_id = f"COM_{commune_counter:04d}"

        # --- commune_type ---
        if rng.random() < urban_ratio * 0.4:
            ctype = "phuong"
        elif rng.random() < 0.15:
            ctype = "thi_tran"
        else:
            ctype = "xa"

        # --- area ---
        area_low, area_high = AREA_BY_TYPE.get(ctype, (20, 80))
        area_km2 = round(float(rng.uniform(area_low, area_high)), 4)

        # --- elevation ---
        elev = round(float(rng.uniform(elev_range[0], elev_range[1])), 2)

        # --- distance to urban ---
        if ctype == "phuong":
            dist_urban = round(float(rng.uniform(0.5, 10)), 2)
        elif ctype == "thi_tran":
            dist_urban = round(float(rng.uniform(2, 30)), 2)
        else:
            dist_urban = round(float(rng.uniform(5, 100)), 2)
        dist_urban = min(dist_urban, 100.0)

        # --- ndvi (vegetation index) ---
        # Urban/residential areas have low NDVI, forest high
        base_ndvi = rng.uniform(0.1, 0.9)
        if ctype == "phuong":
            ndvi = round(min(base_ndvi, 0.45), 4)
        else:
            ndvi = round(float(base_ndvi), 4)

        # --- road density ---
        if ctype == "phuong":
            road_density = round(float(rng.uniform(2.0, 10.0)), 4)
        elif ctype == "thi_tran":
            road_density = round(float(rng.uniform(0.5, 5.0)), 4)
        else:
            # High elevation -> sparse roads
            if elev > 1000:
                road_density = round(float(rng.uniform(0.01, 1.5)), 4)
            else:
                road_density = round(float(rng.uniform(0.05, 4.0)), 4)

        # --- land_use ---
        if ctype == "phuong":
            land_use = rng.choice(["residential", "mixed", "agricultural"], p=[0.6, 0.25, 0.15])
        elif ndvi > 0.65:
            land_use = "forest"
        elif ctype == "thi_tran":
            land_use = rng.choice(["agricultural", "residential", "mixed"], p=[0.5, 0.3, 0.2])
        else:
            land_use = rng.choice(LAND_USE_TYPES, p=[0.2, 0.45, 0.2, 0.15])

        # --- rainfall ---
        rainfall = round(float(rng.uniform(rainfall_range[0], rainfall_range[1])), 2)

        # --- nearest market ---
        if ctype == "phuong":
            market_km = round(float(rng.uniform(0.1, 5.0)), 2)
        elif ctype == "thi_tran":
            market_km = round(float(rng.uniform(0.5, 20.0)), 2)
        else:
            market_km = round(float(rng.uniform(1.0, 50.0)), 2)

        # --- census_population (rule-based) ---
        d_low, d_high = density_range

        # Elevation penalty: high elevation -> lower density
        if elev > 1500:
            d_high = min(d_high, 80)
            d_low  = min(d_low,  50)
        elif elev > 800:
            d_high = min(d_high, 300)

        # Distance penalty: far from urban -> lower density
        if dist_urban > 50:
            d_high = min(d_high, max(d_low + 50, d_high // 3))
        elif dist_urban < 5:
            d_low = max(d_low, d_high // 3)

        # Land use adjustment
        if land_use == "forest":
            d_high = min(d_high, max(d_low + 30, int(d_high * 0.3)))
        elif land_use == "residential":
            d_low = max(d_low, int(d_high * 0.4))

        # phuong -> higher density
        if ctype == "phuong":
            d_low = max(d_low, int(d_high * 0.35))

        # Ensure valid range
        if d_low >= d_high:
            d_high = d_low + max(50, int(d_low * 0.3))

        raw_density = rng.uniform(d_low, d_high)
        # Add 15% noise + 2% chance of outlier (khu công nghiệp)
        noise = rng.normal(0, 0.15)
        if rng.random() < 0.02:
            noise += rng.uniform(0.5, 1.5)  # outlier
        census_density = round(max(10.0, min(raw_density * (1 + noise), 20000.0)), 2)
        census_population = max(100, int(census_density * area_km2))

        # --- microcensus data ---
        n_points = int(rng.integers(5, 51))
        # microcensus total persons ~ census * (0.85 to 1.15)
        micro_ratio = rng.uniform(0.85, 1.15)
        micro_persons = max(10, int(census_population * micro_ratio))
        micro_households = max(3, int(micro_persons / rng.uniform(3.5, 5.5)))
        micro_mean_density = round(micro_persons / area_km2 * rng.uniform(0.9, 1.1), 2)

        # --- missing mechanism ---
        # MNAR: high elevation + far from urban
        is_mnar = (elev > 800 and dist_urban > 30)
        mnar_prob = mnar_rate
        if is_mnar:
            mnar_prob = max(mnar_rate, 0.60)

        rand_miss = rng.random()
        if rand_miss < mnar_prob * 0.8:
            has_data = 0
            miss_mech = "MNAR_high_density" if (census_density > 1000) else "MNAR_high_density"
            if is_mnar:
                miss_mech = "MNAR_high_density"
        elif rand_miss < mnar_prob * 0.8 + 0.05:
            has_data = 0
            miss_mech = "MAR"
        elif rand_miss < mnar_prob * 0.8 + 0.08:
            has_data = 0
            miss_mech = "MCAR"
        else:
            has_data = 1
            miss_mech = "observed"

        # --- n_grid_cells_100m (100m grid = area_km2 * 100) ---
        n_grid = max(10, int(area_km2 * 100))

        rows.append({
            "province":                     province,
            "region":                       region,
            "commune_id":                   commune_id,
            "commune_type":                 ctype,
            "area_commune_km2":             area_km2,
            "elevation_m":                  elev,
            "distance_to_urban_km":         dist_urban,
            "ndvi":                         ndvi,
            "road_density_km_per_km2":      road_density,
            "land_use_type":                land_use,
            "annual_rainfall_mm":           rainfall,
            "nearest_market_km":            market_km,
            "census_population":            census_population,
            "census_density":               census_density,
            "n_microcensus_points":         n_points,
            "microcensus_total_persons":    micro_persons,
            "microcensus_total_households": micro_households,
            "microcensus_mean_density":     micro_mean_density,
            "has_microcensus_data":         has_data,
            "missing_mechanism":            miss_mech,
            "n_grid_cells_100m":            n_grid,
        })

# ─────────────────────────────────────────────────────────────────────────────
# 4. ASSEMBLE DATAFRAME + SHUFFLE
# ─────────────────────────────────────────────────────────────────────────────
df = pd.DataFrame(rows)
df = df.sample(frac=1, random_state=RANDOM_SEED).reset_index(drop=True)
# Renumber commune_id after shuffle
df["commune_id"] = [f"COM_{i+1:04d}" for i in range(len(df))]

# ─────────────────────────────────────────────────────────────────────────────
# 5. SAVE
# ─────────────────────────────────────────────────────────────────────────────
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
print(f"\nSaved to: {OUTPUT_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# 6. SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("SYNTHETIC DATA SUMMARY")
print("=" * 60)
print(f"Total rows:     {len(df):,}")
print(f"Columns:        {len(df.columns)}")
print(f"Provinces:      {df['province'].nunique()}")
print(f"Regions:        {df['region'].nunique()}")

prov_counts = df.groupby(["region", "province"]).size().reset_index(name="count")
summary_path = r"d:\Coding_prj\Final_prj_ML+DL\generate_microcensus_summary.txt"
with open(summary_path, "w", encoding="utf-8") as sf:
    sf.write("Province distribution (rows):\n")
    for _, row in prov_counts.iterrows():
        sf.write(f"  {row['province']:25s} ({row['region']:30s}): {row['count']:3d}\n")
print(f"Province distribution saved to: {summary_path}")



print("\nRegion distribution:")
print(df["region"].value_counts().to_string())

print("\nCommune type distribution:")
print(df["commune_type"].value_counts().to_string())

print("\nMissing mechanism distribution:")
print(df["missing_mechanism"].value_counts().to_string())

print(f"\ncensus_density stats:")
print(f"  Mean:   {df['census_density'].mean():.1f}")
print(f"  Median: {df['census_density'].median():.1f}")
print(f"  Min:    {df['census_density'].min():.1f}")
print(f"  Max:    {df['census_density'].max():.1f}")

print(f"\nAll 34 provinces present: {df['province'].nunique() == 34}")
missing_provinces = set(PROVINCES.keys()) - set(df['province'].unique())
if missing_provinces:
    print(f"  MISSING: {missing_provinces}")
else:
    print("  All provinces covered!")

print("\nDone.")
