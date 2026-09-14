# 🧾 PROMPT: SINH SYNTHETIC DATA CHO VIETPOPNET
> **Cách dùng:** Copy toàn bộ nội dung phần trong khung `═══` bên dưới → Paste vào Claude / ChatGPT / Gemini / Perplexity → Nhận về 1 file Python script sinh data.

---

```
═══════════════════════════════════════════════════════════════
NHIỆM VỤ: Viết Python script sinh synthetic dataset cho dự án
nghiên cứu ước tính mật độ dân số theo ô lưới không gian tại
Việt Nam. Script phải chạy được ngay, không cần API hay data
ngoài — chỉ dùng numpy, pandas, và random seed cố định.
═══════════════════════════════════════════════════════════════

## 1. BỐI CẢNH DỰ ÁN

Tên dự án: VietPopNet
Bài toán: Ước tính mật độ dân số (người/km²) tại các ô lưới
không gian Việt Nam chưa được điều tra thực địa, dựa trên đặc
trưng địa lý quan sát được từ vệ tinh và dữ liệu hành chính.

Đây là synthetic data để train mô hình Deep Learning (MLP
PyTorch). Data phải đủ thực tế để mô hình học được pattern
có nghĩa, nhưng không cần chính xác 100% — chỉ cần các
rule-based constraint sát với thực tế Việt Nam.

Đường dẫn lưu: data/raw/vietnam_grid_synthetic.csv

---

## 2. BỐI CẢNH HÀNH CHÍNH - ĐỊA LÝ VIỆT NAM 2026

### Cải cách hành chính 2025-2026 (BẮT BUỘC áp dụng):
- Việt Nam đã **bỏ cấp huyện** từ 2025
- Chỉ còn 2 cấp: **Tỉnh/Thành phố** → **Xã/Phường/Thị trấn**
- Cả nước có **34 tỉnh thành** chính xác sau sáp nhập:

  **Vùng Đồng bằng sông Hồng & Đông Bắc Bộ (mật độ CAO):**
  | STT | Tên tỉnh mới | Tỉnh sáp nhập |
  |-----|-------------|---------------|
  | 24 | **Hà Nội** | Giữ nguyên |
  | 7  | **Hải Phòng** | Hải Dương + Hải Phòng |
  | 6  | **Hưng Yên** | Hưng Yên + Thái Bình |
  | 8  | **Ninh Bình** | Hà Nam + Ninh Bình + Nam Định |
  | 5  | **Bắc Ninh** | Bắc Ninh + Bắc Giang |
  | 30 | **Quảng Ninh** | Giữ nguyên |

  **Vùng Trung du & Miền núi phía Bắc (mật độ THẤP):**
  | STT | Tên tỉnh mới | Tỉnh sáp nhập |
  |-----|-------------|---------------|
  | 34 | **Cao Bằng** | Giữ nguyên |
  | 29 | **Lạng Sơn** | Giữ nguyên |
  | 3  | **Thái Nguyên** | Bắc Kạn + Thái Nguyên |
  | 1  | **Tuyên Quang** | Tuyên Quang + Hà Giang |
  | 2  | **Lào Cai** | Lào Cai + Yên Bái |
  | 4  | **Phú Thọ** | Vĩnh Phúc + Phú Thọ + Hòa Bình |
  | 26 | **Lai Châu** | Giữ nguyên |
  | 27 | **Điện Biên** | Giữ nguyên |
  | 28 | **Sơn La** | Giữ nguyên |

  **Vùng Bắc Trung Bộ & Duyên hải miền Trung (mật độ TRUNG BÌNH):**
  | STT | Tên tỉnh mới | Tỉnh sáp nhập |
  |-----|-------------|---------------|
  | 31 | **Thanh Hóa** | Giữ nguyên |
  | 32 | **Nghệ An** | Giữ nguyên |
  | 33 | **Hà Tĩnh** | Giữ nguyên |
  | 9  | **Quảng Trị** | Quảng Bình + Quảng Trị |
  | 25 | **Huế** | Giữ nguyên (Thừa Thiên Huế → TP. Huế) |
  | 10 | **Đà Nẵng** | Quảng Nam + Đà Nẵng |
  | 11 | **Quảng Ngãi** | Kon Tum + Quảng Ngãi |

  **Vùng Tây Nguyên & Nam Trung Bộ (mật độ THẤP-TRUNG BÌNH):**
  | STT | Tên tỉnh mới | Tỉnh sáp nhập |
  |-----|-------------|---------------|
  | 12 | **Gia Lai** | Gia Lai + Bình Định |
  | 15 | **Đắk Lắk** | Đắk Lắk + Phú Yên |
  | 13 | **Khánh Hòa** | Ninh Thuận + Khánh Hòa |
  | 14 | **Lâm Đồng** | Lâm Đồng + Đắk Nông + Bình Thuận |

  **Vùng Đông Nam Bộ (mật độ RẤT CAO):**
  | STT | Tên tỉnh mới | Tỉnh sáp nhập |
  |-----|-------------|---------------|
  | 16 | **TP. Hồ Chí Minh** | Bà Rịa-Vũng Tàu + Bình Dương + TP. HCM |
  | 17 | **Đồng Nai** | Đồng Nai + Bình Phước |
  | 18 | **Tây Ninh** | Tây Ninh + Long An |

  **Vùng Đồng bằng sông Cửu Long (mật độ TRUNG BÌNH-CAO):**
  | STT | Tên tỉnh mới | Tỉnh sáp nhập |
  |-----|-------------|---------------|
  | 21 | **Đồng Tháp** | Tiền Giang + Đồng Tháp |
  | 20 | **Vĩnh Long** | Bến Tre + Vĩnh Long + Trà Vinh |
  | 19 | **Cần Thơ** | Cần Thơ + Sóc Trăng + Hậu Giang |
  | 23 | **An Giang** | An Giang + Kiên Giang |
  | 22 | **Cà Mau** | Bạc Liêu + Cà Mau |

### Phân loại vùng địa lý cho rule-based generation:
```python
REGION_DENSITY_PROFILE = {
    "dong_bang_song_hong": {
        # Hà Nội, Hải Phòng (+ Hải Dương), Hưng Yên (+ Thái Bình),
        # Ninh Bình (+ Hà Nam + Nam Định), Bắc Ninh (+ Bắc Giang), Quảng Ninh
        "provinces": ["Hà Nội", "Hải Phòng", "Hưng Yên", "Ninh Bình", "Bắc Ninh", "Quảng Ninh"],
        "pop_density_range": (800, 4500),   # người/km²
        "elevation_range": (0, 50),
        "urban_ratio": 0.60,
        "missing_data_rate": 0.05,
    },
    "mien_nui_phia_bac": {
        # Cao Bằng, Lạng Sơn, Thái Nguyên (+ Bắc Kạn), Tuyên Quang (+ Hà Giang),
        # Lào Cai (+ Yên Bái), Phú Thọ (+ Vĩnh Phúc + Hòa Bình), Lai Châu, Điện Biên, Sơn La
        "provinces": ["Cao Bằng", "Lạng Sơn", "Thái Nguyên", "Tuyên Quang",
                      "Lào Cai", "Phú Thọ", "Lai Châu", "Điện Biên", "Sơn La"],
        "pop_density_range": (20, 150),     # người/km²
        "elevation_range": (300, 3143),     # Fansipan 3143m
        "urban_ratio": 0.05,
        "missing_data_rate": 0.45,          # MNAR: vùng núi khó tiếp cận
    },
    "bac_trung_bo_duyen_hai": {
        # Thanh Hóa, Nghệ An, Hà Tĩnh, Quảng Trị (+ Quảng Bình),
        # Huế, Đà Nẵng (+ Quảng Nam), Quảng Ngãi (+ Kon Tum)
        "provinces": ["Thanh Hóa", "Nghệ An", "Hà Tĩnh", "Quảng Trị",
                      "Huế", "Đà Nẵng", "Quảng Ngãi"],
        "pop_density_range": (80, 1500),
        "elevation_range": (0, 1500),
        "urban_ratio": 0.35,
        "missing_data_rate": 0.18,
    },
    "tay_nguyen_nam_trung_bo": {
        # Gia Lai (+ Bình Định), Đắk Lắk (+ Phú Yên),
        # Khánh Hòa (+ Ninh Thuận), Lâm Đồng (+ Đắk Nông + Bình Thuận)
        "provinces": ["Gia Lai", "Đắk Lắk", "Khánh Hòa", "Lâm Đồng"],
        "pop_density_range": (50, 400),
        "elevation_range": (400, 2400),
        "urban_ratio": 0.18,
        "missing_data_rate": 0.28,
    },
    "dong_nam_bo": {
        # TP. Hồ Chí Minh (+ Bà Rịa-VT + Bình Dương),
        # Đồng Nai (+ Bình Phước), Tây Ninh (+ Long An)
        "provinces": ["TP. Hồ Chí Minh", "Đồng Nai", "Tây Ninh"],
        "pop_density_range": (500, 14000),  # TPHCM nội thành ~14000
        "elevation_range": (0, 200),
        "urban_ratio": 0.72,
        "missing_data_rate": 0.03,
    },
    "dong_bang_song_cuu_long": {
        # Đồng Tháp (+ Tiền Giang), Vĩnh Long (+ Bến Tre + Trà Vinh),
        # Cần Thơ (+ Sóc Trăng + Hậu Giang), An Giang (+ Kiên Giang), Cà Mau (+ Bạc Liêu)
        "provinces": ["Đồng Tháp", "Vĩnh Long", "Cần Thơ", "An Giang", "Cà Mau"],
        "pop_density_range": (200, 1000),
        "elevation_range": (0, 12),         # ĐBSCL gần như phẳng hoàn toàn
        "urban_ratio": 0.30,
        "missing_data_rate": 0.10,
    },
}
```

---

## 3. SCHEMA DỮ LIỆU (FEATURES + TARGET)

### Features (8 cột):

| Cột | Kiểu | Mô tả | Ràng buộc |
|-----|------|-------|-----------|
| `province` | string | Tên tỉnh/thành phố | Một trong 34 tỉnh 2026 |
| `region` | string | Vùng địa lý | 6 vùng như trên |
| `elevation_m` | float | Độ cao (m) | Phải nhất quán với region |
| `distance_to_urban_km` | float | Km đến trung tâm đô thị gần nhất | [0.5, 150] |
| `land_use_type` | string | Loại sử dụng đất | "residential", "agricultural", "forest", "mixed" |
| `road_density_km_per_km2` | float | Km đường/km² diện tích ô lưới | [0.0, 25.0] |
| `ndvi` | float | Chỉ số thực vật (xanh=cao, đô thị=thấp) | [-0.1, 0.9] |
| `annual_rainfall_mm` | float | Lượng mưa trung bình năm | [700, 3500] |
| `nearest_market_km` | float | Km đến chợ/trung tâm thương mại | [0.1, 80] |
| `admin_level` | int | Cấp hành chính ô: 1=phường đô thị, 2=thị trấn, 3=xã nông thôn, 4=xã miền núi | [1, 4] |

### Target (1 cột):
| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `population_density` | float | Người/km² — giá trị thực (không có NaN) |

### Cột phụ cho missing data simulation:
| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `has_survey_data` | bool | True = ô này được điều tra, False = cần predict |
| `missing_mechanism` | string | "MCAR", "MAR", hoặc "MNAR" |

---

## 4. RULE-BASED CONSTRAINTS (BẮT BUỘC ÁP DỤNG)

### Rule 1: Địa lý → Dân số (correlation chặt)
```
elevation cao → population_density thấp
  elevation > 1000m → density < 100 người/km² (gần như chắc chắn)
  elevation > 2000m → density < 40 người/km²

distance_to_urban nhỏ → population_density cao
  distance < 5km + region = đô thị → density > 1000

road_density cao → population_density cao (tương quan dương mạnh)
ndvi cao → population_density thấp (nhiều rừng = ít người)
land_use = "residential" → density cao nhất
land_use = "forest" → density thấp nhất
```

### Rule 2: Ràng buộc density theo tỉnh/thành (chính xác 2026)
```
# Đô thị lớn — nội thành mật độ cực cao:
TP. Hồ Chí Minh (nội thành Q1-Q3):  density [5000, 14000]
Hà Nội (nội thành Hoàn Kiếm-Đống Đa): density [3000, 8000]
Đà Nẵng (trung tâm):                density [1500, 4500]
Cần Thơ (trung tâm):                density [800, 2800]
Huế (đô thị):                       density [700, 2200]

# Tỉnh trung du — mật độ trung bình:
Thanh Hóa:    density [180, 600]
Nghệ An:      density [100, 400]
Phú Thọ:      density [200, 700]   # có Vĩnh Phúc công nghiệp
Bắc Ninh:     density [400, 1500]  # khu công nghiệp Samsung
Quảng Ninh:   density [200, 800]   # Hạ Long đô thị + vùng mỏ

# Tỉnh miền núi — thưa dân:
Sơn La:       density [40, 100]
Điện Biên:    density [30, 90]
Lai Châu:     density [25, 70]    # thưa nhì sau Tuyên Quang
Tuyên Quang:  density [35, 100]   # đã nhập Hà Giang — thưa nhất
Cao Bằng:     density [40, 110]
Lạng Sơn:     density [50, 130]

# Tây Nguyên:
Đắk Lắk:     density [60, 250]
Gia Lai:      density [55, 200]
Lâm Đồng:    density [80, 300]    # Đà Lạt là trung tâm du lịch

# ĐBSCL:
Đồng Tháp:   density [250, 700]
Vĩnh Long:   density [300, 800]
An Giang:    density [280, 750]   # An Giang có dân số đông ĐBSCL
Cà Mau:      density [150, 400]   # vùng rừng ngập mặn thưa hơn
```

### Rule 3: MNAR Pattern (quan trọng cho NCKH)
```
Cơ chế khuyết Missing Not At Random:
- Ô lưới có elevation > 800m VÀ distance_to_urban > 30km
  → xác suất KHÔNG có survey data = 60-80%
  (vùng núi khó tiếp cận → thực địa ít đến → data thưa)
- Ô lưới đô thị (land_use="residential", distance < 10km)
  → xác suất KHÔNG có survey data = 2-8%

Cơ chế khuyết MAR:
- Xã có admin_level=4 (xã miền núi)
  → missing rate cao hơn, nhưng có thể predict từ elevation + ndvi
  
Cơ chế khuyết MCAR:
- ~5% ngẫu nhiên (lỗi thu thập, mất file, v.v.)
```

### Rule 4: Noise thực tế
```
population_density = f(features) × (1 + noise)
  noise ~ Normal(0, 0.15) — 15% noise
  Thêm outlier nhỏ: ~2% ô lưới có density gấp 2-3x dự đoán
  (khu công nghiệp, khu tái định cư đột ngột xuất hiện)
```

### Rule 5: Nhất quán nội bộ
```
Nếu land_use = "forest" → ndvi > 0.5, road_density < 2
Nếu land_use = "residential" → ndvi < 0.3, road_density > 5
Nếu elevation > 1500m → annual_rainfall > 1800 (núi cao mưa nhiều)
Nếu region = ĐBSCL → elevation < 15, ndvi [0.3, 0.6]
```

---

## 5. KÍCH THƯỚC VÀ FORMAT OUTPUT

```python
N_SAMPLES = 2000          # tổng số ô lưới (grid cells)
RANDOM_SEED = 42          # cố định để reproducible
TEST_SIZE = 0.2           # 400 ô dùng để test (có label thật)
SURVEY_COVERAGE = 0.65    # 65% ô có survey data (has_survey_data=True)

# Phân bổ mẫu theo vùng (tỷ lệ xấp xỉ thực tế):
REGION_SAMPLE_RATIO = {
    "dong_bang_song_hong":      0.20,
    "mien_nui_phia_bac":        0.18,
    "bac_trung_bo_duyen_hai":   0.17,
    "tay_nguyen_nam_trung_bo":  0.12,
    "dong_nam_bo":              0.20,
    "dong_bang_song_cuu_long":  0.13,
}
```

Output files:
- `data/raw/vietnam_grid_synthetic.csv`  — dataset chính (grid cells)
- `data/raw/province_population.csv`     — dân số cấp tỉnh (ground truth)
- `data/raw/commune_population.csv`      — dân số cấp xã/phường (validation)

Encoding: UTF-8 | Separator: comma | Float: 2 decimal places | No index

---

## 5B. DỮ LIỆU DÂN SỐ CẤP TỈNH (GROUND TRUTH)

> **Mục đích:** Đây là ground truth để validate mô hình — sau khi model dự đoán density
> tại từng ô lưới, tổng hợp lên cấp tỉnh và so sánh với số này.

### Bước 1: Thu thập dân số cấp tỉnh

Script phải **hardcode** bảng dân số ước tính 2026 cho 34 tỉnh/thành phố.
Nguồn tham khảo khi AI sinh: GSO (Tổng cục Thống kê), Wikipedia, WorldPop.

**Bảng dân số 34 tỉnh/thành 2026 (ước tính sau sáp nhập — AI tự tra và hardcode):**

```python
# Dân số ước tính 2026 (nghìn người) — AI cần tìm và điền số thực
PROVINCE_POPULATION_2026 = {
    # === ĐBSH & Đông Bắc Bộ ===
    "Hà Nội":       9_000,   # ~9 triệu (2024) + tăng trưởng
    "Hải Phòng":    3_900,   # Hải Phòng ~2.2tr + Hải Dương ~1.9tr
    "Hưng Yên":     3_700,   # Hưng Yên ~1.35tr + Thái Bình ~1.9tr (dùng tên Hưng Yên)
    "Ninh Bình":    3_200,   # Hà Nam ~0.9tr + Ninh Bình ~1tr + Nam Định ~1.9tr
    "Bắc Ninh":     2_900,   # Bắc Ninh ~1.5tr + Bắc Giang ~1.8tr
    "Quảng Ninh":   1_400,   # Giữ nguyên
    # === Miền núi phía Bắc ===
    "Cao Bằng":       560,   # Giữ nguyên
    "Lạng Sơn":       800,   # Giữ nguyên
    "Thái Nguyên":  1_650,   # Thái Nguyên ~1.35tr + Bắc Kạn ~0.32tr
    "Tuyên Quang":  1_550,   # Tuyên Quang ~0.8tr + Hà Giang ~0.9tr
    "Lào Cai":      1_600,   # Lào Cai ~0.8tr + Yên Bái ~0.85tr
    "Phú Thọ":      3_200,   # Phú Thọ ~1.5tr + Vĩnh Phúc ~1.2tr + Hòa Bình ~0.95tr
    "Lai Châu":       500,   # Giữ nguyên — thưa nhất
    "Điện Biên":      650,   # Giữ nguyên
    "Sơn La":       1_350,   # Giữ nguyên
    # === Bắc Trung Bộ & Duyên hải ===
    "Thanh Hóa":    3_700,   # Giữ nguyên — đông thứ 3 cả nước
    "Nghệ An":      3_400,   # Giữ nguyên — đông thứ 4
    "Hà Tĩnh":      1_300,   # Giữ nguyên
    "Quảng Trị":    1_700,   # Quảng Bình ~0.9tr + Quảng Trị ~0.65tr (dùng tên Quảng Trị)
    "Huế":          1_150,   # Giữ nguyên (Thừa Thiên Huế)
    "Đà Nẵng":      2_500,   # Đà Nẵng ~1.2tr + Quảng Nam ~1.5tr
    "Quảng Ngãi":   1_800,   # Quảng Ngãi ~1.3tr + Kon Tum ~0.6tr
    # === Tây Nguyên & Nam Trung Bộ ===
    "Gia Lai":      2_700,   # Gia Lai ~1.6tr + Bình Định ~1.9tr (dùng tên Gia Lai)
    "Đắk Lắk":     2_700,   # Đắk Lắk ~2tr + Phú Yên ~1tr
    "Khánh Hòa":   1_600,   # Khánh Hòa ~1.25tr + Ninh Thuận ~0.65tr
    "Lâm Đồng":    2_800,   # Lâm Đồng ~1.35tr + Đắk Nông ~0.75tr + Bình Thuận ~1.3tr
    # === Đông Nam Bộ ===
    "TP. Hồ Chí Minh": 13_000,  # HCM ~9.5tr + Bình Dương ~2.7tr + Bà Rịa-VT ~1.2tr
    "Đồng Nai":     4_200,   # Đồng Nai ~3.3tr + Bình Phước ~1.1tr
    "Tây Ninh":     2_500,   # Tây Ninh ~1.2tr + Long An ~1.7tr
    # === ĐBSCL ===
    "Đồng Tháp":   3_100,   # Tiền Giang ~1.8tr + Đồng Tháp ~1.7tr
    "Vĩnh Long":   2_900,   # Bến Tre ~1.3tr + Vĩnh Long ~1.1tr + Trà Vinh ~1.1tr
    "Cần Thơ":     3_600,   # Cần Thơ ~1.4tr + Sóc Trăng ~1.4tr + Hậu Giang ~0.8tr
    "An Giang":    3_800,   # An Giang ~2tr + Kiên Giang ~1.9tr
    "Cà Mau":      1_700,   # Cà Mau ~1.2tr + Bạc Liêu ~0.95tr
}
# Tổng: ~95-98 triệu người (phù hợp dân số VN 2026)
```

**Output file 1:** `data/raw/province_population.csv`

| Cột | Mô tả |
|-----|-------|
| `province` | Tên tỉnh/thành (34 dòng) |
| `region` | Vùng địa lý (6 vùng) |
| `population_total` | Dân số ước tính 2026 (người) |
| `area_km2` | Diện tích tỉnh (km²) — hardcode theo thực tế |
| `avg_density_provincial` | = population_total / area_km2 (người/km²) |
| `n_communes` | Số xã/phường ước tính của tỉnh (xem bảng bên dưới) |

---

## 5C. DỮ LIỆU DÂN SỐ CẤP XÃ/PHƯỜNG (VALIDATION LAYER)

> **Bối cảnh pháp lý:** Từ 01/07/2025, Việt Nam **bỏ hoàn toàn cấp huyện**.
> Đơn vị hành chính sau 01/07/2025 chỉ còn:
> **Tỉnh/Thành phố → Xã/Phường/Thị trấn**
> Toàn quốc hiện có khoảng **8.000–9.000 xã/phường** (sau sáp nhập xã).

### Số xã/phường ước tính theo tỉnh (hardcode):

```python
PROVINCE_N_COMMUNES = {
    # Đô thị lớn — nhiều phường
    "Hà Nội":            579,   # nhiều phường nội thành + xã ngoại thành
    "TP. Hồ Chí Minh":   321,
    "Hải Phòng":         223,
    "Đà Nẵng":           105,
    "Cần Thơ":           142,
    "Huế":               133,
    # Tỉnh đông dân
    "Thanh Hóa":         520,   # lớn nhất về số xã
    "Nghệ An":           460,
    "Gia Lai":           180,
    "Đắk Lắk":          200,
    "An Giang":          138,
    "Đồng Tháp":        130,
    "Lâm Đồng":         130,
    # Tỉnh trung bình
    "Phú Thọ":          250,
    "Bắc Ninh":         150,
    "Quảng Ninh":       170,
    "Thái Nguyên":      170,
    "Hưng Yên":         210,
    "Ninh Bình":        190,
    "Tây Ninh":         130,
    "Đồng Nai":         135,
    "Vĩnh Long":        107,
    "Khánh Hòa":        135,
    "Quảng Trị":        145,
    "Hà Tĩnh":          229,
    "Quảng Ngãi":       165,
    "Lào Cai":          152,
    "Tuyên Quang":      138,
    # Tỉnh miền núi thưa
    "Sơn La":           202,
    "Điện Biên":        129,
    "Lai Châu":          96,
    "Cao Bằng":         149,
    "Lạng Sơn":         200,
    "Cà Mau":           100,
}
```

### Bước 2: Giả lập phân bổ dân số xuống xã/phường

Với mỗi tỉnh, sinh ra N = `n_communes` xã/phường ảo theo các rule:

```
Rule phân bổ dân số xã/phường:

1. PHÂN LOẠI XÃ (commune_type):
   - "phuong"       : phường đô thị — chiếm 15-40% số xã tùy tỉnh
   - "thi_tran"     : thị trấn — chiếm 5-10%
   - "xa_nong_thon" : xã nông thôn — chiếm 40-60%
   - "xa_mien_nui"  : xã miền núi/vùng sâu — chiếm phần còn lại

2. DIỆN TÍCH XÃ (area_commune_km2):
   - Phường đô thị:   2 – 15 km²
   - Thị trấn:        10 – 40 km²
   - Xã nông thôn:    20 – 80 km²
   - Xã miền núi:     50 – 300 km²
   → Tổng diện tích các xã ≈ diện tích tỉnh (normalize nếu lệch)

3. PHÂN BỔ DÂN SỐ (log-normal distribution):
   - Phường đô thị:   15.000 – 80.000 người/phường
   - Thị trấn:        8.000 – 30.000 người
   - Xã nông thôn:    3.000 – 12.000 người
   - Xã miền núi:     500 – 4.000 người
   → Sau khi sinh, RESCALE để tổng = province_population_total
     (đây là constraint bắt buộc — tổng dân số xã phải khớp với tỉnh)

4. MẬT ĐỘ DÂN SỐ XÃ (commune_density):
   commune_density = commune_population / area_commune_km2
   → Đây là GROUND TRUTH để validate model sau khi dự đoán grid cells

5. THÊM MNAR vào commune layer:
   - Xã miền núi (xa_mien_nui): 40% không có survey data
   - Xã nông thôn: 15% không có survey data
   - Phường/thị trấn: 3% không có survey data
```

**Output file 2:** `data/raw/commune_population.csv`

| Cột | Kiểu | Mô tả |
|-----|------|-------|
| `commune_id` | string | Mã định danh: `{province_code}_{idx:04d}` |
| `commune_name` | string | Tên ảo: `"Xã {idx}"` / `"Phường {idx}"` |
| `commune_type` | string | `phuong` / `thi_tran` / `xa_nong_thon` / `xa_mien_nui` |
| `province` | string | Tên tỉnh |
| `region` | string | Vùng địa lý |
| `area_commune_km2` | float | Diện tích xã (km²) |
| `commune_population` | int | Dân số ước tính (người) |
| `commune_density` | float | Mật độ dân số (người/km²) — **GROUND TRUTH** |
| `has_survey_data` | bool | Có data điều tra thực địa không |
| `missing_mechanism` | string | `MNAR` / `MAR` / `MCAR` / `-` |

### Validation logic (ghi vào comment trong code):
```python
# Sau khi model VietPopNet dự đoán population_density tại từng grid cell:
# 1. Map mỗi grid cell → commune (theo vị trí địa lý hoặc province label)
# 2. Tính predicted_density trung bình/tổng theo từng commune
# 3. So sánh với commune_density trong file này
# 4. Validation metrics:
#    - MAE at commune level
#    - % communes có sai số < 20% (acceptable threshold cho NCKH)
#    - Kiểm tra bias theo commune_type: model có bias với xa_mien_nui không?
```

---

## 6. YÊU CẦU VỀ CODE

### Cấu trúc script:
```
generate_synthetic.py
├── PHẦN 1: Import + Constants
│   ├── PROVINCE_POPULATION_2026     (dân số 34 tỉnh)
│   ├── PROVINCE_N_COMMUNES          (số xã/phường per tỉnh)
│   ├── PROVINCE_AREA_KM2            (diện tích tỉnh)
│   └── REGION_DENSITY_PROFILE       (range theo vùng)
├── PHẦN 2: Helper functions
│   ├── generate_geo_features(region, province, n)
│   ├── compute_population_density(row)   → rule-based float
│   ├── apply_mnar_pattern(df)            → thêm has_survey_data
│   └── validate_constraints(df)         → assert các rule
├── PHẦN 3: Sinh grid cell dataset (2000 ô lưới)
│   ├── Sinh từng vùng theo ratio
│   ├── Apply noise + outliers
│   └── Export: data/raw/vietnam_grid_synthetic.csv
├── PHẦN 4: Sinh province-level data
│   └── Export: data/raw/province_population.csv
├── PHẦN 5: Sinh commune-level data (validation layer)
│   ├── Với mỗi tỉnh: sinh n_communes xã/phường
│   ├── Phân loại commune_type theo tỉ lệ vùng
│   ├── Sinh dân số log-normal → rescale về province total
│   ├── Tính commune_density = population / area
│   └── Export: data/raw/commune_population.csv
└── PHẦN 6: Summary + Sanity checks
    ├── In thống kê 3 file
    ├── Assert: sum(commune_pop) == province_pop mỗi tỉnh
    └── Quick plot (optional)
```

### Yêu cầu code quality:
- Comment tiếng Việt hoặc tiếng Anh đều được
- Mỗi rule quan trọng phải có comment giải thích
- Hàm `validate_constraints(df)` phải raise AssertionError nếu vi phạm rule
- Cuối script in ra summary như sau:

```
═══ SYNTHETIC DATA SUMMARY ═══
Total grid cells:  2000
Regions:           6
Provinces:         34

[FILE 1] vietnam_grid_synthetic.csv
  Density — Mean: [x] | Median: [x] | Min: [x] | Max: [x] người/km²
  Missing: MNAR [x]% | MAR [x]% | MCAR [x]% | Total [x]%
  Correlations — elevation: [r] | dist_urban: [r] | road_density: [r] | ndvi: [r]

[FILE 2] province_population.csv
  Tổng dân số (34 tỉnh): ~[x] triệu người
  Densest:  [tên] — [x] người/km²
  Sparsest: [tên] — [x] người/km²

[FILE 3] commune_population.csv
  Tổng xã/phường: [x]
  phường [x]% | thị trấn [x]% | xã nông thôn [x]% | xã miền núi [x]%
  Assert PASSED: sum(commune_pop) == province_pop cho cả 34 tỉnh ✅
  Missing (MNAR+MAR+MCAR): [x]%

All files saved to: data/raw/
```

---

## 7. VÍ DỤ DÒNG MẪU (để AI hiểu format mong đợi)

**[File 1] vietnam_grid_synthetic.csv** — grid cells:

| province | region | elevation_m | dist_urban_km | land_use | road_density | ndvi | rainfall_mm | market_km | admin_level | population_density | has_survey_data | missing_mechanism |
|----------|--------|-------------|---------------|----------|--------------|------|-------------|-----------|-------------|-------------------|-----------------|-------------------|
| Hà Nội | dong_bang_song_hong | 12.5 | 2.1 | residential | 18.3 | 0.12 | 1680 | 0.8 | 1 | 7823.4 | True | - |
| Tuyên Quang | mien_nui_phia_bac | 1523.0 | 45.2 | forest | 0.8 | 0.78 | 2840 | 38.5 | 4 | 42.1 | False | MNAR |
| TP. Hồ Chí Minh | dong_nam_bo | 8.3 | 1.5 | residential | 22.1 | 0.08 | 1950 | 0.5 | 1 | 11250.7 | True | - |
| Đắk Lắk | tay_nguyen_nam_trung_bo | 680.0 | 28.4 | agricultural | 3.2 | 0.55 | 1920 | 22.1 | 3 | 187.3 | True | - |
| Lai Châu | mien_nui_phia_bac | 1890.0 | 62.3 | forest | 0.4 | 0.82 | 2650 | 51.2 | 4 | 28.6 | False | MNAR |

**[File 2] province_population.csv** — ground truth cấp tỉnh (34 dòng):

| province | region | population_total | area_km2 | avg_density_provincial | n_communes |
|----------|--------|-----------------|----------|----------------------|------------|
| Hà Nội | dong_bang_song_hong | 9000000 | 3358.6 | 2680.1 | 579 |
| Tuyên Quang | mien_nui_phia_bac | 1550000 | 13000.0 | 119.2 | 138 |
| TP. Hồ Chí Minh | dong_nam_bo | 13000000 | 6162.0 | 2109.8 | 321 |

**[File 3] commune_population.csv** — validation ground truth cấp xã/phường (~8500 dòng):

| commune_id | commune_name | commune_type | province | region | area_commune_km2 | commune_population | commune_density | has_survey_data | missing_mechanism |
|------------|-------------|-------------|---------|--------|-----------------|-------------------|-----------------|-----------------|-------------------|
| HAN_0001 | Phường 1 | phuong | Hà Nội | dong_bang_song_hong | 4.2 | 45230 | 10769.0 | True | - |
| HAN_0312 | Xã Tản Lĩnh | xa_nong_thon | Hà Nội | dong_bang_song_hong | 35.6 | 8450 | 237.4 | True | - |
| TUQ_0087 | Xã Minh Tiến | xa_mien_nui | Tuyên Quang | mien_nui_phia_bac | 145.3 | 1820 | 12.5 | False | MNAR |



---

## 8. ĐẦU RA MONG ĐỢI

Ba file output (chạy: `python generate_synthetic.py`):
- `data/raw/vietnam_grid_synthetic.csv` — 2000 grid cells (input cho model)
- `data/raw/province_population.csv`    — 34 dòng (aggregate ground truth)
- `data/raw/commune_population.csv`     — ~8500 dòng (commune-level validation)

Dependencies: chỉ `numpy`, `pandas`, `matplotlib` (optional)
Không cần internet | Chạy trong ~10 giây | `RANDOM_SEED = 42`
**CRITICAL:** `sum(commune_pop per province) == province_total` → phải pass cả 34 tỉnh

---

Hãy viết script Python hoàn chỉnh sinh ra ĐỦ 3 file trên.

Ưu tiên theo thứ tự:
1. Tính đúng của PROVINCE_POPULATION + PROVINCE_N_COMMUNES (hardcode chính xác)
2. Assertion: sum(commune_population per province) == province_total cho cả 34 tỉnh
3. MNAR pattern nhất quán giữa cả 3 file
4. Rule-based constraints địa lý (elevation, density, ndvi)
═══════════════════════════════════════════════════════════════
```

---

## 📌 NOTES KHI DÙNG PROMPT NÀY

- **ChatGPT / Claude:** Paste thẳng, thường cho ra code tốt ngay lần đầu
- **Gemini:** Thêm câu "Chỉ trả lời bằng code Python, không giải thích dài" nếu muốn output gọn
- **Perplexity:** Dùng chế độ "Coding" nếu có, tắt web search để không bị phân tâm

### Sau khi nhận code từ AI:
1. Chạy thử: `python generate_synthetic.py`
2. Kiểm tra summary output — đặc biệt phần **correlation** và **missing rate per region**
3. Nếu elevation cao mà density vẫn cao → rule chưa đúng → yêu cầu AI fix
4. Lưu vào: `d:\Coding_prj\Deep Learning Getting Started\VietPopNet\data\raw\`

---

*Prompt sinh data v2.0 — VietPopNet | Cập nhật: 14/09/2026*
*Dùng cho: Claude / ChatGPT / Gemini / Perplexity*
*Output: 3 files — grid cells + province ground truth + commune validation*
