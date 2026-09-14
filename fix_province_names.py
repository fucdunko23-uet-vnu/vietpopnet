"""
Fix province names — FINAL VERSION using exact UTF-8 byte patterns.
Each garbled province name is matched by its exact UTF-8 byte sequence.
"""

import pandas as pd
import shutil

INPUT_PATH  = r"d:\Coding_prj\Final_prj_ML+DL\data\synthetic_commune_census_microcensus.csv"
OUTPUT_PATH = r"d:\Coding_prj\Final_prj_ML+DL\data\synthetic_commune_census_microcensus.csv"
BACKUP_PATH = r"d:\Coding_prj\Final_prj_ML+DL\data\synthetic_commune_census_microcensus_backup.csv"
LOG_PATH    = r"d:\Coding_prj\Final_prj_ML+DL\fix_province_names_log.txt"

log = open(LOG_PATH, "w", encoding="utf-8")
def p(msg=""): log.write(msg + "\n"); log.flush()

# ─────────────────────────────────────────────────────────────────────────────
# PROVINCE -> REGION (34 provinces 2026)
# ─────────────────────────────────────────────────────────────────────────────
PROVINCE_REGION = {
    "Hà Nội":"dong_bang_song_hong","Hải Phòng":"dong_bang_song_hong",
    "Hưng Yên":"dong_bang_song_hong","Ninh Bình":"dong_bang_song_hong",
    "Bắc Ninh":"dong_bang_song_hong","Quảng Ninh":"dong_bang_song_hong",
    "Cao Bằng":"mien_nui_phia_bac","Lạng Sơn":"mien_nui_phia_bac",
    "Thái Nguyên":"mien_nui_phia_bac","Tuyên Quang":"mien_nui_phia_bac",
    "Lào Cai":"mien_nui_phia_bac","Phú Thọ":"mien_nui_phia_bac",
    "Lai Châu":"mien_nui_phia_bac","Điện Biên":"mien_nui_phia_bac",
    "Sơn La":"mien_nui_phia_bac",
    "Thanh Hóa":"duyen_hai_mien_trung","Nghệ An":"duyen_hai_mien_trung",
    "Hà Tĩnh":"duyen_hai_mien_trung","Quảng Trị":"duyen_hai_mien_trung",
    "Huế":"duyen_hai_mien_trung","Đà Nẵng":"duyen_hai_mien_trung",
    "Quảng Ngãi":"duyen_hai_mien_trung",
    "Gia Lai":"tay_nguyen","Đắk Lắk":"tay_nguyen",
    "Khánh Hòa":"tay_nguyen","Lâm Đồng":"tay_nguyen",
    "TP. Hồ Chí Minh":"dong_nam_bo","Đồng Nai":"dong_nam_bo","Tây Ninh":"dong_nam_bo",
    "Đồng Tháp":"dong_bang_song_cuu_long","Vĩnh Long":"dong_bang_song_cuu_long",
    "Cần Thơ":"dong_bang_song_cuu_long","An Giang":"dong_bang_song_cuu_long",
    "Cà Mau":"dong_bang_song_cuu_long",
}

# ─────────────────────────────────────────────────────────────────────────────
# EXACT BYTE -> CORRECT PROVINCE NAME
# Keys are byte strings decoded as exact UTF-8 sequences found in the CSV.
# Read from Step 2 byte analysis in previous run.
#
# Mapping logic (per synthetic_data_gen_prompt.md 2025-2026 reform):
#   Bình Phước  -> Đồng Nai        (Bình Phước sáp nhập vào Đồng Nai)
#   Bình Thuận  -> Lâm Đồng        (Bình Thuận sáp nhập vào Lâm Đồng)
#   Bình Định   -> Gia Lai         (Bình Định sáp nhập vào Gia Lai)
#   Bắc Ninh    -> Bắc Ninh        (giữ nguyên)
#   Hưng Yên    -> Hưng Yên        (giữ nguyên, đã nhập Thái Bình)
#   Hà Nội      -> Hà Nội          (giữ nguyên)
#   Hải Phòng   -> Hải Phòng       (giữ nguyên)
#   Khánh Hòa   -> Khánh Hòa       (giữ nguyên)
#   Nam Định    -> Ninh Bình       (sáp nhập vào Ninh Bình)
#   Ninh Bình   -> Ninh Bình       (giữ nguyên)
#   Ninh Thuận  -> Khánh Hòa       (sáp nhập vào Khánh Hòa)
#   Phú Yên     -> Đắk Lắk         (sáp nhập vào Đắk Lắk)
#   Quảng Nam   -> Đà Nẵng         (sáp nhập vào Đà Nẵng)
#   Quảng Ngãi  -> Quảng Ngãi      (giữ nguyên)
#   Thừa Thiên-Huế -> Huế          (đổi tên)
#   Thực Ninh   -> Ninh Bình       (noise name)
#   Tây Ninh    -> Tây Ninh        (giữ nguyên)
#   Vĩnh Phúc   -> Phú Thọ         (sáp nhập vào Phú Thọ)
#   Đà Nẵng     -> Đà Nẵng         (giữ nguyên, đã nhập Quảng Nam)
#   Đồng (garbled) -> Đồng Nai     (most likely Đồng Nai)
# ─────────────────────────────────────────────────────────────────────────────

# Byte patterns from log analysis -> correct province
# Each entry: (bytes_pattern, correct_province)
BYTE_TO_PROVINCE = [
    # b'B\xc3\xacnh Ph\xc6\xb0\xe1\xbb\xa3c' = "Bình Phước" (garbled ợ->ược)
    (b'B\xc3\xacnh Ph\xc6\xb0\xe1\xbb\xa3c',   "Đồng Nai"),
    # b'B\xc3\xacnh Thu\xe1\xba\xbdn' = "Bình Thuẽn" (garbled ậ->ẽ, n->n)
    (b'B\xc3\xacnh Thu\xe1\xba\xbdn',           "Lâm Đồng"),
    # b'B\xc3\xacnh \xc4\x90\xe1\xbb\xb5nh' = "Bình Đỵnh" (garbled ị->ỵ)
    (b'B\xc3\xacnh \xc4\x90\xe1\xbb\xb5nh',    "Gia Lai"),
    # b'B\xe1\xba\xafc Ninh' = "Bắc Ninh" correct
    (b'B\xe1\xba\xafc Ninh',                    "Bắc Ninh"),
    # b'H\xc3\xa0\xc2\xbb\xc2\xb0ng Y\xc3\xaan' = "Hà»°ng Yên" = "Hưng Yên" (double-encoded)
    (b'H\xc3\xa0\xc2\xbb\xc2\xb0ng Y\xc3\xaan',"Hưng Yên"),
    # b'H\xc3\xa0\xef\xbf\xbd\xc2\xb0 N\xe1\xbb\x99i' = "Hà?° Nội" = "Hà Nội" (replacement char)
    (b'H\xc3\xa0\xef\xbf\xbd\xc2\xb0 N\xe1\xbb\x99i', "Hà Nội"),
    # b'H\xe1\xba\xa3i Ph\xc3\xb2ng' = "Hải Phòng" correct
    (b'H\xe1\xba\xa3i Ph\xc3\xb2ng',            "Hải Phòng"),
    # b'Kh\xe1\xba\xbdnh H\xc3\xb2a' = "Khẽnh Hòa" (garbled á->ẽ) = "Khánh Hòa"
    (b'Kh\xe1\xba\xbdnh H\xc3\xb2a',            "Khánh Hòa"),
    # b'Nam \xc4\x90\xe1\xbb\xb5nh' = "Nam Đỵnh" = "Nam Định"
    (b'Nam \xc4\x90\xe1\xbb\xb5nh',             "Ninh Bình"),
    # b'Ninh B\xc3\xacnh' = "Ninh Bình" correct
    (b'Ninh B\xc3\xacnh',                       "Ninh Bình"),
    # b'Ninh Thu\xe1\xba\xbdn' = "Ninh Thuẽn" = "Ninh Thuận"
    (b'Ninh Thu\xe1\xba\xbdn',                  "Khánh Hòa"),
    # b'Ph\xe1\xbb\xa5 Y\xc3\xaan' = "Phụ Yên" = "Phú Yên" (garbled ú->ụ)
    (b'Ph\xe1\xbb\xa5 Y\xc3\xaan',              "Đắk Lắk"),
    # b'Qu\xc3\xa1\xc2\xba\xc2\xa3ng Nam' = "Quáº£ng Nam" (double-encoded) = "Quảng Nam"
    (b'Qu\xc3\xa1\xc2\xba\xc2\xa3ng Nam',       "Đà Nẵng"),
    # b'Qu\xc3\xa1\xc2\xba\xc2\xa3ng Ng\xc3\xa3i' = "Quáº£ng Ngãi" = "Quảng Ngãi"
    (b'Qu\xc3\xa1\xc2\xba\xc2\xa3ng Ng\xc3\xa3i', "Quảng Ngãi"),
    # b'Th\xc3\xa1\xc2\xbb\xc2\xa5a Thi\xc3\xaan-Hu\xc3\xa1\xc2\xba\xc2\xbf' = "Thừa Thiên-Huế"
    (b'Th\xc3\xa1\xc2\xbb\xc2\xa5a Thi\xc3\xaan-Hu\xc3\xa1\xc2\xba\xc2\xbf', "Huế"),
    # b'T\xc3\xa2y Ninh' = "Tây Ninh" correct
    (b'T\xc3\xa2y Ninh',                        "Tây Ninh"),
    # b'V\xc3\x84\xc2\xa9nh Ph\xc3\xbac' = "VÄ©nh Phúc" (double-encoded) = "Vĩnh Phúc"
    (b'V\xc3\x84\xc2\xa9nh Ph\xc3\xbac',        "Phú Thọ"),
    # b'\xc4\x90\xc3\xa0 N\xe1\xba\xb5ng' = "Đà Nẵng" correct
    (b'\xc4\x90\xc3\xa0 N\xe1\xba\xb5ng',       "Đà Nẵng"),
    # b'\xc4\xa2\xe1\xba\xafng' = "Ģắng" = garbled "Đắng"? -> Đồng Nai
    (b'\xc4\xa2\xe1\xba\xafng',                 "Đồng Nai"),
    # b'\xc4\xb0\xe0\xba\xbcng' = "İຼng" = garbled -> Đồng Nai
    (b'\xc4\xb0\xe0\xba\xbcng',                 "Đồng Nai"),
    # Also handle "Thực Ninh" (already fixed to this in prev run)
    # and "Hải Dương" which maps to Hải Phòng
]

# Build lookup dict: bytes -> correct name
BYTES_MAP = {pattern: name for pattern, name in BYTE_TO_PROVINCE}

p("=" * 60)
p("VietPopNet -- Fix Province Names (Byte-Exact Matching)")
p("=" * 60)

df = pd.read_csv(INPUT_PATH, encoding="utf-8-sig")
p(f"\n[Step 1] Loaded {len(df):,} rows, {len(df.columns)} columns")

p("\n[Step 2] Applying byte-exact province mapping...")

fixed = 0
unmapped = []

def fix_province(name):
    global fixed
    # Try exact byte match
    name_bytes = name.encode("utf-8")
    if name_bytes in BYTES_MAP:
        fixed += 1
        return BYTES_MAP[name_bytes]
    # Already correct
    if name in PROVINCE_REGION:
        return name
    unmapped.append((repr(name), name.encode("utf-8")))
    return name

df["province"] = df["province"].apply(fix_province)

if unmapped:
    p(f"\n  WARNING: {len(set(str(x) for x in unmapped))} name(s) NOT mapped:")
    seen = set()
    for rep, byt in unmapped:
        if rep not in seen:
            p(f"    {rep}  bytes: {byt}")
            seen.add(rep)
else:
    p(f"  SUCCESS: Fixed {fixed} values, all names mapped!")

p(f"  Total fixes applied: {fixed}")

# Re-assign region
df["region"] = df["province"].apply(lambda x: PROVINCE_REGION.get(x, "UNKNOWN"))

# ─────────────────────────────────────────────────────────────────────────────
# BACKUP + SAVE
# ─────────────────────────────────────────────────────────────────────────────
shutil.copy2(INPUT_PATH, BACKUP_PATH)
df.to_csv(OUTPUT_PATH, index=False, encoding="utf-8-sig")
p(f"\n[Step 3] Backup: {BACKUP_PATH}")
p(f"[Step 4] Saved: {OUTPUT_PATH}")

# ─────────────────────────────────────────────────────────────────────────────
# SUMMARY
# ─────────────────────────────────────────────────────────────────────────────
p("\n" + "=" * 60 + "\nSUMMARY\n" + "=" * 60)
unique_after = sorted(df["province"].unique())
p(f"Total rows: {len(df):,}")
p(f"Unique provinces: {len(unique_after)}")
p("\nProvince list:")

invalid_list = []
for prov in unique_after:
    region = PROVINCE_REGION.get(prov, "UNKNOWN")
    cnt = int((df["province"] == prov).sum())
    flag = "" if prov in PROVINCE_REGION else "  <-- INVALID"
    p(f"  {prov:30s} | {region:35s} | rows: {cnt}{flag}")
    if prov not in PROVINCE_REGION:
        invalid_list.append(prov)

p("\nRegion distribution:")
for region, cnt in df["region"].value_counts().items():
    p(f"  {region}: {cnt}")

if invalid_list:
    p(f"\nWARNING: {len(invalid_list)} invalid remain:")
    for x in invalid_list:
        p(f"  {repr(x)}  bytes: {x.encode('utf-8')}")
else:
    p(f"\nSUCCESS: All {len(unique_after)} province names are valid 2026 names!")

p("\nDone.")
log.close()

# ASCII terminal output
print("Done. Log: " + LOG_PATH)
if invalid_list:
    print(f"WARNING: {len(invalid_list)} invalid remain -- check log!")
else:
    print(f"SUCCESS: All {len(unique_after)} provinces valid!")
