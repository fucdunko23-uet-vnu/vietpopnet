# 🗺️ MASTER PLAN — VietPopNet: Neural Population Estimation for Spatial Grid Cells

> **Cập nhật:** 13/09/2026 | **Chế độ học:** Theory-Light, Practice-Heavy (KHÁC Socratic)
> **Mục tiêu kép:** Học DL Foundation nhanh + Build production portfolio project cho NCKH + FAANG

---

## 🎯 TÊN PROJECT

**`VietPopNet`** — *A bottom-up deep learning pipeline for estimating population density in unsampled Vietnamese spatial grid cells using geographic features and micro-survey data.*

**Đường dẫn project:** `d:\Coding_prj\Deep Learning Getting Started\VietPopNet\`

---

## 💡 4 MỤC TIÊU ĐỒNG THỜI

| # | Tiêu chí | Cách project đáp ứng |
|---|----------|---------------------|
| 1 | **DL Foundation** | Mỗi chương DL → implement thẳng vào 1 module của project, không học trừu tượng |
| 2 | **FAANG Portfolio** | End-to-end pipeline, PyTorch, SHAP, spatial domain, production mindset |
| 3 | **Ôn Applied ML** | FE notebook (MCAR/MAR/MNAR), baseline notebook (RF/XGBoost/MLP sklearn) |
| 4 | **NCKH Lab** | Đây chính là bài toán: dự đoán dân số ô lưới từ vi điều tra + đặc trưng địa lý |

---

## 🏗️ KIẾN TRÚC TỔNG THỂ

```
[Dữ liệu vi điều tra (micro-survey)]    [Đặc trưng địa lý (geo-features)]
         ↓                                         ↓
[Feature Engineering Pipeline]  ←  MCAR/MAR/MNAR, imputation, encoding
         ↓
[Baseline Models]  ←  LinearReg → RF → XGBoost → SHAP explainability
         ↓
[VietPopNet MLP v0] ←  DL Ch.1: Neural Network từ scratch → PyTorch
         ↓
[Loss + Optimizer]  ←  DL Ch.2: MSE/MAE + Adam vs SGD
         ↓
[Regularized MLP]   ←  DL Ch.3: Dropout + L2 + BatchNorm (data nhỏ = dễ overfit)
         ↓
[Full Training Loop] ← DL Ch.6: DataLoader, checkpoint, early stopping, scheduler
         ↓
[Evaluation + SHAP]  ← MAE, RMSE, R² + spatial feature importance map
         ↓
[Visualization]      ← Grid heatmap dân số → artifact đẹp cho báo cáo NCKH + GitHub
```

---

## 🗂️ CẤU TRÚC THƯ MỤC

```
d:\Coding_prj\Deep Learning Getting Started\VietPopNet\
├── data/
│   ├── raw/                    # Synthetic vi điều tra data (mô phỏng cấu trúc thật)
│   ├── processed/              # Sau feature engineering pipeline
│   └── geo_features/           # elevation, land_use, road_density, ndvi, rainfall...
├── notebooks/
│   ├── 00_baseline_check.ipynb          # Kiểm tra nền GD + Backprop
│   ├── 01_eda_feature_engineering.ipynb # Ôn Applied ML FE + MCAR/MAR/MNAR
│   ├── 02_baseline_models.ipynb         # RF + XGBoost + SHAP (ôn ML Algorithms)
│   ├── 03_neural_network.ipynb          # DL Ch.1 — MLP từ scratch
│   ├── 04_loss_optimizers.ipynb         # DL Ch.2 — MSE + Adam
│   ├── 05_regularization.ipynb          # DL Ch.3 — Dropout + L2 + BatchNorm
│   └── 06_full_pipeline.ipynb           # DL Ch.6 — Production training loop
├── src/
│   ├── dataset.py              # PyTorch Dataset + DataLoader
│   ├── model.py                # VietPopNet MLP architecture
│   ├── train.py                # Training loop + early stopping + scheduler
│   └── evaluate.py             # Metrics + SHAP plots
├── experiments/                # Logged runs (loss curves, hyperparams)
├── reports/                    # Output figures + NCKH report assets
├── README.md                   # Portfolio README với architecture diagram
└── requirements.txt
```

---

## 📅 TIMELINE — STEP BY STEP (SENIOR TECH LEAD STYLE)

> **Nguyên tắc ưu tiên:** Không block nhau — mỗi phase cho ra 1 deliverable dùng được ngay.
> **Tốc độ target:** 1 phase = 1-3 buổi học (tùy độ phức tạp và bạn nhớ được bao nhiêu từ trước)

---

### ⚙️ PHASE 0 — FOUNDATION CHECK & PROJECT SETUP (1 buổi)

**Mục tiêu:** Xác nhận điểm xuất phát chính xác + tạo scaffold project

#### Step 0.1 — Kiểm tra nền (KHÔNG BỎ QUA)
- [ ] Trả lời 2 câu: "Tại sao trừ gradient?" + "Chain rule qua sigmoid?"
- [ ] Kết quả → quyết định: lướt nhanh hay dạy lại nhanh GD + Backprop

#### Step 0.2 — Setup environment
```bash
# Tạo virtual environment
python -m venv venv
venv\Scripts\activate

# Install dependencies
pip install torch torchvision numpy pandas matplotlib seaborn scikit-learn xgboost shap jupyter
pip freeze > requirements.txt
```

#### Step 0.3 — Tạo scaffold project
- [ ] Tạo toàn bộ thư mục theo cấu trúc trên
- [ ] Generate synthetic data (script `data/generate_synthetic.py`)
- [ ] Commit lên GitHub với README skeleton

**✅ Deliverable Phase 0:** Project chạy được, data có sẵn, env sạch

---

### 🧠 PHASE 1 — DL CHAPTER 1: NEURAL NETWORK CƠ BẢN (2-3 buổi)

**Mục tiêu học:** Perceptron, feedforward, activation functions (ReLU/Sigmoid/Tanh/Softmax)

#### Step 1.1 — Lý thuyết (Bước 0: kiểm tra nền)
- Bạn đã biết MLP từ khóa ML Algorithms ch.4 (sklearn) → kiểm tra nhanh: "giải thích feedforward pass trong 3 câu?"
- Nếu nhớ → lướt nhanh. Nếu không → dạy 5 phút rồi code ngay

#### Step 1.2 — Code mẫu PyTorch (Bước 2)
```python
# VietPopNet v0.1 — MLP 2 layers, PyTorch thuần
import torch
import torch.nn as nn

class VietPopNet(nn.Module):
    def __init__(self, input_dim):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1)  # Regression: predict population density
        )
    def forward(self, x):
        return self.network(x)
```

#### Step 1.3 — Mini-project: `03_neural_network.ipynb` (Bước 3)
- [ ] Build VietPopNet v0.1 với synthetic data
- [ ] Forward pass thủ công + kiểm tra output shape
- [ ] Thay ReLU → Sigmoid → Tanh, quan sát gradient behavior
- [ ] **Submit code + kết quả → review → chuyển Phase 2**

**✅ Deliverable Phase 1:** Notebook 03 hoàn chỉnh — MLP chạy được, hiểu từng activation

---

### 📉 PHASE 2 — DL CHAPTER 2: LOSS FUNCTIONS + OPTIMIZERS (1-2 buổi)

**Mục tiêu học:** MSE, MAE (regression); Cross-Entropy (classification); SGD vs Adam

#### Step 2.1 — Lý thuyết tối giản
- MSE = trung bình bình phương sai số → phạt nặng outlier
- MAE = trung bình sai số tuyệt đối → robust với outlier
- Adam = SGD + momentum + adaptive learning rate → converge nhanh hơn

#### Step 2.2 — Kết nối với Applied ML Algorithms
- Bạn đã dùng MSE trong Linear Regression → kiểm tra: "MSE và MAE khác nhau thế nào về ảnh hưởng với outlier?" → nếu nhớ → lướt, không nhớ → dạy 3 phút

#### Step 2.3 — Mini-project: `04_loss_optimizers.ipynb`
- [ ] Train VietPopNet v0.1 với MSE Loss + Adam
- [ ] Vẽ loss curve (train vs val)
- [ ] Chạy lại với SGD → so sánh tốc độ hội tụ
- [ ] **So sánh với Random Forest baseline** từ notebook 02

**✅ Deliverable Phase 2:** Biết chọn loss + optimizer phù hợp, có loss curve đẹp cho README

---

### 🛡️ PHASE 3 — DL CHAPTER 3: REGULARIZATION (2 buổi)

**Mục tiêu học:** Dropout, L1/L2 weight decay, Batch Normalization

> [!IMPORTANT]
> Đây là phase quan trọng nhất với VietPopNet. Dữ liệu vi điều tra nhỏ = overfit rất dễ xảy ra. Regularization không phải option — là bắt buộc.

#### Step 3.1 — Tạo overfit scenario có chủ ý
- Dùng model lớn hơn cần thiết → quan sát train loss ↓ nhưng val loss ↑
- **Mục tiêu:** thấy tận mắt overfit trước khi fix

#### Step 3.2 — Fix lần lượt, từng kỹ thuật
- [ ] Thêm L2 weight decay vào Adam optimizer
- [ ] Thêm Dropout layer → quan sát val loss
- [ ] Thêm BatchNorm → quan sát training stability

#### Step 3.3 — Mini-project: `05_regularization.ipynb`
- [ ] Grid search Dropout rate {0.1, 0.3, 0.5} + L2 penalty {1e-4, 1e-3, 1e-2}
- [ ] Vẽ bảng kết quả val MAE theo từng combo
- [ ] Chọn best config → đây là `VietPopNet v1.0`

**✅ Deliverable Phase 3:** VietPopNet v1.0 — không overfit, hiểu lý do chọn từng hyperparameter

---

### 🔄 PHASE 4 — DL CHAPTER 6: FULL TRAINING PIPELINE (2-3 buổi)

**Mục tiêu học:** Dataset/DataLoader, train/eval loop chuẩn, checkpoint, early stopping, LR scheduler

#### Step 4.1 — PyTorch Dataset class
```python
class PopulationDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.FloatTensor(X)
        self.y = torch.FloatTensor(y).unsqueeze(1)
    def __len__(self): return len(self.X)
    def __getitem__(self, idx): return self.X[idx], self.y[idx]
```

#### Step 4.2 — Training loop chuẩn production
- [ ] Tách rõ `train_one_epoch()` + `evaluate()`
- [ ] Early stopping với patience=10
- [ ] `torch.save(model.state_dict(), "best_model.pt")` khi val loss cải thiện
- [ ] ReduceLROnPlateau scheduler

#### Step 4.3 — Mini-project: `06_full_pipeline.ipynb` + `src/train.py`
- [ ] Refactor toàn bộ code vào `src/` → notebook chỉ gọi functions
- [ ] Log experiment: epoch, train_loss, val_loss, val_MAE, val_RMSE, val_R²
- [ ] Vẽ learning curves đẹp cho README

**✅ Deliverable Phase 4:** `src/train.py` chạy được từ command line — production-ready

---

### 🔁 PHASE 5 — APPLIED ML REVIEW INTEGRATION (song song với Phase 1-4)

> **Chạy song song** — mỗi khi bắt đầu Phase mới, làm notebook này trước

#### Step 5.1 — `01_eda_feature_engineering.ipynb`
- [ ] Load synthetic data, EDA cơ bản (distribution, correlation)
- [ ] Tạo missing data có chủ ý (simulate MCAR/MAR/MNAR)
- [ ] Chẩn đoán cơ chế khuyết → apply đúng chiến thuật imputation
- [ ] Feature encoding: land_use (categorical) → one-hot; elevation, rainfall → StandardScaler
- [ ] Feature selection: correlation heatmap + domain knowledge

#### Step 5.2 — `02_baseline_models.ipynb`
- [ ] Linear Regression (baseline đơn giản nhất)
- [ ] Random Forest → feature importance
- [ ] XGBoost → SHAP waterfall plot cho 1 grid cell
- [ ] Bảng so sánh: MAE/RMSE/R² của tất cả models → **đây là context để đánh giá VietPopNet**

**✅ Deliverable Phase 5:** 2 notebooks hoàn chỉnh — Applied ML knowledge được consolidate và documented

---

### 🏆 PHASE 6 — PORTFOLIO POLISH & NCKH REPORT (1-2 buổi)

#### Step 6.1 — README.md production quality
```markdown
# VietPopNet — Neural Population Estimation for Spatial Grid Cells

## Problem
Vietnam's 2029 census faces challenges in hard-to-reach areas...

## Architecture
[Mermaid diagram]

## Results
| Model           | MAE   | RMSE  | R²    |
|----------------|-------|-------|-------|
| Linear Reg     | x.xx  | x.xx  | x.xx  |
| Random Forest  | x.xx  | x.xx  | x.xx  |
| XGBoost        | x.xx  | x.xx  | x.xx  |
| VietPopNet MLP | x.xx  | x.xx  | x.xx  |

## Key Findings
- SHAP analysis: top 3 geographic features driving population density
- Regularization ablation: Dropout 0.3 + L2 1e-3 → best val MAE
```

#### Step 6.2 — NCKH Report Assets
- [ ] Grid heatmap: predicted vs actual population density
- [ ] SHAP beeswarm plot: feature importance across grid cells
- [ ] Residual analysis: mô hình dự đoán sai ở vùng nào (insight cho domain)

#### Step 6.3 — GitHub Push
- [ ] Clean commit history (squash dev commits)
- [ ] Pin repo trên GitHub profile
- [ ] Gắn vào CV + LinkedIn

**✅ Deliverable Phase 6:** Public GitHub repo + NCKH report assets — ready to show HR

---

## 🤖 QUY TẮC DẠY HỌC CHO PHIÊN NÀY

> **QUAN TRỌNG:** Đây là chế độ **"THEORY-LIGHT, PRACTICE-HEAVY"** — KHÁC hoàn toàn với chế độ Socratic ở project GenAI Roadmap.

### Chế độ này là gì?

| | Socratic (GenAI project) | Theory-Light Practice-Heavy (DL project này) |
|-|-|-|
| **Tốc độ** | Chậm, đào sâu nhiều vòng | Nhanh, ưu tiên code ngay |
| **Khi bạn không biết** | Hỏi ngược lại nhiều vòng | Giải thích ngắn (≤5 câu) → code mẫu ngay |
| **Sau bài tập** | Nhiều vòng feedback | 1 lần review → chuyển ngay chương tiếp |
| **Hỏi ngược** | Bắt buộc, nhiều | Tối thiểu — chỉ kiểm tra nền đầu chương |

### Quy trình áp dụng với mỗi chương DL:

**BƯỚC 0 — KIỂM TRA NỀN (1 câu, không hơn):**
- Hỏi thẳng 1 câu ngắn về khái niệm liên quan
- Nhớ rõ → lướt 1-2 câu liên kết → vào thẳng code
- Không nhớ → giải thích ≤5 câu → code mẫu ngay → bài tập

**BƯỚC 1 — LÝ THUYẾT TỐI GIẢN:**
- 3-5 câu, ngôn ngữ đời thường, tối đa 1 công thức cốt lõi
- Chỉ: "cái này là gì, dùng để làm gì, vì sao cần"
- KHÔNG suy diễn dài, KHÔNG edge cases, KHÔNG Socratic hỏi ngược

**BƯỚC 2 — CODE MẪU:**
- 10-20 dòng PyTorch, comment từng dòng quan trọng
- Chạy được ngay

**BƯỚC 3 — MINI-PROJECT:**
- Gắn trực tiếp vào VietPopNet (không phải ví dụ đồ chơi)
- Bạn tự làm → gửi → review 1 lần → chuyển ngay
- Nếu bài tập lộ ra bạn chưa hiểu dù nói "biết rồi" → quay lại dạy kỹ, không cả nể

### Liên kết với Socratic rules (d:\Coding_prj\socratic_tutor_rules.md):

Chỉ áp dụng phân loại Nhóm A/Nhóm B khi **bạn tự chủ động hỏi về code cụ thể** ngoài luồng chính của khoá học này. Trong luồng chính (dạy từng chương theo Phase 0→6 trên), luôn dùng chế độ Theory-Light Practice-Heavy.

---

## 🤖 UNIVERSAL SYSTEM PROMPT — CHO MỌI AI ASSISTANT KHÁC

> Copy đoạn này paste vào đầu conversation với bất kỳ AI nào (Claude, ChatGPT, Gemini...) để chúng hiểu đúng context và giữ bạn đi đúng hướng.

---

```
═══ CONTEXT: VIETPOPNET PROJECT ═══

Tôi là sinh viên năm 3 Kỹ sư CNTT tại UET Hà Nội, đang build project:

TÊN: VietPopNet — Neural Population Estimation for Spatial Grid Cells
MÔ TẢ: Pipeline deep learning bottom-up ước tính mật độ dân số tại các ô lưới không gian Việt Nam chưa được điều tra, dựa trên dữ liệu vi điều tra mẫu và đặc trưng địa lý.
ĐỊA CHỈ: d:\Coding_prj\Deep Learning Getting Started\VietPopNet\

MỤC TIÊU KÉP:
1. Học Deep Learning Foundation (Neural Network, Loss, Regularization, Training Loop)
2. Xây portfolio cho vị trí AI/ML Intern tại VinAI, VietinBank, MB Bank, CMC Telecom (~3-4/2027)
3. Ôn Applied ML: Feature Engineering (MCAR/MAR/MNAR), RF, XGBoost, SHAP
4. Phục vụ đề tài NCKH: điều tra dân số theo ô lưới, phương pháp bottom-up, nội/ngoại suy sang vùng sâu vùng xa

KIẾN TRÚC PIPELINE:
Synthetic micro-survey data + Geo-features → Feature Engineering (FE notebook) → Baseline models (RF/XGBoost) → VietPopNet MLP (PyTorch) → Full Training Loop + Regularization → Evaluation (MAE/RMSE/R² + SHAP) → Grid heatmap visualization

CẤU TRÚC NOTEBOOK:
- 00_baseline_check.ipynb — Kiểm tra nền GD + Backprop
- 01_eda_feature_engineering.ipynb — EDA + MCAR/MAR/MNAR + imputation
- 02_baseline_models.ipynb — LinearReg + RF + XGBoost + SHAP
- 03_neural_network.ipynb — DL Ch.1: MLP từ scratch
- 04_loss_optimizers.ipynb — DL Ch.2: MSE + Adam vs SGD
- 05_regularization.ipynb — DL Ch.3: Dropout + L2 + BatchNorm
- 06_full_pipeline.ipynb — DL Ch.6: DataLoader + train loop + checkpoint

TIMELINE HIỆN TẠI:
Phase 0 (Setup) → Phase 1 (Neural Network) → Phase 2 (Loss/Optimizer) → Phase 3 (Regularization) → Phase 4 (Full Pipeline) → Phase 5 (Applied ML Review, chạy song song) → Phase 6 (Portfolio Polish)

CHẾ ĐỘ DẠY BẮT BUỘC — "THEORY-LIGHT, PRACTICE-HEAVY":
- KHÔNG dạy Socratic nhiều vòng hỏi-đáp
- Với mỗi chương: kiểm tra nền 1 câu → lý thuyết ≤5 câu → code mẫu 10-20 dòng → mini-project gắn với VietPopNet
- Khi tôi hỏi về 1 khái niệm: giải thích ngắn, đưa code mẫu liên quan đến VietPopNet ngay
- Review bài tập: 1 lần, chỉ ra đúng/sai/thiếu → chuyển ngay (không lặp nhiều vòng)
- Nếu bài tập lộ ra tôi chưa hiểu dù nói "biết rồi" → dạy lại kỹ hơn trước khi cho qua

NỀN ĐÃ CÓ (đã xác nhận):
- Applied ML Foundations: 9-bước pipeline, EDA, Titanic project
- Feature Engineering: MCAR/MAR/MNAR, Group Imputation, Indicator flags
- ML Algorithms: Linear Regression, SVM, MLP (sklearn), Random Forest, Boosting
- Math: Linear Algebra, Chain Rule, Gradient Descent (đã tự code), Backprop (đã tính tay)
- Tech: Python, NumPy, pandas, scikit-learn, XGBoost, SHAP

MỤC TIÊU PORTFOLIO:
Repo GitHub public: VietPopNet — README có architecture diagram + evaluation table + SHAP plots + grid heatmap. Stack: PyTorch | scikit-learn | XGBoost | SHAP | pandas | NumPy. Apply intern 3-4/2027.

═══ NHIỆM VỤ CỦA BẠN ═══
Giữ tôi đi đúng theo pipeline VietPopNet đã mô tả. Khi tôi hỏi về bất kỳ khái niệm DL nào, hãy map nó vào đúng notebook/phase tương ứng. Không để tôi đi lạc sang hướng không liên quan.
```

---

## 📊 DATA STRATEGY

### Giai đoạn 1 — Synthetic Data (học DL, không cần data thật)

**Features được sinh ra:**
| Feature | Mô tả | Kiểu |
|---------|-------|------|
| `elevation_m` | Độ cao so với mực nước biển | Float |
| `distance_to_urban_km` | Khoảng cách tới trung tâm đô thị | Float |
| `land_use_type` | Loại sử dụng đất (residential/agri/forest) | Categorical (3 class) |
| `road_density_km_per_km2` | Mật độ đường giao thông | Float |
| `ndvi` | Chỉ số xanh thực vật (vệ tinh) | Float [-1, 1] |
| `annual_rainfall_mm` | Lượng mưa trung bình năm | Float |
| `nearest_market_km` | Khoảng cách chợ/trung tâm thương mại | Float |
| `admin_level` | Cấp hành chính (xã/huyện/tỉnh) | Ordinal |

**Target:** `population_density` (người/km²) — sinh từ combination có noise + MNAR pattern (vùng núi thường có data thưa hơn → simulate đúng thực tế NCKH)

### Giai đoạn 2 — Real Data (khi lab có)
Thay file CSV vào `data/raw/` → pipeline FE + model không đổi. Đây là **production mindset** — HR thích điều này.

---

## ✅ PORTFOLIO CHECKLIST (Deadline: trước 02/2027)

- [ ] GitHub repo `vietpopnet` — public, có README đầy đủ
- [ ] Architecture diagram bằng Mermaid trong README
- [ ] Evaluation table: 4 models × 3 metrics
- [ ] SHAP beeswarm plot — feature importance đẹp
- [ ] Grid heatmap visualization — predicted vs actual density
- [ ] `requirements.txt` + clean code với docstrings
- [ ] Gắn vào CV: "VietPopNet — PyTorch spatial regression pipeline for census estimation"
- [ ] Pin repo trên GitHub profile

---

## ⚠️ OPEN QUESTIONS (cần bạn trả lời trước khi bắt đầu Phase 0)

> [!IMPORTANT]
> **Câu 2 CHƯA được trả lời:** Ngay lúc này, Gradient Descent và Backpropagation — bạn nhớ rõ, mơ hồ, hay không nhớ? Trả lời thật để tôi quyết định điểm xuất phát Phase 0.

> [!NOTE]
> **Lab NCKH có data mẫu chưa?** Nếu có dù chỉ 1 file CSV nhỏ → tôi sẽ thiết kế synthetic data sát cấu trúc thật hơn nhiều.

> [!NOTE]
> **Deadline báo cáo NCKH?** Để cân bằng tốc độ học DL vs deliver kết quả NCKH đúng hạn.

---

*Master Plan v1.0 — 13/09/2026 | VietPopNet | DL Foundation × FAANG Portfolio × NCKH*
