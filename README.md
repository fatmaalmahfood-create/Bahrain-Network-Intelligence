# Bahrain-Network-Intelligence

AI-powered system for predicting mobile network performance and optimizing telecom tower placement using geospatial data and machine learning.

---

## 1. Problem Statement & Motivation

The problem is not that Bahrain has bad internet, it’s that improvements don’t reach everyone equally, so mobile network performance in Bahrain varies significantly across locations, meaning that some regions benefit from strong, reliable connectivity while others experience weaker service. 

To enhance connectivity across the entire country, telecom operators often build new towers. However, this process is typically done manually, relying on the analysis of coverage areas, speed test data, and performance trends, which can be time-consuming and resource-intensive.


**The Goal:** Build an end-to-end AI system that:
1. **Predicts** download speed, upload speed, and latency for any location in Bahrain.
2. **Ranks** the top areas most in need of new tower infrastructure.
3. **Simulates** the network improvement a new tower would produce at a given location.
4. **Answers** natural-language questions about the network via an AI chat agent.


---

## 2. Datasets & Sources

### 2.1. Speedtest Dataset

**Source:** **[Ookla Speedtest Intelligence](https://github.com/teamookla/ookla-open-data)**

**Description:**  
Contains speed test results collected from locations across Bahrain.

| Attribute | Value |
|---|---|
| **Rows** | 40,025 quarterly grid-block measurements |
| **Date Range** | January 2020 – July 2025 |
| **Geography** | 4 governorates · 145 cities · 273 distinct areas |
| **Key Columns** | `avg_d_mbps`, `avg_u_mbps`, `avg_lat_ms`, `tests`, `latitude`, `longitude`, `area`, `city`, `region`, `typeOfArea`, `digital_elevation_model` |
| **Target Stats** | DL: 0–2,012 Mbps (mean 157.6) · UL: 0–190 Mbps (mean 22.5) · Latency: 3–1,673 ms (mean 24.6) |


---
---

## 2.2. Tower Dataset

**Source:** **[CellMapper](https://www.cellmapper.net/map?MCC=426&MNC=4&type=NR&latitude=26.198864906176013&longitude=50.48261799705054&zoom=12.7186391737821&showTowers=true&showIcons=true&showTowerLabels=true&clusterEnabled=true&tilesEnabled=true&showOrphans=false&showNoFrequencyOnly=false&showFrequencyOnly=false&showBandwidthOnly=false&DateFilterType=Last&showHex=false&showVerifiedOnly=false&showUnverifiedOnly=false&showLTECAOnly=false&showENDCOnly=false&showBand=0&showSectorColours=true&mapType=roadmap&darkMode=false&imperialUnits=false)**  

**Description:**  
Contains information about telecom towers and includes the following columns:

| Attribute | Value |
|---|---|
| **Rows** | 2,854 tower records |
| **Active Towers** | 1,966 (`Visible = True`) |
| **Operators** | Batelco · Zain · STC |
| **RAT Type** | LTE (4G) and LTE-A (4G+/5G) |
| **Key Columns** | `Latitude`, `Longitude`, `RAT Type`, `RAT SubType`, `Operator`, `Tower Type`, `Visible` |
 
**Preprocessing applied to towers:**
- Filtered to `Visible = True` only — removed inactive/decommissioned sites.
- Used GPS coordinates to spatially join with speed test grid blocks.


---

## 3. Model Architecture & Training Pipeline

### 3.1 Feature Engineering (21 Features)
 
All features were computed from the two raw datasets and grouped into four categories:
 
| Category | Features |
|---|---|
| **Location** | latitude, longitude, digital_elevation_model, region (encoded), city (encoded), typeOfArea (encoded) |
| **Tower Infrastructure** | nearest_tower_dist_km, towers_within_1km, towers_within_2km, towers_within_5km, tower_density_ratio, distance_x_density |
| **Temporal** | year_trend (year − min_year), quarter, demand_growth_pct |
| **Historical Speed** | area_median_d, area_median_u, area_median_lat, area_rolling_avg_d, area_test_count |
 
**Key engineering decisions:**
- `nearest_tower_dist_km` — computed using the **Haversine formula** (great-circle distance) between each speed test grid centroid and every active tower. O(n×m) operation parallelized with NumPy broadcasting.
- `tower_density_ratio` = `towers_within_1km / (towers_within_5km + 1)` — captures relative local density vs. wider area.
- `distance_x_density` = `nearest_tower_dist_km × tower_density_ratio` — interaction feature.
- `area_median_*` values — computed from **quarterly aggregates** (grouped by `area` and `year_quarter`), not raw rows, to match the training data granularity.
- Label encoding for `region`, `city`, `typeOfArea` — consistent encoders saved to `assets/encoders.json`.


### 3.2 Data Split
 
```
70% Train / 15% Validation / 15% Test
Random state = 42
```
 
No temporal split was used — data was shuffled randomly. This is a known limitation (see §5 — data leakage risk).
 
### 3.3 Models Trained
 
Four models were trained and compared on all three targets simultaneously:
 
| Model | Notes |
|---|---|
| Linear Regression | Baseline. StandardScaler applied to X. |
| Decision Tree | `max_depth=10`, `min_samples_leaf=4` |
| Random Forest | `n_estimators=100`, `max_depth=10` |
| **Gradient Boosting** ✓ | See hyperparameters below |


### 3.4 Winning Model: Gradient Boosting Hyperparameters
 
Hyperparameters were tuned iteratively by monitoring validation MAPE:
 
```python
GradientBoostingRegressor(
    n_estimators     = 200,
    learning_rate    = 0.03,
    max_depth        = 5,
    subsample        = 0.8,
    min_samples_leaf = 4,
    random_state     = 42
)
```
 
Three separate models were trained — one per target (download, upload, latency).
 
---


## 4. Evaluation Results & Baseline Comparison
 
### 4.1 Quantitative Metrics — Download Speed Prediction
 
| Model | MAPE  | MAE (Mbps) ↓ | R² |
|---|---|---|---|
| Linear Regression | 62.6% | 89.3 | 0.18 |
| Decision Tree | 62.3% | 81.7 | 0.22 |
| Random Forest | 57.9% | 71.2 | 0.38 |
| Custom Neural Network | ~58.0% | ~70.5 | 0.36 |
| **Gradient Boosting** | **41.7%** | **44.2** | **0.55** |
 
> **Why MAPE?** Mean Absolute Percentage Error is the most interpretable metric for network operators. Saying "the model is 41.7% off on average" is far clearer than MAE in absolute Mbps, especially when speeds range from 4 Mbps to 2,012 Mbps across the dataset.
 
### 4.2 Key Finding: Year Explains ~50% of Variance
 
A single-feature model using only `year_trend` achieves R² ≈ 0.50. This means **network-wide infrastructure upgrades** (5G rollout, backhaul improvements) are the dominant driver of speed — more important than local tower count in most areas.
 
Speed improvements:
```
2020: 38 Mbps → 2021: 62 → 2022: 124 → 2023: 195 → 2024: 248 → 2025: 287 Mbps
```
 
### 4.3 Qualitative Error Analysis
 
**Where the model succeeds (low error):**
 
| Area | Predicted | Actual | Error | Why |
|---|---|---|---|---|
| A'ali | 162 Mbps | 178 Mbps | 9% | Dense training data, stable tower environment |
| Manama | 295 Mbps | 312 Mbps | 5% | 16K+ rows from Capital Governorate |
| Seef District | 241 Mbps | 258 Mbps | 7% | Many nearby towers, consistent history |
 
**Where the model struggles (high error):**
 
| Area | Predicted | Actual | Error | Root Cause |
|---|---|---|---|---|
| Hawar Island | 48 Mbps | 18 Mbps | 167% | Only ~5 training rows — model guesses from geography |
| High-speed 5G zone | 310 Mbps | 445 Mbps | 30% under | Extreme speeds underrepresented in training data |
| New 2026 tower area | 72 Mbps | 190 Mbps | 62% under | Tower built after dataset snapshot — not in features |
 
### 4.4 Feature Importance (Gradient Boosting)
 
Top 5 most important features for download speed prediction:
1. `year_trend` — 0.31 importance score
2. `area_median_d` — 0.24
3. `nearest_tower_dist_km` — 0.12
4. `towers_within_5km` — 0.09
5. `quarter` — 0.07
---
 
## 5. Limitations & Future Work
 
### Data Imbalance
The dataset is heavily skewed toward Northern Governorate (~40% of records) and Capital Governorate (~35%). Southern Governorate accounts for only ~7% of records. The model has learned Manama very well but generalizes poorly to rural and island areas — exactly the areas most in need of infrastructure planning.
 
**Mitigation considered:** Stratified sampling by governorate during train/test split. Not implemented in current version.

### Static Infrastructure Snapshot
The tower dataset reflects the state of Bahrain's cell network as of the last CellMapper update. Towers built or upgraded in 2026 are not captured. Recommendations may therefore underestimate coverage in rapidly developing areas.

---

## 6. How to Run the Project
 
### Prerequisites
- Python 3.10+
- Node.js (optional, for JS utilities)
### Step 1: Clone the Repository
### Step 2: Create Virtual Environment & Install Dependencies
### Step 3: Prepare the Data
### Step 4: Train the Models
### Step 5: Launch the Dashboard
### Step 6: Run the Evaluation Notebook


---
 
## 7. Repo Structure
 
```
bahrain-network-intelligence/
│
├── assets/
│   ├── encoders.json             
│   └── figures/
│
├── data/
│   ├── speedtests.csv
|   |── bahrain_towers.csv      
│   └── processed/
│       └── features_engineered.csv
|       └── speeds_clean.csv
│
├── notebooks/
│   ├── EDA.ipynb
|   ├── evaluation.ipynb        
│   ├── preprocessing.ipynb   
│   ├── featureEngineering.ipynb        
│   └── train.ipynb       
│
├── scripts/
│   ├── app.py                   
│   ├── predict.py                
│   ├── recommend.py                             
│
├── outputs/
│   └── models/
│       ├── gb_models.pkl         
│       ├── feature_cols2.json     
│       |── best_model_name.txt
|       |── X_test.npy
|       |── X_train.npy
│       |── X_val.npy
|       |── y_train.npy
|       |── y_test.npy
│       └── y_val.npy
│
├── presentations/
│   ├── Bahrain_Network_IntelligenceLATEST.pptx
│   |── Demo.mp4

│
├── README.md
├── requirements.txt
├── .gitignore
└── LICENSE
```
 
---

 
## 8. Screenshots & Demo
 
### Demo Video
**[Watch the demo on YouTube](https://youtu.be/cfNElJsryyY)**
 

### Dashboard Screenshots
 
| Feature | Description |
|---|---|
| Speed Distribution Map | Click any Bahrain grid cell to see historical speed data |
| Tower Map | 1,966 active towers color-coded by operator (Batelco/Zain/STC) |
| AI Speed Prediction | Click map → auto-fills 17 input fields → predicts DL/UL/latency |
| Recommendations Table | Top 5 tower locations with priority scores |
 
*(See `assets/figures/` for full-resolution screenshots)*
 
---
 
## 9. License & Acknowledgments
 
### Acknowledgments
 
| Resource | Role |
|---|---|
| **Ookla Speedtest Intelligence** | Speed test measurements dataset (open data program) |
| **CellMapper** | Crowdsourced cell tower database |
| **Streamlit** | Dashboard framework |
| **scikit-learn** | ML model training and evaluation |
| **PyTorch** | Custom neural network architecture |
| **General Assembly Bahrain** | Capstone program and instructional support |
 
### Author
**Fatema S Mahfoodh**  
Data Science Capstone · General Assembly Bahrain · 2026
