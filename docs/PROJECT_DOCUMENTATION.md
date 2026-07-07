# Personalized Nutrition AI from Blood Test Reports
## Complete Project Documentation

**Project Type:** Final Year / Mini Project — B.Tech / B.E. Computer Science  
**Technology Stack:** Python · Streamlit · SQLite · Scikit-learn · OCR  
**Version:** 1.0 | **Year:** 2024–25

---

## Table of Contents

1. Abstract
2. Introduction
3. Problem Statement
4. Objectives
5. System Architecture
6. Data Flow Diagram
7. ER Diagram
8. Database Schema
9. Module Descriptions
10. Algorithms Used
11. Testing
12. Future Enhancements
13. Conclusion
14. Installation Guide

---

## 1. Abstract

The *Personalized Nutrition AI from Blood Test Reports* is an intelligent web application that leverages Artificial Intelligence, Machine Learning, and Optical Character Recognition to analyze blood test reports and generate personalized, actionable dietary recommendations. Users can upload reports in PDF, image, or CSV format. The system automatically extracts eight critical blood parameters, classifies each against clinical normal ranges, and produces food-specific recommendations tailored to each user's deficiencies or excesses. A Random Forest machine learning model predicts overall health risk (Low / Medium / High), and downloadable PDF reports are generated for medical reference. The system is built entirely on open-source technologies including Python, Streamlit, SQLite, Scikit-learn, Pytesseract, and FPDF2.

---

## 2. Introduction

Blood tests are among the most informative diagnostic tools available. They reveal whether essential nutrients are at healthy levels, whether metabolic parameters such as glucose and cholesterol are in range, and whether the body is deficient or overdosed on vitamins and minerals. However, interpreting blood reports requires domain expertise, and most patients receive reports without guidance on what to eat or avoid.

This project addresses that gap by creating an AI-powered assistant that:
- Reads blood reports automatically using OCR
- Interprets every parameter against established clinical ranges
- Recommends specific foods and lifestyle changes
- Predicts health risk using a trained machine learning model
- Generates a professional downloadable PDF report

The application is accessible via any web browser through a Streamlit interface, requiring no technical knowledge from the end user.

---

## 3. Problem Statement

Millions of patients receive blood test reports annually but lack actionable nutritional guidance. Healthcare access is unequal, and scheduling a dietitian consultation is expensive and time-consuming. Furthermore, blood reports contain dense medical jargon that is difficult for laypeople to interpret. There is a clear need for an automated, intelligent, and accessible system that bridges the gap between diagnostic results and personalized nutrition advice.

**Existing problems addressed:**
- Patients do not understand their blood report values
- No immediate dietary guidance is provided with reports
- Nutritionist consultations are costly
- No longitudinal tracking of blood health across multiple reports

---

## 4. Objectives

1. Develop a secure, multi-user web application for blood report analysis
2. Implement OCR-based extraction of blood parameter values from uploaded files
3. Compare extracted values against standard clinical reference ranges
4. Generate personalized food and lifestyle recommendations for each deficiency
5. Train a machine learning model to predict overall health risk level
6. Provide a visual dashboard with charts and health score metrics
7. Enable downloadable PDF report generation
8. Store all data securely in a structured SQLite database

---

## 5. System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    PRESENTATION LAYER                       │
│              Streamlit Web Application (app.py)             │
│   Login │ Upload │ Manual Entry │ Dashboard │ Report        │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                    BUSINESS LOGIC LAYER                     │
│                                                             │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  ocr_reader  │  │  nutrition   │  │    ml_model      │  │
│  │   .py        │  │  _engine.py  │  │     .py          │  │
│  │              │  │              │  │                  │  │
│  │ PDF/Image/   │  │ Rule-based   │  │ Random Forest    │  │
│  │ CSV parsing  │  │ analysis     │  │ Risk Predictor   │  │
│  │ + OCR        │  │ + food DB    │  │                  │  │
│  └──────────────┘  └──────────────┘  └──────────────────┘  │
│                                                             │
│  ┌──────────────────────────────────────────────────────┐   │
│  │              report_generator.py                     │   │
│  │          FPDF2-based PDF Report Builder              │   │
│  └──────────────────────────────────────────────────────┘   │
└────────────────────────┬────────────────────────────────────┘
                         │
┌────────────────────────▼────────────────────────────────────┐
│                     DATA LAYER                              │
│              database.py  ←→  SQLite (nutrition_ai.db)      │
│   Users │ BloodReports │ Recommendations │ PredictionHistory│
└─────────────────────────────────────────────────────────────┘
```

**Three-tier architecture:**
- **Presentation Layer:** Streamlit handles all UI rendering and user interaction
- **Business Logic Layer:** Python modules handle OCR, analysis, ML prediction, and report generation
- **Data Layer:** SQLite stores all user data, reports, recommendations, and prediction history

---

## 6. Data Flow Diagram

```
User Uploads File / Enters Values
            │
            ▼
    ┌───────────────┐
    │  ocr_reader   │  (PDF/Image → Pytesseract OCR → Regex extraction)
    │  or Manual    │  (CSV → Pandas parsing)
    └───────┬───────┘
            │  dict: {parameter: value}
            ▼
    ┌───────────────┐
    │  database.py  │  save_blood_report() → BloodReports table
    └───────┬───────┘
            │  report_id
            ▼
    ┌───────────────────┐
    │ nutrition_engine  │  classify_parameter() → Low / Normal / High
    │                   │  get_recommendations() → food lists
    └───────┬───────────┘
            │  analysis_results list
            ▼
    ┌───────────────┐
    │  database.py  │  save_recommendations() → Recommendations table
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │   ml_model    │  predict_risk() → Random Forest → risk_level
    └───────┬───────┘
            │
            ▼
    ┌───────────────┐
    │  database.py  │  save_prediction() → PredictionHistory table
    └───────┬───────┘
            │
            ▼
    ┌─────────────────┐
    │ report_generator│  generate_pdf_report() → .pdf file
    └─────────────────┘
            │
            ▼
    Streamlit Dashboard (charts, cards, download button)
```

---

## 7. ER Diagram

```
┌──────────────────┐         ┌──────────────────────┐
│      Users       │         │     BloodReports      │
├──────────────────┤         ├──────────────────────┤
│ PK id            │◄───1:N──│ PK id                │
│    username (UQ) │         │ FK user_id           │
│    password      │         │    report_date        │
│    full_name     │         │    hemoglobin         │
│    age           │         │    glucose            │
│    gender        │         │    cholesterol        │
│    email         │         │    vitamin_d          │
│    created_at    │         │    iron               │
└──────────────────┘         │    calcium            │
                             │    vitamin_b12        │
                             │    triglycerides      │
                             │    source_type        │
                             └──────────┬───────────┘
                                        │
                             1:N        │        1:N
                  ┌──────────┘                   └─────────┐
                  ▼                                         ▼
   ┌──────────────────────┐           ┌───────────────────────────┐
   │    Recommendations   │           │     PredictionHistory     │
   ├──────────────────────┤           ├───────────────────────────┤
   │ PK id                │           │ PK id                     │
   │ FK report_id         │           │ FK user_id                │
   │ FK user_id           │           │ FK report_id              │
   │    parameter         │           │    risk_level             │
   │    status            │           │    health_score           │
   │    recommendation    │           │    predicted_at           │
   │    created_at        │           │    model_version          │
   └──────────────────────┘           └───────────────────────────┘
```

---

## 8. Database Schema

### Table: Users
| Column | Type | Constraints | Description |
|--------|------|-------------|-------------|
| id | INTEGER | PK, AUTOINCREMENT | Unique user ID |
| username | TEXT | UNIQUE, NOT NULL | Login identifier |
| password | TEXT | NOT NULL | bcrypt hashed password |
| full_name | TEXT | | Patient's full name |
| age | INTEGER | | Patient age |
| gender | TEXT | | Male / Female / Other |
| email | TEXT | | Contact email |
| created_at | TEXT | DEFAULT now | Account creation timestamp |

### Table: BloodReports
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Report ID |
| user_id | INTEGER FK | References Users.id |
| report_date | TEXT | Timestamp of report upload |
| hemoglobin | REAL | Hemoglobin value (g/dL) |
| glucose | REAL | Blood glucose (mg/dL) |
| cholesterol | REAL | Total cholesterol (mg/dL) |
| vitamin_d | REAL | Vitamin D level (ng/mL) |
| iron | REAL | Serum iron (µg/dL) |
| calcium | REAL | Calcium (mg/dL) |
| vitamin_b12 | REAL | Vitamin B12 (pg/mL) |
| triglycerides | REAL | Triglycerides (mg/dL) |
| source_type | TEXT | 'manual', 'csv', or 'ocr' |

### Table: Recommendations
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Recommendation ID |
| report_id | INTEGER FK | References BloodReports.id |
| user_id | INTEGER FK | References Users.id |
| parameter | TEXT | Blood parameter name |
| status | TEXT | 'Low', 'Normal', or 'High' |
| recommendation | TEXT | JSON array of food suggestions |
| created_at | TEXT | Creation timestamp |

### Table: PredictionHistory
| Column | Type | Description |
|--------|------|-------------|
| id | INTEGER PK | Prediction ID |
| user_id | INTEGER FK | References Users.id |
| report_id | INTEGER FK | References BloodReports.id |
| risk_level | TEXT | 'Low', 'Medium', or 'High' |
| health_score | REAL | 0–100 calculated score |
| predicted_at | TEXT | Prediction timestamp |
| model_version | TEXT | ML model version tag |

---

## 9. Module Descriptions

### app.py — Main Application
Entry point for the Streamlit web application. Handles:
- Page routing (Login, Dashboard, Upload, Manual, Analysis, Prediction, History, Download)
- Session state management
- CSS theming and styling
- Integration of all backend modules

### database.py — Data Access Layer
Manages all SQLite interactions:
- `init_db()` — Creates all tables on first run
- `register_user()` / `login_user()` — Auth with bcrypt hashing
- `save_blood_report()` / `get_user_reports()` — CRUD for blood data
- `save_recommendations()` / `get_recommendations()` — Recommendation persistence
- `save_prediction()` / `get_prediction_history()` — ML result storage

### ocr_reader.py — File Parsing Module
Three parsing strategies:
- **Image files:** OpenCV preprocessing (grayscale, denoising, Otsu thresholding) → Pytesseract OCR → regex extraction
- **PDF files:** PyPDF2 text extraction → fallback to pdf2image + OCR
- **CSV files:** Pandas parsing supporting two formats (parameter/value columns, or parameters as column headers)

### nutrition_engine.py — Recommendation Engine
Rule-based AI component:
- `NORMAL_RANGES` dictionary maps each parameter to clinical min/max/unit
- `classify_parameter()` — Returns Low / Normal / High
- `NUTRITION_DB` — Comprehensive food recommendation database with foods-to-eat, foods-to-avoid, and health tips
- `analyze_blood_report()` — Master function producing full analysis list
- `calculate_health_score()` — Computes 0–100 score based on parameter normalcy ratio

### ml_model.py — Machine Learning Module
- `generate_training_data()` — Creates 3,000-sample synthetic dataset using realistic normal distributions with clinical reference ranges as labeling thresholds
- `train_model()` — Trains Random Forest (200 trees, max_depth=10) with StandardScaler and LabelEncoder; saves artifacts via joblib
- `predict_risk()` — Loads model and predicts risk + probability breakdown
- `get_feature_importance()` — Returns explainability DataFrame

### report_generator.py — PDF Report Builder
Uses fpdf2 to generate a styled A4 PDF containing:
- Patient information header
- Health score summary with risk level
- Color-coded blood parameter table
- Detailed nutrition recommendations per parameter
- Medical disclaimer footer

---

## 10. Algorithms Used

### A. Regex-Based Parameter Extraction
- **Algorithm:** Named-group regular expressions with case-insensitive matching
- **Pattern:** `parameter_name[\s\S]{0,30}?(\d{1,3}\.?\d*)`
- **Time Complexity:** O(n) where n = text length
- **Purpose:** Extract numeric blood values from unstructured OCR text

### B. Rule-Based Classification
- **Algorithm:** Threshold comparison against pre-defined normal ranges
- **Logic:** `if value < min → Low | if value > max → High | else → Normal`
- **Time Complexity:** O(1) per parameter
- **Purpose:** Clinical classification of each blood parameter

### C. Random Forest Classifier
- **Algorithm:** Ensemble of 200 decision trees with majority voting
- **Features:** 8 blood parameters (StandardScaler normalized)
- **Labels:** Low / Medium / High (based on count of out-of-range parameters)
- **Training:** 80/20 train-test split, stratified sampling
- **Evaluation Metric:** Accuracy, Precision, Recall, F1-Score
- **Time Complexity:** O(n_trees × n_features × log(n_samples)) for training

### D. Health Score Formula
```
Health Score = (Count of Normal Parameters / Total Parameters) × 100
```
- Simple, interpretable, and deterministic
- Produces a 0–100 score independent of value magnitudes

### E. Image Preprocessing Pipeline (OpenCV)
1. Convert BGR → Grayscale
2. Fast Non-Local Means Denoising
3. Otsu's Binarization (adaptive thresholding)
4. Pass to Tesseract OCR with `--psm 6` (uniform block mode)

---

## 11. Testing

### Unit Tests

| Module | Function | Test Case | Expected Result |
|--------|----------|-----------|-----------------|
| nutrition_engine | classify_parameter('hemoglobin', 8.0) | Low value | 'Low' |
| nutrition_engine | classify_parameter('glucose', 85.0) | Normal value | 'Normal' |
| nutrition_engine | classify_parameter('cholesterol', 250) | High value | 'High' |
| nutrition_engine | calculate_health_score([...5 normal, 3 abnormal]) | 5/8 normal | 62.5 |
| ml_model | predict_risk({all normal values}) | Healthy profile | 'Low' risk |
| database | register_user('test', 'pass123', ...) | New user | (True, msg) |
| database | register_user('test', 'pass123', ...) again | Duplicate | (False, error) |
| ocr_reader | read_csv_file(sample_template) | Valid CSV | 8 parameters |

### Integration Tests

| Scenario | Steps | Expected Outcome |
|----------|-------|-----------------|
| Full pipeline (Manual) | Enter values → Analyse → View Analysis | Report saved, recs displayed |
| Full pipeline (CSV) | Upload CSV → Extract → Analyse | OCR values used for analysis |
| PDF download | Run analysis → Generate PDF | PDF downloadable with all data |
| Auth flow | Register → Logout → Login | Session persists correctly |
| History tracking | Submit 3 reports | All 3 visible in history tab |

### Sample Test Values

```python
NORMAL_TEST_CASE = {
    "hemoglobin": 14.0, "glucose": 88.0, "cholesterol": 175.0,
    "vitamin_d": 35.0, "iron": 100.0, "calcium": 9.5,
    "vitamin_b12": 500.0, "triglycerides": 120.0
}
# Expected: Health Score = 100, Risk = Low

DEFICIENCY_TEST_CASE = {
    "hemoglobin": 9.5,  "glucose": 65.0,   "cholesterol": 240.0,
    "vitamin_d": 10.0,  "iron": 40.0,      "calcium": 7.5,
    "vitamin_b12": 150.0, "triglycerides": 200.0
}
# Expected: Health Score = 0, Risk = High
```

---

## 12. Future Enhancements

1. **Integration with Hospital APIs** — Direct pull of blood reports from lab APIs (LOINC-based HL7 FHIR)
2. **Large Language Model Integration** — GPT-4 / Gemini for nuanced, conversational nutrition advice
3. **Wearable Device Sync** — Integration with Fitbit / Apple Health for real-time monitoring
4. **Meal Planner Module** — Weekly meal plans auto-generated from deficiency profile
5. **Doctor Referral System** — Auto-flag critical values and suggest nearest relevant specialists
6. **Multi-language Support** — Tamil, Hindi, and other regional language reports
7. **Mobile Application** — React Native or Flutter-based mobile app
8. **Progressive Profile Learning** — Track improvements across multiple reports over time
9. **Telemedicine Integration** — Video consultation booking with nutritionists
10. **Barcode/QR Scanning** — Scan food product barcodes to check nutritional suitability

---

## 13. Conclusion

The *Personalized Nutrition AI from Blood Test Reports* system successfully demonstrates the practical application of Artificial Intelligence in the healthcare domain. By combining OCR-based data extraction, rule-based clinical analysis, and machine learning prediction, the system provides accessible, personalized dietary guidance without requiring expert medical consultation for routine report interpretation.

The project achieves all stated objectives: secure multi-user authentication, automated blood value extraction from multiple file formats, parameter-level nutrition recommendations, ML-based risk prediction, and professional PDF report generation. The system is built entirely on open-source, freely available technologies, making it deployable on any standard computer without licensing costs.

This work demonstrates that AI-driven health tools can democratize access to expert-level dietary guidance, particularly for populations with limited healthcare access.

---

## 14. Installation Guide

### Prerequisites
- Python 3.9 or higher
- Tesseract-OCR (system-level, for OCR features)
- VS Code (recommended IDE)

### Step 1: Install Tesseract-OCR

**Windows:**
```
Download from: https://github.com/UB-Mannheim/tesseract/wiki
Install to: C:\Program Files\Tesseract-OCR\
Add to PATH environment variable
```

**Linux (Ubuntu/Debian):**
```bash
sudo apt-get install tesseract-ocr
```

**macOS:**
```bash
brew install tesseract
```

### Step 2: Clone / Set Up Project

```bash
# Create project folder
mkdir nutrition_ai
cd nutrition_ai

# Copy all project files into this folder
# (app.py, database.py, ocr_reader.py, nutrition_engine.py,
#  ml_model.py, report_generator.py, requirements.txt)
```

### Step 3: Create Virtual Environment

```bash
# Windows
python -m venv venv
venv\Scripts\activate

# Linux / macOS
python3 -m venv venv
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

If you encounter issues with opencv on some systems:
```bash
pip install opencv-python-headless
```

### Step 5: Pre-train the ML Model (Optional)

```bash
python ml_model.py
# This trains the model and saves it to models/ folder
# It runs automatically on first launch if not done manually
```

### Step 6: Run the Application

```bash
streamlit run app.py
```

The app opens automatically at: **http://localhost:8501**

### Step 7: First-time Setup

1. Click **Register** and create an account
2. Log in with your credentials
3. Navigate to **Manual Entry** or **Upload Report**
4. Enter / upload blood values
5. Click **Analyse** to see results
6. Go to **Download Report** for the PDF

### Project Folder Structure

```
nutrition_ai/
├── app.py                  # Main Streamlit application
├── database.py             # SQLite database operations
├── ocr_reader.py           # OCR and file parsing
├── nutrition_engine.py     # Nutrition recommendation engine
├── ml_model.py             # Machine learning module
├── report_generator.py     # PDF report generator
├── requirements.txt        # Python dependencies
│
├── models/                 # ML model artifacts (auto-created)
│   ├── rf_model.pkl
│   ├── scaler.pkl
│   └── label_encoder.pkl
│
├── reports/                # Generated PDF reports (auto-created)
├── uploads/                # Uploaded files (auto-created)
├── data/                   # Datasets (auto-created)
│
├── nutrition_ai.db         # SQLite database (auto-created on first run)
│
└── docs/
    └── PROJECT_DOCUMENTATION.md   # This file
```

### VS Code Setup Tips

1. Install the **Python** extension by Microsoft
2. Install the **Pylance** extension for autocomplete
3. Set the interpreter to your virtual environment Python:
   - `Ctrl+Shift+P` → "Python: Select Interpreter" → choose `venv`
4. Open integrated terminal: `Ctrl+`` ` ``
5. Run: `streamlit run app.py`

### Troubleshooting

| Problem | Solution |
|---------|----------|
| `tesseract not found` | Install Tesseract-OCR and add to PATH |
| `bcrypt` import error | `pip install bcrypt` |
| `fpdf2` not found | `pip install fpdf2` |
| Model not found | Run `python ml_model.py` first |
| Port 8501 in use | `streamlit run app.py --server.port 8502` |
| SQLite locked | Restart the app; close other instances |
