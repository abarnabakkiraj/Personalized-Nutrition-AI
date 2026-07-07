# 🩺 Personalized Nutrition AI from Blood Test Reports

> Final Year / Mini Project — B.Tech CSE  
> AI-powered blood report analysis & personalized nutrition recommendations

---

## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. (Optional) Pre-train ML model
python ml_model.py

# 3. Launch the app
streamlit run app.py
```

Open: **http://localhost:8501**

---

## Features

| Feature | Description |
|---------|-------------|
| 🔐 Auth | Register / Login with bcrypt-hashed passwords |
| 📤 File Upload | PDF, Image (JPG/PNG), CSV blood reports |
| 🔍 OCR | Pytesseract extracts values from images/PDFs |
| 🧪 Analysis | 8 blood parameters classified vs. clinical ranges |
| 🥗 Nutrition | Per-parameter food and lifestyle recommendations |
| 🤖 ML Model | Random Forest predicts Low/Medium/High health risk |
| 📊 Dashboard | Gauge chart, radar chart, parameter cards |
| 📈 History | Trend charts for health score over time |
| 📄 PDF Report | Downloadable personalized nutrition report |

## Blood Parameters Analyzed

- Hemoglobin · Glucose · Total Cholesterol · Vitamin D
- Serum Iron · Calcium · Vitamin B12 · Triglycerides

## Project Structure

```
nutrition_ai/
├── app.py              ← Streamlit UI (main entry point)
├── database.py         ← SQLite operations
├── ocr_reader.py       ← OCR + file parsing
├── nutrition_engine.py ← Rule-based recommendation engine
├── ml_model.py         ← Random Forest risk predictor
├── report_generator.py ← PDF report builder
├── requirements.txt    ← Python dependencies
├── data/               ← Sample datasets
├── docs/               ← Full project documentation
└── models/             ← Saved ML model (auto-generated)
```

## Technology Stack

Python · Streamlit · SQLite · Scikit-learn  
Pandas · NumPy · OpenCV · Pytesseract  
Matplotlib · Plotly · FPDF2 · bcrypt

---

See `docs/PROJECT_DOCUMENTATION.md` for the complete academic documentation including Abstract, System Architecture, ER Diagram, Algorithms, Testing, and Future Enhancements.
