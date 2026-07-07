"""
nutrition_engine.py - Nutrition Recommendation Engine
Compares blood values against normal ranges, classifies status (Low/Normal/High),
and generates personalized food and lifestyle recommendations.
"""

# ── Normal Reference Ranges ──────────────────────────────────────────────────
# Source: Standard clinical reference ranges (adult, mixed gender baseline)
# Tuple format: (min_normal, max_normal, unit)
NORMAL_RANGES = {
    "hemoglobin":    (12.0, 17.5,  "g/dL"),
    "glucose":       (70.0, 100.0, "mg/dL"),
    "cholesterol":   (0.0,  200.0, "mg/dL"),
    "vitamin_d":     (20.0, 50.0,  "ng/mL"),
    "iron":          (60.0, 170.0, "µg/dL"),
    "calcium":       (8.5,  10.5,  "mg/dL"),
    "vitamin_b12":   (200.0, 900.0,"pg/mL"),
    "triglycerides": (0.0,  150.0, "mg/dL"),
}

# ── Nutrition Recommendations Database ──────────────────────────────────────
# Format: {parameter: {status: {"foods": [...], "avoid": [...], "tips": [...]}}}
NUTRITION_DB = {
    "hemoglobin": {
        "Low": {
            "foods": [
                "🥬 Spinach & dark leafy greens",
                "🫘 Lentils and kidney beans",
                "🥩 Red meat (lean cuts)",
                "🫀 Beetroot",
                "🌰 Pumpkin seeds",
                "🐟 Sardines",
                "🍫 Dark chocolate (70%+)",
                "🥜 Peanut butter",
            ],
            "avoid": ["Tea/coffee immediately after meals (inhibits iron absorption)"],
            "tips": [
                "Pair iron-rich foods with Vitamin C (citrus, tomatoes) to boost absorption.",
                "Cook in cast iron pans when possible.",
                "Get checked for iron-deficiency anaemia.",
            ],
        },
        "High": {
            "foods": ["Stay well hydrated", "🫐 Blueberries", "🍵 Green tea"],
            "avoid": ["Excess red meat", "Iron supplements without doctor advice"],
            "tips": ["Consult a doctor — high Hb may indicate polycythaemia."],
        },
    },

    "glucose": {
        "Low": {
            "foods": [
                "🍌 Banana",
                "🥛 Milk",
                "🍯 Honey (small amount)",
                "🧃 Fruit juice",
                "🍞 Whole-grain bread",
            ],
            "avoid": ["Skipping meals", "Excessive alcohol"],
            "tips": [
                "Eat small, frequent meals every 3–4 hours.",
                "Carry glucose tablets for emergencies.",
            ],
        },
        "High": {
            "foods": [
                "🥦 Broccoli & bitter gourd",
                "🌾 Oats & barley",
                "🫘 Legumes and beans",
                "🐟 Fatty fish (salmon, mackerel)",
                "🥑 Avocado",
                "🥜 Almonds and walnuts",
                "🍋 Lemon water (before meals)",
            ],
            "avoid": [
                "White rice in excess",
                "Sugary drinks and sodas",
                "Processed carbohydrates",
                "Deep-fried foods",
            ],
            "tips": [
                "Exercise for at least 30 minutes daily.",
                "Monitor blood sugar regularly.",
                "Consider consulting a diabetologist.",
            ],
        },
    },

    "cholesterol": {
        "High": {
            "foods": [
                "🌾 Oats (beta-glucan lowers LDL)",
                "🐟 Fatty fish (omega-3)",
                "🫒 Olive oil",
                "🥑 Avocado",
                "🍎 Apples, grapes, strawberries",
                "🥜 Almonds and walnuts",
                "🧄 Garlic",
                "🫘 Beans and lentils",
            ],
            "avoid": [
                "Trans fats (margarine, fried fast food)",
                "Full-fat dairy",
                "Egg yolks in excess",
                "Processed meats",
            ],
            "tips": [
                "Aim for 30 min of aerobic exercise 5 days/week.",
                "Quit smoking — it lowers HDL.",
                "Reduce saturated fat to <7% of daily calories.",
            ],
        },
        "Low": {
            "foods": ["🥚 Eggs", "🧀 Cheese (moderate)", "🥩 Lean meats"],
            "avoid": [],
            "tips": ["Very low cholesterol is rare; consult your doctor."],
        },
    },

    "vitamin_d": {
        "Low": {
            "foods": [
                "☀️  20 min of morning sunlight daily",
                "🐟 Salmon, tuna, mackerel",
                "🥚 Egg yolk",
                "🍄 UV-exposed mushrooms",
                "🥛 Fortified milk and dairy",
                "🧈 Cod liver oil",
                "🌾 Fortified cereals",
            ],
            "avoid": ["Excessive sunscreen during morning sun exposure"],
            "tips": [
                "Vitamin D3 + K2 supplements are commonly prescribed.",
                "Magnesium helps activate Vitamin D.",
                "Retest after 3 months of supplementation.",
            ],
        },
        "High": {
            "foods": ["Stay hydrated", "🥦 Calcium-rich foods"],
            "avoid": ["Vitamin D supplements", "Fortified foods in excess"],
            "tips": ["Hypervitaminosis D is serious — consult a doctor immediately."],
        },
    },

    "iron": {
        "Low": {
            "foods": [
                "🥩 Lean red meat",
                "🍗 Poultry",
                "🐟 Seafood",
                "🫘 Legumes (chickpeas, lentils)",
                "🥬 Spinach",
                "🌰 Pumpkin seeds",
                "🍫 Dark chocolate",
                "🍅 Tomatoes (with iron-rich foods)",
            ],
            "avoid": ["Calcium supplements with iron-rich meals", "Excess coffee/tea"],
            "tips": [
                "Take iron supplements on an empty stomach if prescribed.",
                "Vitamin C doubles iron absorption from plant sources.",
            ],
        },
        "High": {
            "foods": ["🍵 Green tea", "☕ Coffee (in moderation — reduces iron absorption)"],
            "avoid": ["Red meat in excess", "Iron-fortified foods", "Iron supplements"],
            "tips": ["High serum iron may indicate haemochromatosis — see a specialist."],
        },
    },

    "calcium": {
        "Low": {
            "foods": [
                "🥛 Milk, yoghurt, cheese",
                "🥬 Kale and broccoli",
                "🐟 Sardines with bones",
                "🌾 Fortified orange juice",
                "🫘 White beans",
                "🥜 Almonds",
                "🍊 Oranges",
            ],
            "avoid": ["High sodium foods (causes calcium loss)", "Excess caffeine"],
            "tips": [
                "Calcium needs Vitamin D for absorption — check Vit D levels too.",
                "Split calcium supplements — body absorbs max 500 mg at once.",
            ],
        },
        "High": {
            "foods": ["💧 Drink 2–3 L water daily"],
            "avoid": ["Calcium supplements", "Antacids with calcium", "Fortified foods"],
            "tips": ["Hypercalcaemia can indicate parathyroid issues — consult a doctor."],
        },
    },

    "vitamin_b12": {
        "Low": {
            "foods": [
                "🥩 Beef liver (richest source)",
                "🐟 Clams, salmon, tuna",
                "🥚 Eggs",
                "🥛 Dairy products",
                "🌾 Fortified plant milks (for vegans)",
                "🍫 Nutritional yeast",
                "🫘 Fortified cereals",
            ],
            "avoid": ["Proton pump inhibitors (reduce B12 absorption)"],
            "tips": [
                "B12 deficiency is common in vegans and vegetarians.",
                "Injections are more effective than oral supplements for severe deficiency.",
                "B12 is critical for nerve health and red blood cell formation.",
            ],
        },
        "High": {
            "foods": [],
            "avoid": ["B12 supplements"],
            "tips": ["Excessively high B12 (without supplementing) may warrant investigation."],
        },
    },

    "triglycerides": {
        "High": {
            "foods": [
                "🐟 Omega-3 rich fish (salmon, mackerel)",
                "🥜 Walnuts and flaxseeds",
                "🫒 Olive oil",
                "🥦 Non-starchy vegetables",
                "🍊 Citrus fruits (in moderation)",
                "🌾 Whole grains only",
                "🧄 Garlic",
            ],
            "avoid": [
                "Alcohol (significantly raises triglycerides)",
                "Sugary foods and drinks",
                "Refined carbohydrates (white bread, pasta)",
                "Fruit juice in large amounts",
                "Full-fat dairy",
            ],
            "tips": [
                "Lose 5–10% body weight if overweight.",
                "Exercise reduces triglycerides significantly.",
                "High triglycerides + low HDL = high cardiovascular risk.",
            ],
        },
        "Low": {
            "foods": [],
            "avoid": [],
            "tips": ["Low triglycerides are generally healthy."],
        },
    },
}

PARAMETER_LABELS = {
    "hemoglobin":    "Hemoglobin",
    "glucose":       "Blood Glucose",
    "cholesterol":   "Total Cholesterol",
    "vitamin_d":     "Vitamin D",
    "iron":          "Serum Iron",
    "calcium":       "Calcium",
    "vitamin_b12":   "Vitamin B12",
    "triglycerides": "Triglycerides",
}


def classify_parameter(param: str, value: float) -> str:
    """
    Classifies a blood parameter value as 'Low', 'Normal', or 'High'.
    Returns the classification string.
    """
    if param not in NORMAL_RANGES:
        return "Unknown"
    lo, hi, _ = NORMAL_RANGES[param]
    if value < lo:
        return "Low"
    elif value > hi:
        return "High"
    return "Normal"


def get_recommendations(param: str, status: str) -> dict:
    """Returns the recommendation dict for a given parameter and status."""
    param_db = NUTRITION_DB.get(param, {})
    return param_db.get(status, {"foods": [], "avoid": [], "tips": ["Values are normal. Maintain a balanced diet."]})


def analyze_blood_report(values: dict) -> list:
    """
    Master function: takes a dict of {parameter: value} and returns
    a list of analysis results, each containing:
      - parameter, label, value, unit, status, normal_range, recommendation
    """
    results = []
    for param, value in values.items():
        if value is None or value == "":
            continue
        try:
            value = float(value)
        except (ValueError, TypeError):
            continue

        status = classify_parameter(param, value)
        rec    = get_recommendations(param, status)
        lo, hi, unit = NORMAL_RANGES.get(param, (None, None, ""))

        results.append({
            "parameter":    param,
            "label":        PARAMETER_LABELS.get(param, param.replace("_", " ").title()),
            "value":        value,
            "unit":         unit,
            "status":       status,
            "normal_range": f"{lo} – {hi} {unit}" if lo is not None else "N/A",
            "foods":        rec.get("foods", []),
            "avoid":        rec.get("avoid", []),
            "tips":         rec.get("tips", []),
        })
    return results


def calculate_health_score(results: list) -> float:
    """
    Calculates a health score (0–100) based on how many parameters are normal.
    Normal = 100 pts, Low/High = deducted proportionally.
    """
    if not results:
        return 0.0
    normal_count = sum(1 for r in results if r["status"] == "Normal")
    score = (normal_count / len(results)) * 100
    return round(score, 1)


def get_risk_level(health_score: float) -> str:
    """Maps health score to risk category."""
    if health_score >= 80:
        return "Low"
    elif health_score >= 50:
        return "Medium"
    return "High"
