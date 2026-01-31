import pandas as pd
import re

### rewrite this into one function??


def process_text(text): #function takes in a catalog string strips values and units as well as constant labels
    if not text or pd.isna(text):
        return ""
    
    #
    text = re.sub(r'Item Name:\s*', '', text)
    text = re.sub(r'Bullet Point \d+:\s*', '', text)
    text = re.sub(r'Product Description:\s*', '', text)
    
    # rm value unit and use separately
    text = re.sub(r'Value:\s*[\d.]+\s*Unit:\s*\w+\s*', '', text, flags=re.IGNORECASE)

    text = re.sub(r'\s*-\s*-\s*(-\s*)+', ' ', text)
    
    #whitespace
    text = re.sub(r'\s+', ' ', text).strip()

    
    return text


def extract_numeric_features(text): #function takes in a catalog string pulls the measurement and normalises it based on categories below and corresponding value
    value_match = re.search(r'Value:\s*([\d.]+)', text)
    value = float(value_match.group(1)) if value_match else 0.0
    unit_match = re.search(r'Unit:\s*([^\n\r]+)', text, re.IGNORECASE)
    unit = unit_match.group(1).lower().strip() if unit_match else "unknown"
    unit = re.sub(r"\s+", " ", unit)          # collapse repeated spaces
    unit = unit.strip()

    unit_normalization = {
        # WEIGHT
        "ounce": "weight",
        "ounces": "weight",
        "oz": "weight",
        "gram": "weight",
        "grams": "weight",
        "gramm": "weight",
        "lb": "weight",
        "pound": "weight",
        "pounds": "weight",

        # VOLUME
        "fl oz": "volume",
        "fl. oz": "volume",
        "fluid ounce": "volume",
        "fluid ounces": "volume",
        "fluid ounce(s)": "volume",
        "milliliter": "volume",
        "millilitre": "volume",
        "liters": "volume",

        # COUNT
        "count": "count",
        "ct": "count",
        "each": "count",
        "bottle": "count",
        "pack": "count",
        "packs": "count",
        "can": "count",
        "bag": "count",
        "box": "count",
        "piece": "count",
        "per box": "count",
        "each / pack: 1": "count",

        # MISSING
        "none": "other",
        "unknown": "other",
    }

    unit_group = unit_normalization.get(unit, "other")
    unit_map = {"weight": 0, "volume": 1, "count": 2}
    OTHER_CODE = 3
    unit_encoded = unit_map.get(unit_group, OTHER_CODE)

    return [value, unit_encoded]


