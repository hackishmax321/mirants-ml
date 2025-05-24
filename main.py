from fastapi import FastAPI, HTTPException
import numpy as np
import joblib
from pydantic import BaseModel

# Load the trained Random Forest model
rf_model = joblib.load('random_forest_model.joblib')

# Define your country encoding dictionary (you'll need to define this)
# Example: country_encoding = {'United States': 1, 'Canada': 2, ...}
country_encoding = {
    "Afghanistan": 0,
    "Albania": 1,
    "Algeria": 2,
    "American Samoa": 3,
    "Andorra": 4,
    "Angola": 5,
    "Antigua and Barbuda": 6,
    "Argentina": 7,
    "Armenia": 8,
    "Aruba": 9,
    "Australia": 10,
    "Austria": 11,
    "Azerbaijan": 12,
    "Bahrain": 13,
    "Bangladesh": 14,
    "Barbados": 15,
    "Belarus": 16,
    "Belgium": 17,
    "Belize": 18,
    "Benin": 19,
    "Bermuda": 20,
    "Bhutan": 21,
    "Bosnia and Herzegovina": 22,
    "Botswana": 23,
    "Brazil": 24,
    "Brunei Darussalam": 25,
    "Bulgaria": 26,
    "Burkina Faso": 27,
    "Burundi": 28,
    "Cabo Verde": 29,
    "Cambodia": 30,
    "Cameroon": 31,
    "Canada": 32,
    "Cayman Islands": 33,
    "Central African Republic": 34,
    "Chad": 35,
    "Chile": 36,
    "China": 37,
    "Colombia": 38,
    "Comoros": 39,
    "Costa Rica": 40,
    "Croatia": 41,
    "Cuba": 42,
    "Cyprus": 43,
    "Czechia": 44,
    "Denmark": 45,
    "Djibouti": 46,
    "Dominica": 47,
    "Dominican Republic": 48,
    "Ecuador": 49,
    "El Salvador": 50,
    "Equatorial Guinea": 51,
    "Eritrea": 52,
    "Estonia": 53,
    "Eswatini": 54,
    "Ethiopia": 55,
    "Faroe Islands": 56,
    "Fiji": 57,
    "Finland": 58,
    "France": 59,
    "French Polynesia": 60,
    "Gabon": 61,
    "Georgia": 62,
    "Germany": 63,
    "Ghana": 64,
    "Gibraltar": 65,
    "Greece": 66,
    "Greenland": 67,
    "Grenada": 68,
    "Guam": 69,
    "Guatemala": 70,
    "Guinea": 71,
    "Guinea-Bissau": 72,
    "Guyana": 73,
    "Haiti": 74,
    "Honduras": 75,
    "Hungary": 76,
    "Iceland": 77,
    "India": 78,
    "Indonesia": 79,
    "Iraq": 80,
    "Ireland": 81,
    "Isle of Man": 82,
    "Israel": 83,
    "Italy": 84,
    "Jamaica": 85,
    "Japan": 86,
    "Jordan": 87,
    "Kazakhstan": 88,
    "Kenya": 89,
    "Kiribati": 90,
    "Kuwait": 91,
    "Latvia": 92,
    "Lebanon": 93,
    "Lesotho": 94,
    "Liberia": 95,
    "Libya": 96,
    "Liechtenstein": 97,
    "Lithuania": 98,
    "Luxembourg": 99,
    "Madagascar": 100,
    "Malawi": 101,
    "Malaysia": 102,
    "Maldives": 103,
    "Mali": 104,
    "Malta": 105,
    "Marshall Islands": 106,
    "Mauritania": 107,
    "Mauritius": 108,
    "Mexico": 109,
    "Monaco": 110,
    "Mongolia": 111,
    "Montenegro": 112,
    "Morocco": 113,
    "Mozambique": 114,
    "Myanmar": 115,
    "Namibia": 116,
    "Nauru": 117,
    "Nepal": 118,
    "Netherlands": 119,
    "New Caledonia": 120,
    "New Zealand": 121,
    "Nicaragua": 122,
    "Niger": 123,
    "Nigeria": 124,
    "North Macedonia": 125,
    "Northern Mariana Islands": 126,
    "Norway": 127,
    "Oman": 128,
    "Pakistan": 129,
    "Palau": 130,
    "Panama": 131,
    "Papua New Guinea": 132,
    "Paraguay": 133,
    "Peru": 134,
    "Philippines": 135,
    "Poland": 136,
    "Portugal": 137,
    "Puerto Rico": 138,
    "Qatar": 139,
    "Romania": 140,
    "Russian Federation": 141,
    "Rwanda": 142,
    "Samoa": 143,
    "San Marino": 144,
    "Sao Tome and Principe": 145,
    "Saudi Arabia": 146,
    "Senegal": 147,
    "Serbia": 148,
    "Seychelles": 149,
    "Sierra Leone": 150,
    "Singapore": 151,
    "Sint Maarten (Dutch part)": 152,
    "Slovenia": 153,
    "Solomon Islands": 154,
    "Somalia": 155,
    "South Africa": 156,
    "South Sudan": 157,
    "Spain": 158,
    "Sri Lanka": 159,
    "Sudan": 160,
    "Suriname": 161,
    "Sweden": 162,
    "Switzerland": 163,
    "Syrian Arab Republic": 164,
    "Tajikistan": 165,
    "Thailand": 166,
    "Timor-Leste": 167,
    "Togo": 168,
    "Tonga": 169,
    "Trinidad and Tobago": 170,
    "Tunisia": 171,
    "Turkmenistan": 172,
    "Turks and Caicos Islands": 173,
    "Tuvalu": 174,
    "Uganda": 175,
    "Ukraine": 176,
    "United Arab Emirates": 177,
    "United Kingdom": 178,
    "United States": 179,
    "Uruguay": 180,
    "Uzbekistan": 181,
    "Vanuatu": 182,
    "Viet Nam": 183,
    "Zambia": 184,
    "Zimbabwe": 185
} 

country_population_millions = [
    40.1, 2.8, 44.2, 0.05, 0.08, 35.0, 0.1, 45.4, 3.0, 0.1,
    26.0, 9.0, 10.0, 1.7, 170.0, 0.3, 9.4, 11.6, 0.4, 12.1,
    0.06, 0.8, 3.3, 2.4, 212.6, 0.44, 6.9, 20.9, 11.2, 0.56,
    16.0, 27.2, 38.0, 0.07, 5.5, 17.2, 19.5, 1440.0, 50.9, 0.9,
    5.2, 3.9, 11.3, 1.2, 10.7, 5.8, 0.99, 0.07, 11.1, 18.0,
    6.5, 1.4, 3.6, 1.3, 126.5, 0.05, 0.9, 5.5, 67.3, 0.28,
    2.3, 3.7, 83.0, 33.5, 0.03, 10.5, 0.06, 0.1, 0.1, 17.0,
    0.17, 17.1, 13.1, 11.6, 1.6, 10.4, 2.7, 9.6, 0.36, 0.7,
    0.4, 12.1, 38.5, 5.5, 0.08, 2.8, 0.04, 4.6, 6.8, 0.06,
    4.6, 10.0, 25.0, 0.01, 30.0, 17.4, 2.3, 6.7, 216.7, 2.0,
    0.05, 5.1, 2.5, 213.0, 0.29, 3.5, 9.4, 6.8, 34.0, 113.0,
    126.0, 10.5, 19.4, 18.1, 47.0, 54.9, 4.6, 0.09, 0.2, 35.3,
    17.7, 6.7, 0.1, 8.4, 0.09, 7.4, 39.9, 43.8, 9.8, 4.3,
    22.0, 10.1, 1.1, 6.7, 0.01, 45.7, 31.3, 9.3, 4.3, 0.1,
    8.9, 1.2, 3.8, 1.1, 4.2, 0.4, 3.3, 7.0, 12.3, 9.3,
    1.2, 1.4, 8.4, 5.9, 1.0, 0.6, 6.3, 1.9, 0.1, 44.3,
    0.02, 30.3, 19.7, 83.9, 67.3, 333.0, 3.5, 35.2, 0.3, 0.3,
    47.0, 20.0, 19.0, 46.0, 19.5
]

app = FastAPI()

class PredictionRequest(BaseModel):
    country_name: str
    year: int

@app.post("/predict/net-migration")
def predict_net_migration(request: PredictionRequest):
    try:
        # Encode country
        if request.country_name not in country_encoding:
            raise HTTPException(
                status_code=400, 
                detail=f"Country '{request.country_name}' not found in encoding dictionary."
            )

        country_code = country_encoding[request.country_name]

        # Get population in millions from predefined list
        try:
            population_millions = country_population_millions[country_code]
        except IndexError:
            raise HTTPException(
                status_code=500, 
                detail=f"No population data available for '{request.country_name}' (code {country_code})."
            )

        # Prepare input
        X_input = np.array([[country_code, request.year, population_millions]])

        # Predict
        prediction = rf_model.predict(X_input)[0]
        if request.year > 2023:
            delta_year = request.year - 2023
            adjustment_factor = 1 + delta_year * 0.03 
            prediction *= adjustment_factor

        return {"predicted_net_migration": round(prediction, 2)}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# For testing
@app.get("/")
def read_root():
    return {"message": "Net Migration Prediction API"}