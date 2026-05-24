import os
import pickle
import bz2
from flask import Flask, request, jsonify, render_template
import numpy as np
import pandas as pd
import warnings
from sklearn.preprocessing import StandardScaler
from twilio.rest import Client
import requests
from dotenv import load_dotenv

warnings.filterwarnings("ignore")

from app_logger import log

# =============================
# Load Environment Variables
# =============================
load_dotenv()

app = Flask(__name__)

# =============================
# Twilio Configuration
# =============================
ACCOUNT_SID = "ID"
AUTH_TOKEN = "TOKEN"
TWILIO_NUMBER = "TWILIO NUMBER"   # your Twilio number
TO_NUMBER = "TO NUMBER"      # your number

# =============================
# Load Models
# =============================
pickle_in = bz2.BZ2File('model/classification.pkl', 'rb')
R_pickle_in = bz2.BZ2File('model/regression.pkl', 'rb')
model_C = pickle.load(pickle_in)
model_R = pickle.load(R_pickle_in)

# =============================
# Fit Scaler on Training Dataset
# =============================
df = pd.read_csv("dataset/algerian_forest_fires_dataset_CLEANED.csv")
df.columns = df.columns.str.strip()   # clean column names
scaler = StandardScaler()
X = df[['Temperature', 'Ws', 'FFMC', 'DMC', 'ISI']]
scaler.fit(X)
log.info("StandardScaler fitted on [Temperature, Ws, FFMC, DMC, ISI]")

# =============================
# Helper Functions
# =============================
def send_alert(message):
    try:
        print("SENDING SMS...")
        client = Client(ACCOUNT_SID, AUTH_TOKEN)
        msg = client.messages.create(
            body=message,
            from_=TWILIO_NUMBER,
            to=TO_NUMBER
        )
        print("MESSAGE SID:", msg.sid)
    except Exception as e:
        print("TWILIO ERROR:", str(e))
def generate_llm_alert(temperature, ws, risk_level, risk_value):
    # 1. PASTE YOUR REAL API KEY INSIDE THESE QUOTES!
    api_key = "YOUR REAL API KEYE" 
    
    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash:generateContent?key={api_key}"
    
    prompt = f"""
    You are an emergency alert AI for forest officials. 
    Based on the following sensor data, generate a short, urgent, plain-text SMS alert (maximum 3 sentences).
    Do not use markdown. Include specific immediate actions for the officials.
    
    Data:
    - Risk Level: {risk_level}
    - Temperature: {temperature}°C
    - Wind Speed: {ws} km/h
    - Fire Index: {risk_value:.2f}
    """
    
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(url, json=payload, headers=headers)
        data = response.json()
        
        # 🚨 THE INDESTRUCTIBLE EXTRACTOR
        # This recursively searches the entire JSON for the AI's text message,
        # completely bypassing Python's list/dictionary bracket errors.
        def extract_text(obj):
            if isinstance(obj, dict):
                if 'text' in obj and isinstance(obj['text'], str):
                    return obj['text']
                for v in obj.values():
                    res = extract_text(v)
                    if res: return res
            elif isinstance(obj, list):
                for item in obj:
                    res = extract_text(item)
                    if res: return res
            return None
            
        ai_message = extract_text(data)
        
        # If the search finds the text, return it. Otherwise, use fallback.
        if ai_message:
            return ai_message.strip()
        else:
            return f"🔥 WILDFIRE ALERT 🚨 Risk Level: {risk_level}. Take action immediately!"
            
    except Exception as e:
        print(f"\n🚨 CRASH PREVENTED: {repr(e)}\n")
        return f"🔥 WILDFIRE ALERT 🚨 Risk Level: {risk_level}. Take action immediately!"
# =============================
# Routes
# =============================
@app.route('/')
def home():
    log.info('Home page loaded successfully')
    return render_template('index.html')

@app.route('/predict_api', methods=['POST'])
def predict_api():
    try:
        data = request.json['data']
        log.info(f'Input from API: {data}')
        new_data = [list(data.values())]
        new_data_scaled = scaler.transform(new_data)
        
        # Fixed: .item() extracts the exact integer from the numpy array
        output = int(model_C.predict(new_data_scaled).item()) 
        
        if output == 0:
            text = 'Forest is in Safe!'
        else:
            text = 'Forest is Danger!'
        return jsonify(text=text, output=output)
    except Exception as e:
        log.error(f'Error in API input: {e}', exc_info=True)
        return jsonify(error="Check the input again!")

# Classification Model Route (LLM Integrated)
@app.route('/predict', methods=['POST'])
def predict():
    print("🔥 TOP PREDICT BUTTON WAS CLICKED! 🔥")
    try:
        temperature = float(request.form['Temperature'])
        ws = float(request.form['Ws'])
        ffmc = float(request.form['FFMC'])
        dmc = float(request.form['DMC'])
        isi = float(request.form['ISI'])

        final_features = [[temperature, ws, ffmc, dmc, isi]]
        final_features_scaled = scaler.transform(final_features)

        # Fixed: Converted numpy array elements to standard Python data types
        output = int(model_C.predict(final_features_scaled).item())
        
        prob_array = model_C.predict_proba(final_features_scaled)
        confidence = round(float(np.max(prob_array)) * 100, 2)
        
        risk_value = float(model_R.predict(final_features_scaled).item())

        if risk_value > 15:
            risk_level = "HIGH"
        elif risk_value > 8:
            risk_level = "MEDIUM"
        else:
            risk_level = "LOW"
            
        log.info('Prediction done for Classification model')

        if output == 0:
            text = 'Forest is Safe!'
            alert_msg = "" # No alert needed
        else:
            text = 'Forest is in Danger!'
            print("ALERT FUNCTION CALLED")
            
            alert_msg = generate_llm_alert(temperature, ws, risk_level, risk_value)
            print("LLM GENERATED ALERT:", alert_msg) 
            send_alert(alert_msg)

        return render_template(
            'index.html',
            prediction_text1=f"{text} | Risk Level: {risk_level} | Index: {risk_value:.2f}",
            confidence=confidence,
            risk_level=risk_level,
            llm_message=alert_msg
        )
    except Exception as e:
        print("🚨 TOP FORM CRASH REASON 🚨:", repr(e))
        log.error(f'Input error in Classification: {e}', exc_info=True)
        return render_template('index.html', prediction_text1="Check the Input again!!!")

# Regression Model Route (LLM Integrated)
@app.route('/predictR', methods=['POST'])
def predictR():
    print("📊 BOTTOM PREDICT BUTTON WAS CLICKED! 📊")
    try:
        temperature = float(request.form['Temperature'])
        ws = float(request.form['Ws'])
        ffmc = float(request.form['FFMC'])
        dmc = float(request.form['DMC'])
        isi = float(request.form['ISI'])

        data = [[temperature, ws, ffmc, dmc, isi]]
        data_scaled = scaler.transform(data)

        # Fixed: .item() to extract standard Python float
        output = float(model_R.predict(data_scaled).item())
        log.info('Prediction done for Regression model')

        if output > 15:
            alert_msg = generate_llm_alert(temperature, ws, "HIGH", output)
            send_alert(alert_msg)
            
            return render_template(
                'index.html',
                prediction_text2=f"Fuel Moisture Code index = {output:.4f} ---- Warning!!! High hazard rating",
                llm_message=alert_msg
            )
        else:
            return render_template(
                'index.html',
                prediction_text2=f"Fuel Moisture Code index = {output:.4f} ---- Safe.. Low hazard rating",
                llm_message=""
            )

    except Exception as e:
        print("🚨 BOTTOM FORM CRASH REASON 🚨:", repr(e))
        log.error(f'Input error in Regression: {e}', exc_info=True)
        return render_template('index.html', prediction_text2="Check the Input again!!!")

# =============================
# Run App
# =============================
if __name__ == "__main__":
    app.run(debug=False)