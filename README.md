

# 🔥 AI-Powered Wildfire Detection and Prediction System

An end-to-end machine learning web application designed to predict wildfire risks, generate autonomous emergency response plans using Generative AI, and instantly alert forest officials via SMS.

## 📖 About the Project

Wildfires cause devastating environmental and property damage every year. The biggest challenge for emergency responders is analyzing real-time weather data quickly enough to formulate an actionable plan.

This project solves that by combining **Machine Learning** for risk calculation with **Google Gemini AI** for strategic planning, and **Twilio** for instant communication.

## ✨ Key Features

* **Real-Time Risk Detection:** Uses Machine Learning to calculate immediate wildfire danger based on Temperature, Wind Speed, and Fuel Moisture Codes.
* **Autonomous AI Action Plans:** Integrates **Google Gemini 2.5 Flash** to analyze extreme weather conditions and instantly generate custom, tactical emergency response plans.
* **Instant SMS Alerts:** Bypasses internet reliance for responders in the field by routing the AI-generated emergency plans directly to their mobile phones via the **Twilio API**.
* **Future Risk Regression:** Predicts future fire intensity indexes to help pre-position resources.

## 🛠️ Tech Stack

* **Backend:** Python, Flask
* **Machine Learning:** Scikit-Learn, Pandas, NumPy
* **External APIs:** Google Gemini API (v1beta), Twilio SMS API
* **Frontend:** HTML5, CSS3, JavaScript

## 🚀 How to Run Locally

1. Clone the repository to your local machine.
2. Create and activate a virtual environment (`venv`).
3. Install the required dependencies:
```bash
pip install flask twilio requests scikit-learn numpy

```


4. Add your API keys to `app.py` (Ensure you do not commit your keys to public version control!):
* `api_key = "YOUR_GOOGLE_GEMINI_KEY"`
* `TWILIO_ACCOUNT_SID = "YOUR_TWILIO_SID"`
* `TWILIO_AUTH_TOKEN = "YOUR_TWILIO_TOKEN"`


5. Run the application:
```bash
python app.py

```


6. Open your web browser and navigate to `http://127.0.0.1:5000/`.

## 🤝 Credits

* **Backend & AI Integrations:** Custom-built for this event (Flask, ML routing, Twilio SMS implementation, Google Gemini API integration).
* **UI Boilerplate & Base Dataset:** Adapted from Aravind Selvam Repository
