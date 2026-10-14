from http.server import BaseHTTPRequestHandler
import json
import os
import joblib
import pandas as pd

# Load model once on cold start
MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'startup_profit_model.pkl')
model = None

def get_model():
    global model
    if model is None:
        if os.path.exists(MODEL_PATH):
            model = joblib.load(MODEL_PATH)
    return model

class handler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header('Access-Control-Allow-Origin', '*')
        self.send_header('Access-Control-Allow-Methods', 'GET, POST, OPTIONS')
        self.send_header('Access-Control-Allow-Headers', 'Content-Type')
        self.end_headers()

    def do_POST(self):
        try:
            content_length = int(self.headers.get('Content-Length', 0))
            body = self.rfile.read(content_length)
            data = json.loads(body.decode('utf-8'))

            rd_spend = float(data.get('rd_spend', 0.0))
            admin = float(data.get('administration', 0.0))
            marketing = float(data.get('marketing_spend', 0.0))
            state = str(data.get('state', 'California'))

            m = get_model()
            if m is not None:
                df = pd.DataFrame([{
                    'R&D Spend': rd_spend,
                    'Administration': admin,
                    'Marketing Spend': marketing,
                    'State': state
                }])
                pred = float(m.predict(df)[0])
            else:
                # Mathematical fallback using exact trained regression coefficients
                # Profit = 54028.04 + 0.8056*RD - 0.0688*Admin + 0.0299*Mkt + 938.79*FL + 6.99*NY
                fl = 1.0 if state.lower() == 'florida' else 0.0
                ny = 1.0 if state.lower() == 'new york' else 0.0
                pred = 54028.04 + 0.805629 * rd_spend - 0.068788 * admin + 0.029855 * marketing + 938.793006 * fl + 6.987760 * ny

            response_data = {
                "success": True,
                "predicted_profit": round(pred, 2),
                "inputs": {
                    "rd_spend": rd_spend,
                    "administration": admin,
                    "marketing_spend": marketing,
                    "state": state
                },
                "model_metrics": {
                    "r2_score": 0.8987,
                    "rmse": 9055.96
                }
            }

            self.send_response(200)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps(response_data).encode('utf-8'))

        except Exception as e:
            self.send_response(500)
            self.send_header('Content-Type', 'application/json')
            self.send_header('Access-Control-Allow-Origin', '*')
            self.end_headers()
            self.wfile.write(json.dumps({"success": False, "error": str(e)}).encode('utf-8'))

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Access-Control-Allow-Origin', '*')
        self.end_headers()
        self.wfile.write(json.dumps({
            "status": "healthy",
            "message": "50 Startups Profit Prediction API is running on Vercel Serverless",
            "model_ready": os.path.exists(MODEL_PATH)
        }).encode('utf-8'))
