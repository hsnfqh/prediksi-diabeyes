from flask import Flask, render_template, request, jsonify
import pickle
import numpy as np
import os
import joblib
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

# Global variables untuk model
model = None
scaler = None

def load_model():
    """Load model dari pickle/joblib file"""
    global model
    try:
        model_path = 'model_diabetes.pkl'
        if os.path.exists(model_path):
            try:
                model = joblib.load(model_path)
            except Exception:
                with open(model_path, 'rb') as f:
                    model = pickle.load(f)
            print("[OK] Model berhasil dimuat!")
            return True
        else:
            print("[ERROR] File model tidak ditemukan!")
            return False
    except Exception as e:
        print(f"[ERROR] Error loading model: {e}")
        return False

# Load model saat module di-import / app startup
load_model()

@app.route('/')
def index():
    """Render halaman utama"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediksi diabetes dengan 4 parameters"""
    try:
        if model is None:
            load_model()
        if model is None:
            return jsonify({'error': 'Model belum dimuat'}), 500
        
        # Ambil data dari request
        data = request.json
        
        # Extract features (hanya 4 parameter sesuai model)
        features = {
            'usia': [float(data.get('usia', 0))],
            'berat_badan': [float(data.get('berat_badan', 0))],
            'olahraga': [float(data.get('olahraga', 0))],
            'gula_harian': [float(data.get('gula_harian', 0))]
        }
        
        import pandas as pd
        input_data = pd.DataFrame(features)
        
        # Prediksi
        prediction = model.predict(input_data)[0]
        prediction_proba = model.predict_proba(input_data)
        
        # Format hasil
        risk_percentage = round(float(prediction_proba[0][1]) * 100, 1)
        is_high_risk = bool(prediction == 1)

        # Extract values for recommendations
        usia_val = float(data.get('usia', 0))
        berat_val = float(data.get('berat_badan', 0))
        olahraga_val = float(data.get('olahraga', 0))
        gula_val = float(data.get('gula_harian', 0))

        recommendations = []

        if is_high_risk:
            status_title = "Berisiko Tinggi Diabetes"
            summary_text = (
                f"Berdasarkan analisis data kesehatan (Usia {int(usia_val)} th, "
                f"Konsumsi Gula {int(gula_val)}g/hari, Olahraga {olahraga_val}x/minggu), "
                f"estimasi tingkat risiko Anda berada di angka {risk_percentage}%."
            )
            if gula_val > 50:
                recommendations.append({
                    "icon": "🥤",
                    "title": "Kurangi Konsumsi Gula",
                    "desc": f"Asupan gula harian Anda ({int(gula_val)}g) melebihi batas aman (maks. 50g/hari). Batasi minuman manis dan camilan tinggi gula."
                })
            if olahraga_val < 3:
                recommendations.append({
                    "icon": "🏃‍♂️",
                    "title": "Tingkatkan Aktivitas Fisik",
                    "desc": f"Frekuensi olahraga saat ini ({olahraga_val}x/minggu) masih kurang. Tingkatkan minimal 3–5 kali seminggu (30 menit per sesi)."
                })
            if berat_val > 75:
                recommendations.append({
                    "icon": "⚖️",
                    "title": "Kendalikan Berat Badan",
                    "desc": "Lakukan kombinasi diet seimbang dan kardio teratur untuk membantu menjaga berat badan ideal dan meningkatkan sensitivitas insulin."
                })
            recommendations.append({
                "icon": "🩺",
                "title": "Konsultasi Medis",
                "desc": "Disarankan untuk melakukan pemeriksaan gula darah rutin (GDP / HbA1c) di klinik atau dokter terdekat."
            })
        else:
            status_title = "Risiko Diabetes Rendah"
            summary_text = (
                f"Hasil analisis menunjukkan tingkat risiko Anda tergolong rendah ({risk_percentage}%). "
                f"Pertahankan gaya hidup sehat yang sedang Anda jalankan!"
            )
            if gula_val > 50:
                recommendations.append({
                    "icon": "⚠️",
                    "title": "Waspadai Asupan Gula",
                    "desc": f"Meskipun risiko masih rendah, konsumsi gula harian ({int(gula_val)}g) tergolong tinggi. Kurangi secara bertahap."
                })
            if olahraga_val < 3:
                recommendations.append({
                    "icon": "💪",
                    "title": "Tingkatkan Olahraga Rutin",
                    "desc": f"Usahakan olahraga minimal 3x seminggu untuk memelihara kebugaran dan metabolisme gula darah."
                })
            else:
                recommendations.append({
                    "icon": "🎉",
                    "title": "Pertahankan Aktivitas Fisik",
                    "desc": f"Kedisiplinan olahraga Anda ({olahraga_val}x/minggu) sangat baik untuk mencegah risiko metabolisme di masa mendatang."
                })
            recommendations.append({
                "icon": "🥗",
                "title": "Pola Makan Sehat",
                "desc": "Perbanyak asupan makanan berserat tinggi seperti sayuran, buah utuh, dan air putih."
            })

        result = {
            'prediction': int(prediction),
            'is_high_risk': is_high_risk,
            'status_title': status_title,
            'risk_percentage': risk_percentage,
            'summary_text': summary_text,
            'recommendations': recommendations
        }
        
        return jsonify(result)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 400

@app.route('/info', methods=['GET'])
def info():
    """Info tentang model"""
    try:
        model_info = {
            'type': str(type(model).__name__) if model else 'Unknown',
            'features': 8,
            'feature_names': [
                'Pregnancies',
                'Glucose',
                'Blood Pressure',
                'Skin Thickness',
                'Insulin',
                'BMI',
                'Diabetes Pedigree Function',
                'Age'
            ]
        }
        return jsonify(model_info)
    except Exception as e:
        return jsonify({'error': str(e)}), 400

if __name__ == '__main__':
    # Load model saat startup
    load_model()
    
    # Jalankan Flask app
    app.run(debug=True, host='0.0.0.0', port=5000)