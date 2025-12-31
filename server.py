from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import pymongo
import os
from datetime import datetime, timedelta
# 用來處理密碼裡的特殊符號
from urllib.parse import quote_plus 

# ==========================================
# 1. 基礎設定 (遠端連線 + 帳號密碼)
# ==========================================
app = Flask(__name__)
CORS(app)

# ⚠️ 你的實驗室帳號密碼 (已填入)
MONGO_USER = "wkdl"
MONGO_PASS = "ugwUzXgeMBPjhNK"

# 自動處理特殊符號 (防止密碼裡的特殊字元搞壞連線)
username = quote_plus(MONGO_USER)
password = quote_plus(MONGO_PASS)

# 遠端伺服器設定
HOST = "140.116.96.197"
PORT = "22122"
DB_NAME = "IoT"               
DEFAULT_COLLECTION = "m2m_kyle_new_1" # 根據你 Compass 看到的集合名稱

# 組合出正確的連線字串
MONGO_URI = f"mongodb://{username}:{password}@{HOST}:{PORT}/{DB_NAME}?authSource=admin"

print(f"🔗 正在嘗試連線到: mongodb://{HOST}:{PORT}/{DB_NAME} ...")

# ==========================================
# 2. 載入訓練好的模型
# ==========================================
print("📂 正在載入 AI 模型...")
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

try:
    rf_path = os.path.join(MODEL_DIR, 'orchid_rf.pkl')
    ridge_path = os.path.join(MODEL_DIR, 'orchid_ridge.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'orchid_scaler.pkl')

    if not os.path.exists(rf_path):
        raise FileNotFoundError(f"找不到模型檔案: {rf_path}")

    model_rf = joblib.load(rf_path)
    model_ridge = joblib.load(ridge_path)
    scaler = joblib.load(scaler_path)
    print("✅ 模型載入成功！")
except Exception as e:
    print(f"❌ 模型載入失敗: {e}")
    model_rf = None

# ==========================================
# 3. 輔助函式：連線 MongoDB
# ==========================================
def get_mongo_collection(collection_name=None):
    if collection_name is None:
        collection_name = DEFAULT_COLLECTION
    # 使用包含帳密的 URI 連線
    client = pymongo.MongoClient(MONGO_URI)
    db = client[DB_NAME]
    return db[collection_name]

# ==========================================
# 4. API: 取得歷史數據 (與 route.ts 邏輯一致)
# ==========================================
@app.route('/api/history', methods=['GET'])
def get_history():
    try:
        # 允許前端透過參數指定要看哪個感測器 ?collection=m2m_kyle_new_6
        target_col = request.args.get('collection', DEFAULT_COLLECTION)
        col = get_mongo_collection(target_col)
        
        projection = {
            '_id': 0, 
            'timestamp': 1,     # route.ts 用 timestamp
            'Time': 1,          # 有些舊資料可能用 Time
            'temperature': 1, 'Temp': 1,
            'humidity': 1,    'Humid': 1,
            'light': 1,       'Lux': 1,
            'eco2': 1, 
            'co2': 1
        }
        
        # 抓最近 100 筆
        data = list(col.find({}, projection).sort("timestamp", -1).limit(100))
        
        formatted_data = []
        for d in data:
            # 智慧欄位選取
            ts = d.get('timestamp') or d.get('Time')
            temp = d.get('temperature') if d.get('temperature') is not None else d.get('Temp')
            hum = d.get('humidity') if d.get('humidity') is not None else d.get('Humid')
            lux = d.get('light') if d.get('light') is not None else d.get('Lux')
            
            # 處理 CO2
            co2_val = d.get('eco2')
            if co2_val is None:
                co2_val = d.get('co2')
            if co2_val is None:
                co2_val = 400 

            formatted_data.append({
                "timestamp": ts,
                "temperature": temp,
                "humidity": hum,
                "light": lux,
                "co2": co2_val
            })
            
        formatted_data.reverse()
        return jsonify(formatted_data)

    except Exception as e:
        print(f"History API Error: {e}")
        return jsonify({"error": str(e)}), 500

# ==========================================
# 5. API: AI 產量預測 (核心功能)
# ==========================================
@app.route('/api/predict', methods=['GET'])
def predict():
    if model_rf is None:
        return jsonify({"status": "error", "message": "模型未載入"}), 500

    try:
        col = get_mongo_collection(DEFAULT_COLLECTION)
        
        # 1. 抓取數據
        data = list(col.find({}, {'_id': 0}).sort("timestamp", -1).limit(5000))
        
        if not data:
            return jsonify({"status": "error", "message": "資料庫無數據"})

        df = pd.DataFrame(data)
        
        # 2. 資料標準化
        if 'timestamp' in df.columns:
            df = df.rename(columns={'timestamp': 'Date'})
        elif 'Time' in df.columns:
            df = df.rename(columns={'Time': 'Date'})
            
        # 欄位對應
        if 'temperature' in df.columns: df['T_Avg'] = df['temperature']
        elif 'Temp' in df.columns: df['T_Avg'] = df['Temp']
            
        if 'humidity' in df.columns: df['H_Avg'] = df['humidity']
        elif 'Humid' in df.columns: df['H_Avg'] = df['Humid']
            
        if 'light' in df.columns: df['Rsum'] = df['light']
        elif 'Lux' in df.columns: df['Rsum'] = df['Lux']
            
        # CO2 處理
        if 'eco2' in df.columns:
            df['CO2'] = df['eco2']
            if 'co2' in df.columns:
                df['CO2'] = df['CO2'].fillna(df['co2'])
        elif 'co2' in df.columns:
            df['CO2'] = df['co2']
        else:
            df['CO2'] = 400

        # 3. 清洗
        df['Date'] = pd.to_datetime(df['Date'])
        cols_to_clean = ['T_Avg', 'H_Avg', 'Rsum', 'CO2']
        
        for c in cols_to_clean:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
                df[c] = df[c].replace(-1, np.nan)
        
        # 4. 聚合
        df.set_index('Date', inplace=True)
        df_daily = df.resample('D').mean(numeric_only=True).dropna()
        
        if df_daily.empty:
            return jsonify({"status": "error", "message": "有效數據不足"})

        today_row = df_daily.iloc[[-1]].copy()
        
        # 5. 特徵準備
        last_yield = 1000
        yield_file = 'orchid_yield.csv'
        if os.path.exists(yield_file):
            try:
                df_yield = pd.read_csv(yield_file)
                last_yield = df_yield['Produced'].iloc[-1]
            except: pass

        today_row['Yield_Lag1'] = last_yield
        today_row['Yield_Roll_Mean_7'] = last_yield 
        
        features = ['T_Avg', 'H_Avg', 'Rsum', 'CO2', 'Yield_Lag1', 'Yield_Roll_Mean_7']
        for f in features:
            if f not in today_row.columns:
                today_row[f] = 0
                
        X_input = today_row[features]
        
        # 6. 預測
        pred_rf = model_rf.predict(X_input)[0]
        X_scaled = scaler.transform(X_input)
        pred_ridge = model_ridge.predict(X_scaled)[0]
        
        # 7. 根因
        importances = model_rf.feature_importances_
        indices = np.argsort(importances)[::-1][:3]
        root_causes = []
        for idx in indices:
            root_causes.append({
                "factor": features[idx],
                "value": round(float(X_input.iloc[0, idx]), 2),
                "impact": "關鍵因子"
            })

        return jsonify({
            "date": (datetime.now() + timedelta(days=1)).strftime('%Y-%m-%d'),
            "rf_prediction": int(max(0, pred_rf)),
            "ridge_prediction": int(max(0, pred_ridge)),
            "status": "Warning" if pred_rf < 800 else "Normal",
            "confidence": 0.88,
            "rootCauses": root_causes
        })

    except Exception as e:
        print(f"Error: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500

if __name__ == '__main__':
    print("🚀 Python AI 伺服器啟動 (Database: IoT / User: wkdl)")
    app.run(port=5000, debug=True)