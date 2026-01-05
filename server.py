from flask import Flask, jsonify, request
from flask_cors import CORS
import pandas as pd
import numpy as np
import joblib
import pymongo
import os
from datetime import datetime, timedelta
from urllib.parse import quote_plus

app = Flask(__name__)
# 允許所有網域連線 (包含 Vercel)
CORS(app)

# ==========================================
# 1. 基礎設定
# ==========================================
# ⚠️ 安全性修正：不要在這裡寫死密碼，Render 環境變數設定好即可
MONGO_USER = os.getenv("MONGO_USER") 
MONGO_PASS = os.getenv("MONGO_PASS") 

# 如果本地測試沒有環境變數，提醒使用者
if not MONGO_USER or not MONGO_PASS:
    print("⚠️ 警告: 未偵測到 MONGO_USER 或 MONGO_PASS 環境變數")

username = quote_plus(MONGO_USER) if MONGO_USER else ""
password = quote_plus(MONGO_PASS) if MONGO_PASS else ""

HOST = "140.116.96.197"
PORT = "22122"
DB_NAME = "IoT"

# 🔥 修改點：將預設集合改成 "m2m_kyle_new_9" (實驗上)，確保讀到最新數據
DEFAULT_COLLECTION = "m2m_kyle_new_9"

MONGO_URI = f"mongodb://{username}:{password}@{HOST}:{PORT}/{DB_NAME}?authSource=admin"

print(f"🔗 正在嘗試連線到 MongoDB... 預設集合: {DEFAULT_COLLECTION}")

# ==========================================
# 2. 載入訓練好的模型
# ==========================================
MODEL_DIR = os.path.join(os.path.dirname(__file__), 'models')

# 加上全域變數初始值，避免當掉
model_rf = None
model_ridge = None
scaler = None

try:
    rf_path = os.path.join(MODEL_DIR, 'orchid_rf.pkl')
    ridge_path = os.path.join(MODEL_DIR, 'orchid_ridge.pkl')
    scaler_path = os.path.join(MODEL_DIR, 'orchid_scaler.pkl')

    if os.path.exists(rf_path):
        model_rf = joblib.load(rf_path)
        model_ridge = joblib.load(ridge_path)
        scaler = joblib.load(scaler_path)
        print("✅ 模型載入成功！")
    else:
        print(f"❌ 找不到模型檔案: {rf_path}")

except Exception as e:
    print(f"❌ 模型載入發生錯誤: {e}")

# ==========================================
# 3. 輔助函式
# ==========================================
def get_mongo_collection(collection_name=None):
    if collection_name is None:
        collection_name = DEFAULT_COLLECTION
    
    # 這裡加入錯誤處理，如果連線字串是空的就報錯
    if not MONGO_URI.startswith("mongodb"):
        raise ValueError("MongoDB URI 設定錯誤，請檢查環境變數")

    client = pymongo.MongoClient(MONGO_URI, serverSelectionTimeoutMS=5000) # 設定 5秒逾時，避免卡死
    db = client[DB_NAME]
    return db[collection_name]

# ==========================================
# 4. API: 歷史數據
# ==========================================
@app.route('/api/history', methods=['GET'])
def get_history():
    try:
        # 允許前端透過 ?collection=xxx 來指定要看哪一個感測器
        target_col = request.args.get('collection', DEFAULT_COLLECTION)
        col = get_mongo_collection(target_col)
        
        projection = {
            '_id': 0, 
            'timestamp': 1, 'Time': 1,
            'temperature': 1, 'Temp': 1,
            'humidity': 1, 'Humid': 1,
            'light': 1, 'Lux': 1,
            'eco2': 1, 'co2': 1
        }
        
        # 抓取最新的 14 筆數據 (搭配前端圖表需求)
        data = list(col.find({}, projection).sort("timestamp", -1).limit(14))
        
        formatted_data = []
        for d in data:
            ts = d.get('timestamp') or d.get('Time')
            temp = d.get('temperature') if d.get('temperature') is not None else d.get('Temp')
            hum = d.get('humidity') if d.get('humidity') is not None else d.get('Humid')
            lux = d.get('light') if d.get('light') is not None else d.get('Lux')
            
            # 同時支援 co2 與 eco2
            co2_val = d.get('co2')
            if co2_val is None:
                co2_val = d.get('eco2')
            if co2_val is None:
                co2_val = 400

            formatted_data.append({
                "timestamp": ts,
                "temperature": temp,
                "humidity": hum,
                "light": lux,
                "co2": co2_val
            })
            
        formatted_data.reverse() # 轉成 時間舊 -> 時間新
        return jsonify(formatted_data)

    except Exception as e:
        print(f"History API Error: {e}")
        return jsonify({"error": str(e)}), 500

# ==========================================
# 5. API: 預測
# ==========================================
@app.route('/api/predict', methods=['GET'])
def predict():
    if model_rf is None:
        return jsonify({"status": "error", "message": "模型未載入，請檢查伺服器日誌"}), 500

    try:
        # 使用預設集合 (m2m_kyle_new_9) 進行預測
        col = get_mongo_collection(DEFAULT_COLLECTION)
        
        # 抓取最近 5000 筆來確保有足夠數據做平均 (Resample)
        data = list(col.find({}, {'_id': 0}).sort("timestamp", -1).limit(5000))
        
        if not data:
            return jsonify({"status": "error", "message": "資料庫無數據"})

        df = pd.DataFrame(data)
        
        # --- 資料處理邏輯 ---
        if 'timestamp' in df.columns: df = df.rename(columns={'timestamp': 'Date'})
        elif 'Time' in df.columns: df = df.rename(columns={'Time': 'Date'})
            
        if 'temperature' in df.columns: df['T_Avg'] = df['temperature']
        elif 'Temp' in df.columns: df['T_Avg'] = df['Temp']
            
        if 'humidity' in df.columns: df['H_Avg'] = df['humidity']
        elif 'Humid' in df.columns: df['H_Avg'] = df['Humid']
            
        if 'light' in df.columns: df['Rsum'] = df['light']
        elif 'Lux' in df.columns: df['Rsum'] = df['Lux']
            
        # 融合 CO2
        if 'co2' in df.columns:
            df['CO2'] = df['co2']
            if 'eco2' in df.columns:
                 df['CO2'] = df['CO2'].fillna(df['eco2'])
        elif 'eco2' in df.columns:
            df['CO2'] = df['eco2']
        else:
            df['CO2'] = 400

        df['Date'] = pd.to_datetime(df['Date'])
        cols_to_clean = ['T_Avg', 'H_Avg', 'Rsum', 'CO2']
        for c in cols_to_clean:
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors='coerce')
                df[c] = df[c].replace(-1, np.nan)
        
        df.set_index('Date', inplace=True)
        # 按日平均
        df_daily = df.resample('D').mean(numeric_only=True).dropna()
        
        if df_daily.empty:
            return jsonify({"status": "error", "message": "有效數據不足"})

        # 取最後一天 (通常就是今天)
        today_row = df_daily.iloc[[-1]].copy()
        
        # 假定 Yield (因為沒有即時產量輸入，先用固定值)
        last_yield = 1000
        yield_file = os.path.join(os.path.dirname(__file__), 'orchid_yield.csv')
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
        
        pred_rf = model_rf.predict(X_input)[0]
        X_scaled = scaler.transform(X_input)
        pred_ridge = model_ridge.predict(X_scaled)[0]
        
        # 找出關鍵因子
        importances = model_rf.feature_importances_
        indices = np.argsort(importances)[::-1][:3]
        root_causes = []
        for idx in indices:
            root_causes.append({
                "factor": features[idx],
                "value": round(float(X_input.iloc[0, idx]), 2),
                "impact": "關鍵因子"
            })

        # 回傳預測結果 (加上 UTC+8 調整顯示日期)
        # Render 伺服器是 UTC，我們加 8 小時讓它顯示台灣時間
        tw_time = datetime.utcnow() + timedelta(hours=8)
        predict_date = (tw_time + timedelta(days=1)).strftime('%Y-%m-%d')

        return jsonify({
            "date": predict_date,
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
    port = int(os.environ.get('PORT', 5000))
    print(f"🚀 Python AI Server starting on port {port}...")
    app.run(host='0.0.0.0', port=port, debug=False)
