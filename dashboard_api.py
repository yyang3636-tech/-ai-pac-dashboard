import pandas as pd
import uvicorn
import pickle
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse

print("載入測試資料串流...")
# 改讀取精簡版的展示資料集
df_demo = pd.read_csv('demo_data.csv')

# 擷取資料，並轉成 list of dicts 提升速度
stream_data = []
for i in range(len(df_demo)): 
    row = df_demo.iloc[i]
    
    stream_data.append({
        "timestamp": row['timestamp'],
        "sensors": {
            "ae_c4003": row['ae_c4003'] if pd.notna(row['ae_c4003']) else None,
            "ae_c4004": row['ae_c4004'] if pd.notna(row['ae_c4004']) else None,
            "ae_c4005": row['ae_c4005'] if pd.notna(row['ae_c4005']) else None,
            "ae_c4006": row['ae_c4006'] if pd.notna(row['ae_c4006']) else None,
            "ait_c5011": row['ait_c5011'] if pd.notna(row['ait_c5011']) else None,
        },
        "health": {
            "ae_c4003": row['ae_c4003_health'] if pd.notna(row['ae_c4003_health']) else 0,
            "ae_c4004": row['ae_c4004_health'] if pd.notna(row['ae_c4004_health']) else 0,
            "ae_c4005": row['ae_c4005_health'] if pd.notna(row['ae_c4005_health']) else 0,
            "ae_c4006": row_h['ae_c4006_health'] if pd.notna(row['ae_c4006_health']) else 0,
            "ait_c5011": row_h['ait_c5011_health'] if pd.notna(row['ait_c5011_health']) else 0,
        },
        "predicted_ss": float(row['predicted_ss']) if pd.notna(row['predicted_ss']) else 20.0
    })

print("載入 ML 模型...")
with open('best_model_pac_dose.pkl', 'rb') as f:
    model = pickle.load(f)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# 前 480 筆是「已經先跑 8 小時」
current_idx = 480

@app.get("/")
def serve_frontend():
    """當使用者連接到網址時，直接回傳 dashboard 網頁"""
    with open('index.html', 'r', encoding='utf-8') as f:
        return HTMLResponse(content=f.read())

@app.get("/api/reset")
def reset_stream():
    global current_idx
    current_idx = 480
    return {"status": "ok"}

@app.get("/api/init_data")
def get_init_data():
    """回傳一開始的 8 小時 (480 筆) 歷史資料"""
    init_chunk = []
    for i in range(480):
        data = stream_data[i]
        pac_dose_pred = model.predict([[data['predicted_ss']]])[0]
        data['pac_dose_ppm'] = float(pac_dose_pred * 1e6)
        init_chunk.append(data)
    return init_chunk

@app.get("/api/next_data")
def get_next_data():
    """回傳第 8 小時又 1 分鐘之後的最新資料"""
    global current_idx
    if current_idx >= len(stream_data):
        current_idx = 480 # 循環播放
    
    data = stream_data[current_idx]
    current_idx += 1
    
    # 呼叫 ML 模型進行預測
    pac_dose_pred = model.predict([[data['predicted_ss']]])[0]
    data['pac_dose_ppm'] = float(pac_dose_pred * 1e6)
    
    return data

if __name__ == "__main__":
    print("啟動 API 伺服器在 http://localhost:8000 ...")
    uvicorn.run(app, host="0.0.0.0", port=8000)
