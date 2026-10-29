"""
fastapi_app.py
--------------
REST API & Authentication endpoints for Smart Disease Prediction System.
"""
import os
import sys
import json
from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel, EmailStr
from typing import List, Optional

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
import predict as predict_module
import auth as auth_module

app = FastAPI(title="Smart Disease Prediction System API", version="1.0.0")

MODEL_PATH = os.path.join(os.path.dirname(__file__), "models", "ensemble_model.pkl")

# Initialize DB
auth_module.init_db()

class RegisterRequest(BaseModel):
    full_name: str
    email: EmailStr
    password: str
    date_of_birth: str

class LoginRequest(BaseModel):
    email: EmailStr
    password: str

class PredictRequest(BaseModel):
    user_id: Optional[int] = None
    symptoms: List[str]

@app.get("/")
def root():
    return {"status": "online", "message": "Smart Disease Prediction System API is running."}

@app.post("/api/auth/register")
def register(req: RegisterRequest):
    try:
        user_id = auth_module.register_patient(req.full_name, req.email, req.password, req.date_of_birth)
        return {"success": True, "user_id": user_id, "message": "Registration successful."}
    except Exception as e:
        raise HTTPException(status_code=400, detail="Email already registered or invalid payload.")

@app.post("/api/auth/login")
def login(req: LoginRequest):
    user = auth_module.authenticate_patient(req.email, req.password)
    if not user:
        raise HTTPException(status_code=401, detail="Invalid email or password.")
    return {
        "success": True,
        "user": {
            "id": user["id"],
            "full_name": user["full_name"],
            "email": user["email"],
            "date_of_birth": user["date_of_birth"]
        }
    }

@app.post("/api/predict")
def predict(req: PredictRequest):
    bundle = predict_module.load_model_bundle(MODEL_PATH)
    res = predict_module.predict_disease(req.symptoms, bundle, top_k=3)
    if req.user_id and res.get("predicted_disease"):
        auth_module.save_prediction(
            user_id=req.user_id,
            symptoms=res.get("recognized_symptoms", []),
            primary_condition=res.get("predicted_disease"),
            confidence=res.get("confidence", 0.0) or 0.0,
            top_3_json=json.dumps(res.get("top_k_predictions", []))
        )
    return res

@app.get("/api/history/{user_id}")
def history(user_id: int):
    records = auth_module.get_patient_history(user_id)
    return {"user_id": user_id, "history": records}
