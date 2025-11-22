"""
Real-Time Model Serving API

FastAPI application for serving device failure predictions in real-time.
Loads production model from MLflow Model Registry.

Endpoints:
- POST /predict/device-failure: Predict device failure risk
- GET /health: Health check
- GET /model-info: Model metadata

Author: MLOps Team
Date: 2025-11-22
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
import mlflow
import mlflow.pyfunc
import pandas as pd
import numpy as np
from datetime import datetime
import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="LIMS MLOps API",
    description="Real-time predictions for lab device failure and outbreak risk",
    version="1.0.0",
)

# Global model cache
MODEL_CACHE = {}


class DeviceFeatures(BaseModel):
    """Input features for device failure prediction."""

    device_id: str = Field(..., description="Unique device identifier")
    calibration_overdue: bool = Field(..., description="Calibration is overdue (>30 days)")
    maintenance_overdue: bool = Field(..., description="Maintenance is overdue (>90 days)")
    calibration_overdue_days: int = Field(0, ge=0, description="Days calibration is overdue")
    maintenance_overdue_days: int = Field(0, ge=0, description="Days maintenance is overdue")
    error_rate_7d: float = Field(0.0, ge=0, le=1, description="7-day rolling error rate")
    warning_rate_7d: float = Field(0.0, ge=0, le=1, description="7-day rolling warning rate")
    qc_pass_rate_7d: float = Field(1.0, ge=0, le=1, description="7-day QC pass rate")
    result_volume_trend: float = Field(0.0, description="Result volume trend (normalized)")
    avg_turnaround_time_7d: float = Field(0.0, ge=0, description="Avg turnaround time (hours)")
    result_volume: int = Field(0, ge=0, description="Daily result volume")

    class Config:
        schema_extra = {
            "example": {
                "device_id": "DEVICE_123",
                "calibration_overdue": True,
                "maintenance_overdue": False,
                "calibration_overdue_days": 45,
                "maintenance_overdue_days": 0,
                "error_rate_7d": 0.08,
                "warning_rate_7d": 0.12,
                "qc_pass_rate_7d": 0.85,
                "result_volume_trend": -0.15,
                "avg_turnaround_time_7d": 2.5,
                "result_volume": 150,
            }
        }


class PredictionResponse(BaseModel):
    """Response model for predictions."""

    device_id: str
    failure_probability: float = Field(..., ge=0, le=1, description="Probability of failure (0-1)")
    risk_score: int = Field(..., ge=0, le=100, description="Risk score (0-100)")
    risk_level: str = Field(..., description="Risk level: LOW, MEDIUM, HIGH, CRITICAL")
    recommendation: str = Field(..., description="Recommended action")
    prediction_timestamp: str = Field(..., description="Timestamp of prediction (ISO format)")


class ModelInfo(BaseModel):
    """Model metadata."""

    model_name: str
    model_version: str
    model_stage: str
    loaded_at: str
    mlflow_run_id: Optional[str]


def load_model(model_name: str, stage: str = "Production") -> mlflow.pyfunc.PyFuncModel:
    """
    Load model from MLflow Model Registry.

    Args:
        model_name: Name of registered model
        stage: Model stage (Production, Staging, None)

    Returns:
        Loaded MLflow model
    """
    cache_key = f"{model_name}_{stage}"

    # Check cache
    if cache_key in MODEL_CACHE:
        logger.info(f"Using cached model: {cache_key}")
        return MODEL_CACHE[cache_key]

    # Load from MLflow
    logger.info(f"Loading model: {model_name} (stage: {stage})")

    model_uri = f"models:/{model_name}/{stage}"
    model = mlflow.pyfunc.load_model(model_uri)

    # Cache model
    MODEL_CACHE[cache_key] = model

    logger.info(f"Model loaded successfully: {cache_key}")
    return model


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    logger.info("Starting LIMS MLOps API...")

    # Set MLflow tracking URI (configured via environment variable)
    mlflow_uri = os.getenv("MLFLOW_TRACKING_URI", "databricks")
    mlflow.set_tracking_uri(mlflow_uri)

    # Pre-load production model
    try:
        load_model("device_failure_predictor", stage="Production")
        logger.info("Production model loaded successfully")
    except Exception as e:
        logger.error(f"Failed to load production model: {e}")
        logger.warning("API will attempt to load model on first request")


@app.get("/")
async def root():
    """Root endpoint."""
    return {
        "service": "LIMS MLOps API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "predict": "/predict/device-failure",
            "health": "/health",
            "model_info": "/model-info",
        },
    }


@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models_loaded": len(MODEL_CACHE),
    }


@app.get("/model-info", response_model=ModelInfo)
async def get_model_info():
    """Get information about the production model."""
    try:
        model_name = "device_failure_predictor"
        stage = "Production"

        # Get model metadata from MLflow
        client = mlflow.tracking.MlflowClient()
        model_version = client.get_latest_versions(model_name, stages=[stage])[0]

        return ModelInfo(
            model_name=model_name,
            model_version=model_version.version,
            model_stage=stage,
            loaded_at=datetime.now().isoformat(),
            mlflow_run_id=model_version.run_id,
        )
    except Exception as e:
        logger.error(f"Error getting model info: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/predict/device-failure", response_model=PredictionResponse)
async def predict_device_failure(features: DeviceFeatures) -> PredictionResponse:
    """
    Predict device failure risk.

    Args:
        features: Device features

    Returns:
        Prediction with risk score and recommendations
    """
    try:
        logger.info(f"Prediction request for device: {features.device_id}")

        # Load model
        model = load_model("device_failure_predictor", stage="Production")

        # Prepare features
        feature_df = pd.DataFrame(
            [
                {
                    "calibration_overdue": int(features.calibration_overdue),
                    "maintenance_overdue": int(features.maintenance_overdue),
                    "calibration_overdue_days": features.calibration_overdue_days,
                    "maintenance_overdue_days": features.maintenance_overdue_days,
                    "error_rate_7d": features.error_rate_7d,
                    "warning_rate_7d": features.warning_rate_7d,
                    "qc_pass_rate_7d": features.qc_pass_rate_7d,
                    "result_volume_trend": features.result_volume_trend,
                    "avg_turnaround_time_7d": features.avg_turnaround_time_7d,
                    "result_volume": features.result_volume,
                }
            ]
        )

        # Make prediction
        prediction = model.predict(feature_df)[0]
        probability = model._model_impl.predict_proba(feature_df)[0][1]  # Probability of class 1

        # Calculate risk score (0-100)
        risk_score = int(probability * 100)

        # Determine risk level
        if risk_score >= 80:
            risk_level = "CRITICAL"
            recommendation = "IMMEDIATE ACTION REQUIRED: Schedule emergency maintenance and calibration. Consider taking device offline."
        elif risk_score >= 60:
            risk_level = "HIGH"
            recommendation = "Schedule urgent maintenance and calibration within 24 hours. Monitor device closely."
        elif risk_score >= 30:
            risk_level = "MEDIUM"
            recommendation = "Schedule maintenance and calibration within 7 days. Increase monitoring frequency."
        else:
            risk_level = "LOW"
            recommendation = "Continue normal operations. Follow standard maintenance schedule."

        # Log prediction
        logger.info(
            f"Device {features.device_id}: Risk={risk_level} ({risk_score}%), Probability={probability:.3f}"
        )

        return PredictionResponse(
            device_id=features.device_id,
            failure_probability=round(probability, 4),
            risk_score=risk_score,
            risk_level=risk_level,
            recommendation=recommendation,
            prediction_timestamp=datetime.now().isoformat(),
        )

    except Exception as e:
        logger.error(f"Prediction error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")


@app.post("/predict/device-failure/batch", response_model=List[PredictionResponse])
async def predict_device_failure_batch(
    features_list: List[DeviceFeatures],
) -> List[PredictionResponse]:
    """
    Batch prediction for multiple devices.

    Args:
        features_list: List of device features

    Returns:
        List of predictions
    """
    logger.info(f"Batch prediction request for {len(features_list)} devices")

    predictions = []
    for features in features_list:
        try:
            pred = await predict_device_failure(features)
            predictions.append(pred)
        except Exception as e:
            logger.error(f"Error predicting for device {features.device_id}: {e}")
            # Continue with other devices

    return predictions


# For local development and testing
if __name__ == "__main__":
    import uvicorn

    # Run server
    uvicorn.run(
        "api_server:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        log_level="info",
    )
