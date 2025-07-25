from flask import Blueprint, jsonify
from services.predictService import predict_future_sales

predict_bp = Blueprint('predict', __name__)

@predict_bp.route('/api/predict', methods=['GET', 'OPTIONS'])
def predict():
    forecast_data = predict_future_sales()
    return jsonify(forecast_data)
