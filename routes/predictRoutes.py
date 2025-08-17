from datetime import datetime
from flask import Blueprint, jsonify, request
from services.predictService import forecast_items_qty_sold, predict_future_sales

predict_bp = Blueprint('predict', __name__)

@predict_bp.route('/api/predict', methods=['GET', 'OPTIONS'])
def predict():
    forecast_data = predict_future_sales()
    return jsonify(forecast_data)

@predict_bp.route("/api/predict/items", methods=["GET"])
def predict_items():
    try:
        # Get query params with default values
        month = request.args.get("month", default=datetime.now().month, type=int)
        year = request.args.get("year", default=datetime.now().year, type=int)

        # Call your forecasting function
        forecast_data = forecast_items_qty_sold(month, year)

        return jsonify(forecast_data)
    except Exception as e:
        print("Error:", str(e))
        return jsonify({"error": "Internal Server Error"}), 500