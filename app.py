from flask import Flask
from routes.predictRoutes import predict_bp
from flask_cors import CORS
import os

app = Flask(__name__)

# CORS config
CORS(app, resources={
    r"/api/predict/*": {
        "origins": [
            "http://localhost:5173",
        ],
        "methods": ["GET", "POST", "OPTIONS"],
        "allow_headers": ["Content-Type", "Authorization"]
    }
})

# Register blueprint
app.register_blueprint(predict_bp)

# This block only runs locally, not in production
if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))
    app.run(host='0.0.0.0', port=port, debug=True)
