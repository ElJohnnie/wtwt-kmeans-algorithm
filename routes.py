from flask import request, jsonify
from services.ml_service import MLService

def setup_routes(app):
    @app.route('/ml', methods=['POST'])
    def ml():
        ml_service = MLService()
        response, status_code = ml_service.process_data(request.get_json())
        print(response)
        return jsonify(response), status_code
