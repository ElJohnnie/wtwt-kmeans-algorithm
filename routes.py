from flask import request
from services.ml_service import MLService

def setup_routes(app):
    @app.route('/ml', methods=['GET'])
    def ml_route():
        ml_service = MLService()
        response = ml_service.process_data(request.get_json())
        return response
