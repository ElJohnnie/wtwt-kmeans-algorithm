from models.ml_algorithm import MLEngine

class MLService:
    def __init__(self):
        self.engine = MLEngine()

    def process_data(self, data):
        result = self.engine.predict(data)  # disparar algoritmo de machine learning
        return result
