from models.ml_algorithm import MLEngine

class MLService:
    def __init__(self):
        self.engine = MLEngine()
    
    def validate_data(self, data):
        required_fields = ['mood', 'primaryGenre', 'secondaryGenre', 'epoch']
        moods = ['Feliz', 'Triste', 'Ansioso', 'Animado', 'Entediado']
        genres = ['Action', 'Animation', 'Comedy', 'Drama', 'Fantasy', 'Sci-fi', 'Thriller', 'Horror']

        for field in required_fields:
            if field not in data:
                return False, f"Campo '{field}' ausente no body da requisição."
        
        if data['mood'] not in moods:
            return False, "Valor inválido para o campo 'mood'."
        if data['primaryGenre'] not in genres:
            return False, "Valor inválido para o campo 'primaryGenre'."
        if data['secondaryGenre'] not in genres:
            return False, "Valor inválido para o campo 'secondaryGenre'."
        if not isinstance(data['epoch'], int):
            return False, "O campo 'epoch' deve ser um número inteiro."

        return True, ""

    def process_data(self, data):
        is_valid, message = self.validate_data(data)
        if not is_valid:
            return {"message": message, "data": None}, 400  # Retorna mensagem de erro e status HTTP 400

        mood = data.get('mood')
        primary_genre = data.get('primaryGenre')
        secondary_genre = data.get('secondaryGenre')
        epoch = data.get('epoch')
        result = self.engine.predict(mood, primary_genre, secondary_genre, epoch)
        return result, 200
