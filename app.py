from flask import Flask
from dotenv import load_dotenv
import os
from routes import setup_routes

load_dotenv()

app = Flask(__name__)
setup_routes(app)

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default_secret')

def home():
    return f"Environment: {os.getenv('FLASK_ENV', 'unknown')}"

if __name__ == '__main__':
    debug = os.getenv('DEBUG', 'False') == 'True'
    port = int(os.getenv('PORT', 5050))
    app.run(debug=debug, port=port)
