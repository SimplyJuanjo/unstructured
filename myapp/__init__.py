from flask import Flask
from myapp.routes.main import main

def create_app():
    app = Flask(__name__)
    app.config.from_object('myapp.config.config.Config')
    app.register_blueprint(main)

    return app
