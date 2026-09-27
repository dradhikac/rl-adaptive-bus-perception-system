"""
web_flask package initializer
=============================
Exposes the Flask app and SocketIO instance for testing and WSGI deployment.
"""

from .app import app, socketio, create_app

__all__ = ["app", "socketio", "create_app"]
