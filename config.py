import os

class Config:
    DOCKER_HOST = os.environ.get('DOCKER_HOST', 'localhost')
    LLM_API = os.environ.get('LLM_API', 'local')
    STRIPE_SECRET_KEY = os.environ.get('STRIPE_SECRET_KEY', '')
    STRIPE_PUBLISHABLE_KEY = os.environ.get('STRIPE_PUBLISHABLE_KEY', '')
    ENABLED = os.environ.get('ENABLED', 'False') == 'True'
    MAX_IDENTICAL_ITERATIONS = 3
    DOCKER_IMAGE_PYTHON = "python:3.9-slim"
    DOCKER_IMAGE_DOTNET = "mcr.microsoft.com/dotnet/sdk:6.0"
    LOGGING_ENABLED = True
