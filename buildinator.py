import os
import json
import sqlite3
import subprocess
import difflib
from flask import Flask, render_template, request, jsonify
from flask_sqlalchemy import SQLAlchemy
from llama_cpp import Llama
import docker
import tempfile
import logging
from queue import Queue
from threading import Thread

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app_builder.db"
db = SQLAlchemy(app)

# Configurable options
MAX_IDENTICAL_ITERATIONS = 3
DOCKER_IMAGE_PYTHON = "python:3.9-slim"
DOCKER_IMAGE_DOTNET = "mcr.microsoft.com/dotnet/sdk:6.0"
LOGGING_ENABLED = True

# Set up logging
if LOGGING_ENABLED:
    logging.basicConfig(level=logging.DEBUG)
else:
    logging.basicConfig(level=logging.INFO)

# Database models
class Iteration(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    app_name = db.Column(db.String(100), nullable=False)
    prompt = db.Column(db.Text, nullable=False)
    input_code = db.Column(db.Text, nullable=False)
    output_code = db.Column(db.Text, nullable=False)
    build_output = db.Column(db.Text, nullable=False)
    is_release_candidate = db.Column(db.Boolean, default=False)

    def __repr__(self):
        return f"Iteration('{self.app_name}', '{self.prompt}', '{self.id}')"

class App(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    prompt = db.Column(db.Text, nullable=False)
    input_code = db.Column(db.Text, nullable=False)
    language = db.Column(db.String(10), nullable=False)
    is_queued = db.Column(db.Boolean, default=False)

    def __repr__(self):
        return f"App('{self.name}', '{self.prompt}', '{self.id}')"

# LLM function
def run_llm(prompt, input_code, language):
    try:
        # Instantiate LLM for each iteration
        llama_params = {
            "n_threads": 0,
            "n_threads_batch": 0,
            "use_mmap": False,
            "use_mlock": False,
            "n_gpu_layers": 0,
            "main_gpu": 0,
            "tensor_split": "",
            "top_p": 0.95,
            "n_ctx": 131072,
            "rope_freq_base": 0,
            "numa": False,
            "verbose": True,
            "top_k": 40,
            "temperature": 0.8,
            "repeat_penalty": 1.01,
            "max_tokens": 65536,
            "typical_p": 0.68,
            "n_batch": 2048,
            "min_p": 0,
            "frequency_penalty": 0,
            "presence_penalty": 0.5
        }
        llama = Llama("./model/Mistral-Nemo-Instruct-2407-Q8_0.gguf", **llama_params)

        # Generate complete revision of code, addressing build errors, surrounded by triple backticks
        response = llama.create_completion( f"Generate ONLY a complete revision of the {language} code, addressing any build errors, surrounded by triple backticks:\\n```{input_code}```\\n{prompt}")
        output_code = response['choices'][0]['text'].split("```")[1].split("```")[0]
        return output_code
    except Exception as e:
        # Handle LLM call failure, return input code without throwing errors
        logging.error(f"LLM call failed: {e}")
        return input_code

# Docker execution function
def execute_code(code, language):
    client = docker.from_env()
    if language == "py":
        container = client.containers.run(
            DOCKER_IMAGE_PYTHON,
            command=f"bash -c 'echo \"{code}\" > main.py && python main.py'",
            detach=True,
            remove=True,
            stdout=True,
            stderr=True
        )
    elif language == "cs":
        container = client.containers.run(
            DOCKER_IMAGE_DOTNET,
            command=f"bash -c 'echo \"{code}\" > main.cs && dotnet run'",
            detach=True,
            remove=True,
            stdout=True,
            stderr=True
        )
    build_output = container.logs(stdout=True, stderr=True).decode("utf-8")
    return build_output

# Build queue
build_queue = Queue()

# Web interface routes
@app.route("/")
def index():
    apps = App.query.all()
    return render_template("index.html", apps=apps)

@app.route("/build", methods=["POST"])
def build():
    app_name = request.form["app_name"]
    prompt = request.form["prompt"]
    language = request.form["language"]
    input_code = request.form["input_code"]

    # Check if app already exists
    app = App.query.filter_by(name=app_name).first()
    if app:
        app.prompt = prompt
        app.input_code = input_code
        app.language = language
        app.is_queued = True
    else:
        app = App(name=app_name, prompt=prompt, input_code=input_code, language=language, is_queued=True)
        db.session.add(app)

    db.session.commit()

    # Add app to build queue
    build_queue.put(app)

    return jsonify({"message": "App added to build queue"})

@app.route("/queue")
def queue():
    apps = App.query.filter_by(is_queued=True).all()
    return render_template("queue.html", apps=apps)

@app.route("/delete_app/<int:app_id>")
def delete_app(app_id):
    app = App.query.get(app_id)
    if app:
        db.session.delete(app)
        db.session.commit()
    return jsonify({"message": "App deleted"})

@app.route("/delete_iteration/<int:iteration_id>")
def delete_iteration(iteration_id):
    iteration = Iteration.query.get(iteration_id)
    if iteration:
        db.session.delete(iteration)
        db.session.commit()
    return jsonify({"message": "Iteration deleted"})

@app.route("/remove_from_queue/<int:app_id>")
def remove_from_queue(app_id):
    app = App.query.get(app_id)
    if app:
        app.is_queued = False
        db.session.commit()
    return jsonify({"message": "App removed from queue"})

# Build worker
def build_worker():
    while True:
        app = build_queue.get()
        if app:
            # Run LLM
            output_code = run_llm(app.prompt, app.input_code, app.language)

            # Execute code in Docker
            build_output = execute_code(output_code, app.language)

            # Store iteration in database
            iteration = Iteration(
                app_name=app.name,
                prompt=app.prompt,
                input_code=app.input_code,
                output_code=output_code,
                build_output=build_output,
                is_release_candidate=(build_output.strip() == "")
            )
            db.session.add(iteration)
            db.session.commit()

            # Update app status
            app.is_queued = False
            db.session.commit()

            # Log build result
            logging.info(f"Build result for {app.name}: {build_output}")

        build_queue.task_done()

# Start build worker
build_thread = Thread(target=build_worker)
build_thread.daemon = True
build_thread.start()

if __name__ == "__main__":
    with app.app_context():
        db.create_all()
    app.run(debug=True)
