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

app = Flask(__name__)
app.config["SQLALCHEMY_DATABASE_URI"] = "sqlite:///app_builder.db"
db = SQLAlchemy(app)

# Configurable options
MAX_IDENTICAL_ITERATIONS = 3
DOCKER_IMAGE = "python:3.9-slim"  # or "mcr.microsoft.com/dotnet/core/sdk:3.1" for C#

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
        response = llama.create_completion( f"Generate ONLY a complete revision of the {language} code, addressing any build errors, surrounded by triple backticks:\n```{input_code}```\n{prompt}")
        output_code = response['choices'][0]['text'].split("```")[1].split("```")[0]
        return output_code
    except Exception as e:
        # Handle LLM call failure, return input code without throwing errors
        print(f"LLM call failed: {e}")
        return input_code

# Docker execution function
def execute_code(code, language):
    client = docker.from_env()
    container = client.containers.run(
        DOCKER_IMAGE,
        command=f"bash -c 'echo \"{code}\" > main.{language} && ./main.{language}'",
        detach=True,
        remove=True,
        stdout=True,
        stderr=True
    )
    build_output = container.logs(stdout=True, stderr=True).decode("utf-8")
    return build_output

# Web interface routes
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/build", methods=["POST"])
def build():
    app_name = request.form["app_name"]
    prompt = request.form["prompt"]
    language = request.form["language"]
    input_code = request.form["input_code"]

    # Check for identical iterations
    identical_iterations = 0
    last_iteration = Iteration.query.filter_by(app_name=app_name).order_by(Iteration.id.desc()).first()
    if last_iteration:
        if last_iteration.input_code == input_code:
            identical_iterations += 1
        if identical_iterations >= MAX_IDENTICAL_ITERATIONS:
            return jsonify({"error": "Too many identical iterations. Please revise your prompt."})

    # Run LLM
    output_code = run_llm(prompt, input_code, language)

    # Execute code in Docker
    build_output = execute_code(output_code, language)

    # Store iteration in database
    iteration = Iteration(
        app_name=app_name,
        prompt=prompt,
        input_code=input_code,
        output_code=output_code,
        build_output=build_output,
        is_release_candidate=(build_output.strip() == "")
    )
    db.session.add(iteration)
    db.session.commit()

    return jsonify({"output_code": output_code, "build_output": build_output, "is_release_candidate": iteration.is_release_candidate})

@app.route("/iterations/<app_name>")
def iterations(app_name):
    iterations = Iteration.query.filter_by(app_name=app_name).order_by(Iteration.id.asc()).all()
    return render_template("iterations.html", iterations=iterations, app_name=app_name)

@app.route("/diff/<app_name>/<int:iteration_id>")
def diff(app_name, iteration_id):
    iteration = Iteration.query.get(iteration_id)
    previous_iteration = Iteration.query.filter_by(app_name=app_name).filter(Iteration.id < iteration_id).order_by(Iteration.id.desc()).first()
    if previous_iteration:
        diff = difflib.unified_diff(previous_iteration.output_code.splitlines(), iteration.output_code.splitlines())
        return jsonify({"diff": "\\n".join(diff)})
    else:
        return jsonify({"error": "No previous iteration found."})

if __name__ == "__main__":
    db.create_all()
    app.run(debug=True)
