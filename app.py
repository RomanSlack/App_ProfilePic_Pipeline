import logging

from flask import Flask, render_template, request, jsonify

from processing import process_image, init_models

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB upload limit

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

_models_ready = False


@app.before_request
def ensure_models():
    global _models_ready
    if not _models_ready:
        logger.info("Initializing models (first request)...")
        init_models()
        _models_ready = True
        logger.info("Models ready.")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/process", methods=["POST"])
def api_process():
    if "image" not in request.files:
        return jsonify({"error": "No image file provided."}), 400

    file = request.files["image"]
    if file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    try:
        image_bytes = file.read()
        result = process_image(image_bytes)
        return jsonify(result)
    except ValueError as e:
        return jsonify({"error": str(e)}), 422
    except Exception:
        logger.exception("Processing failed")
        return jsonify({"error": "Processing failed. Please try another photo."}), 500


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
