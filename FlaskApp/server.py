from flask import Flask, jsonify, request
import subprocess
import json
from flask_cors import CORS
import os

app = Flask(__name__)
CORS(app)


@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "Welcome to the Wine Testing API!"}), 200


@app.route("/run-test", methods=["POST"])
def run_test():
    data = request.get_json()
    wine_name = data.get("wine_name", "").strip()

    if not wine_name:
        return jsonify({"error": "Wine name is required."}), 400

    # Define the path to test.py
    test_script_path = os.path.join(
        "C:/Users/aidan/codeprojects/ML/ArduinoWineSniffer/ML/WineSnifferDeepNetworkModel/PCAModel",
        "test.py",
    )

    # Check if test.py exists
    if not os.path.isfile(test_script_path):
        return jsonify({"error": f"test.py not found at {test_script_path}"}), 500

    try:
        # Run the test.py script with the provided wine name
        result = subprocess.run(
            [
                "python",
                test_script_path,
                wine_name,
            ],
            check=True,
            text=True,  # Capture output as text
            capture_output=True,
            timeout=60,  # Prevent hanging indefinitely
        )

        # Attempt to parse the output as JSON
        predictions = json.loads(result.stdout)
        return jsonify(predictions), 200

    except subprocess.TimeoutExpired:
        return jsonify({"error": "Script execution timed out."}), 500
    except subprocess.CalledProcessError as e:
        # Attempt to parse the error output as JSON
        try:
            error_output = json.loads(e.stderr)
            return jsonify({"error": error_output.get("error", "Script failed.")}), 500
        except json.JSONDecodeError:
            return jsonify({"error": f"Script error: {e.stderr}"}), 500
    except json.JSONDecodeError:
        return jsonify({"error": "Failed to decode JSON from script output."}), 500
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    # It's recommended to run Flask with a production-ready server in production
    app.run(host="0.0.0.0", port=5000, debug=True)
