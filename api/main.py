from flask import Flask, request, jsonify
from flask_cors import CORS
import requests

app = Flask(__name__)
CORS(app)

API_URL = "https://api-inference.huggingface.co/models/Salesforce/blip-image-captioning-base"
HEADERS = {
    "Authorization": "Bearer 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",
    "Content-Type": "application/octet-stream"
}

def query_huggingface_api(image_data):
    response = requests.post(API_URL, headers=HEADERS, data=image_data)
    if response.status_code != 200:
        raise Exception(f"Failed to query the API: {response.status_code} - {response.text}")
    return response.json()

@app.route('/caption', methods=['POST'])
def caption_image():
    if 'image' not in request.files:
        return jsonify({"error": "No image provided"}), 400

    image_file = request.files['image']
    image_data = image_file.read()

    try:
        result = query_huggingface_api(image_data)
        if 'error' in result:
            return jsonify({"error": result['error']}), 500
        return jsonify({"caption": result[0]['generated_text']})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
