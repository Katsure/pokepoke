from flask import Flask, request, jsonify
from werkzeug.utils import secure_filename
from PIL import Image
import torch
import torchvision.transforms as transforms
import torchvision.models as models
import os
import requests
from flask_cors import CORS
from dotenv import load_dotenv  # <-- NEW

# --- Load environment variables ---
load_dotenv()  # <-- NEW

# --- Configurations ---
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
CORS(app)

LLM_API_URL = "https://openrouter.ai/api/v1/chat/completions"
LLM_API_KEY = os.getenv("OPENROUTER_API_KEY")  # <-- NEW: from .env

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load a pretrained ResNet18 model for visual feature extraction
model = models.resnet18(pretrained=True).to(device)
model.eval()

# Preprocessing pipeline
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def extract_features(image_path: str) -> str:
    # Open and convert the image
    try:
        image = Image.open(image_path).convert("RGB")
        print(f"Image type: {type(image)}")  # Debug: Check initial type
    except Exception as e:
        raise ValueError(f"Failed to open image: {str(e)}")

    # Apply transformation
    try:
        tensor = transform(image)
        print(f"Transformed tensor type: {type(tensor)}")  # Debug: Check tensor type
        if not isinstance(tensor, torch.Tensor):
            raise ValueError(f"Transform output is not a tensor, got {type(tensor)}")
        print(f"Transformed tensor shape: {tensor.shape if hasattr(tensor, 'shape') else 'No shape attribute'}")  # Debug: Check shape
    except Exception as e:
        raise ValueError(f"Transformation failed: {str(e)}")

    # Ensure tensor is 4D (batch, channels, height, width)
    try:
        if len(tensor.shape) == 3:  # Add batch dimension if missing
            tensor = tensor.unsqueeze(0)
        elif len(tensor.shape) != 4:
            raise ValueError(f"Unexpected tensor shape: {tensor.shape}")
        tensor = tensor.to(device)
        print(f"Unsqueezed tensor shape: {tensor.shape}, device: {tensor.device}")  # Debug
    except Exception as e:
        raise ValueError(f"Tensor shaping failed: {str(e)}")

    with torch.no_grad():
        try:
            outputs = model(tensor)
            class_index = outputs.argmax().item()
        except Exception as e:
            raise ValueError(f"Model inference failed: {str(e)}")

    # Provide a human-like description from class index
    return f"Visual profile score: {class_index} (used for team personality match)"

def get_pokemon_team(description: str) -> str:
    prompt = f"""
Based on the visual description: "{description}", assign a Pokémon team to this trainer:
- The starter Pokémon can be any Pokémon, not limited to the classic starters.
- Choose exactly 5 other Pokémon that match their appearance or personality.
- Return ONLY the following format with no additional text, asterisks, or rationale:
Starter: [Name]
Team: [Name1, Name2, Name3, Name4, Name5]
"""

    headers = {
        "Authorization": f"Bearer {LLM_API_KEY}",
        "Content-Type": "application/json"
    }

    data = {
        "model": "deepseek/deepseek-r1:free",
        "messages": [
            {"role": "system", "content": "You are Professor Oak, the Pokémon Professor. Respond strictly in the requested format with no extra text."},
            {"role": "user", "content": prompt.strip()}
        ]
    }

    response = requests.post(LLM_API_URL, headers=headers, json=data)
    response.raise_for_status()
    return response.json()["choices"][0]["message"]["content"]

@app.route('/generate_team', methods=['POST'])
def generate_team():
    if 'image' not in request.files:
        return jsonify({"error": "No image uploaded"}), 400

    file = request.files['image']
    if file.filename == '' or not allowed_file(file.filename):
        return jsonify({"error": "Invalid image format"}), 400

    filename = secure_filename(file.filename or "uploaded.jpg")
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    try:
        visual_description = extract_features(filepath)
        team = get_pokemon_team(visual_description)
        return jsonify({
            "visual_description": visual_description,
            "team": team
        })
    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
