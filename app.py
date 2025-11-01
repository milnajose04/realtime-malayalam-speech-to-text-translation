from flask import Flask, request, jsonify
from flask_cors import CORS
import whisper
import tempfile
import os

app = Flask(__name__)
CORS(app)  # Allow frontend to connect

# Load Whisper model once at startup
MODEL_SIZE = "large"
print("Loading Whisper model (this may take a while)...")
model = whisper.load_model(MODEL_SIZE)

@app.route('/transcribe', methods=['POST'])
def transcribe():
    if 'audio' not in request.files:
        return jsonify({'error': 'No audio file uploaded.'}), 400

    audio_file = request.files['audio']
    with tempfile.NamedTemporaryFile(delete=False, suffix='.wav') as temp:
        audio_file.save(temp.name)
        temp_path = temp.name

    try:
        result = model.transcribe(
            temp_path,
            language="ml",
            task="transcribe",
            initial_prompt="നമസ്കാരം",
            fp16=False
        )
        text = result['text'].strip()
        print(f"Transcription: {text}")
        return jsonify({'transcription': text})
    except Exception as e:
        return jsonify({'error': str(e)})
    finally:
        os.remove(temp_path)
@app.route('/save_transcription', methods=['POST'])
def save_transcription():
    data = request.json
    edited_text = data.get('edited_text')
    
    # Example: save edited transcription to a file
    with open("saved_transcription.txt", "w", encoding="utf-8") as f:
        f.write(edited_text)
    
    return jsonify({"message": "Transcription saved successfully!"})

if __name__ == "__main__":
    app.run(debug=True)