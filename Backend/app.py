import os
import sys
import tempfile
import traceback
from flask import Flask, request, jsonify
from flask_cors import CORS
import joblib
from pydub import AudioSegment
from extract_features_parselmouth import extract_from_wav


app = Flask(__name__)
CORS(app)


# Setup logging
import logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
    ]
)
logger = logging.getLogger(__name__)


MODEL_FILE = 'parkinsons_model.pkl'  
if not os.path.exists(MODEL_FILE):
    raise RuntimeError(f"Model file not found: {MODEL_FILE}")


obj = joblib.load(MODEL_FILE)
model = obj['model']
scaler = obj['scaler']


# UPDATED: Only 6 features that the model was trained on
FEATURE_ORDER = [
    "MDVP:Fo(Hz)",
    "MDVP:Fhi(Hz)",
    "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)",
    "MDVP:Shimmer",
    "HNR"
]


def convert_to_wav(input_path, out_path, target_sr=44100):
    """Convert audio to standardized WAV format."""
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_frame_rate(target_sr).set_channels(1).set_sample_width(2) 
    audio.export(out_path, format="wav")


@app.route('/', methods=['GET'])
def index():
    """Root endpoint."""
    return jsonify({
        "service": "Parkinson's Voice Detector",
        "version": "1.0",
        "endpoints": {
            "POST /predict": "Send audio file for prediction",
            "GET /health": "Health check"
        }
    }), 200


@app.route('/health', methods=['GET'])
def health():
    """Health check endpoint."""
    logger.info("Health check requested")
    return jsonify({"status": "ok", "model": "loaded"}), 200


@app.route('/predict', methods=['POST'])
def predict():
    """Predict Parkinson's from audio."""
    logger.info("Received prediction request")
    
    if 'audio' not in request.files:
        logger.error("No audio file in request")
        return jsonify({"error": "No audio file provided"}), 400


    uploaded = request.files['audio']
    logger.info(f"Processing file: {uploaded.filename}")


    # Save uploaded file to temp location
    tmp_fd, tmp_path = tempfile.mkstemp(suffix=os.path.splitext(uploaded.filename or "upload")[1] or '.tmp')
    os.close(tmp_fd)
    uploaded.save(tmp_path)
    logger.info(f"Uploaded file saved to: {tmp_path}")


    # Convert to WAV
    tmp_wav_fd, tmp_wav_path = tempfile.mkstemp(suffix='.wav')
    os.close(tmp_wav_fd)
    
    try:
        logger.info("Converting to WAV...")
        convert_to_wav(tmp_path, tmp_wav_path, target_sr=44100)
        logger.info("Conversion successful")
    except Exception as e:
        logger.error(f"Audio conversion failed: {str(e)}")
        logger.error(traceback.format_exc())
        try:
            os.remove(tmp_path)
        except Exception:
            pass
        return jsonify({
            "error": f"Failed to convert audio to WAV: {str(e)}", 
            "trace": traceback.format_exc()
        }), 500


    try:
        logger.info("Extracting features...")
        feats = extract_from_wav(tmp_wav_path)
        logger.info(f"Features extracted: {feats.keys()}")
        
        # Check for errors in feature extraction
        if 'error' in feats:
            logger.error(f"Feature extraction error: {feats['error']}")
            return jsonify({"error": feats['error']}), 400


        duration = feats.get('_meta_duration', 0.0)
        if duration < 0.8:
            logger.warning(f"Audio too short: {duration:.3f}s")
            return jsonify({
                "error": f"Audio too short ({duration:.3f}s). Please record at least 1 second."
            }), 400


        # Build feature vector with ONLY the 6 features the model expects
        row = [feats.get(col, 0.0) for col in FEATURE_ORDER]
        logger.info(f"Feature vector: {row}")
        
        # Transform using raw array (not DataFrame) to avoid feature name validation issues
        X_scaled = scaler.transform([row])
        
        # Predict
        prob = float(model.predict_proba(X_scaled)[0, 1]) if hasattr(model, "predict_proba") else None
        pred = int(model.predict(X_scaled)[0])
        message = "High likelihood of Parkinson's" if pred == 1 else "Low likelihood of Parkinson's"


        logger.info(f"Prediction: {pred}, Probability: {prob}")
        
        return jsonify({
            "prediction": pred, 
            "probability": prob, 
            "message": message, 
            "meta": {
                "duration": feats.get('_meta_duration'), 
                "samplerate": feats.get('_meta_samplerate')
            }
        }), 200
        
    except Exception as e:
        logger.error(f"Prediction failed: {str(e)}")
        logger.error(traceback.format_exc())
        tb = traceback.format_exc()
        return jsonify({
            "error": f"Extraction/prediction failed: {str(e)}", 
            "trace": tb
        }), 500
        
    finally:
        for p in (tmp_path, tmp_wav_path):
            try:
                os.remove(p)
                logger.info(f"Cleaned up: {p}")
            except Exception as e:
                logger.warning(f"Cleanup failed for {p}: {str(e)}")


if __name__ == "__main__":
    logger.info("Starting Flask app...")
    app.run(host='0.0.0.0', port=8000, debug=True)
