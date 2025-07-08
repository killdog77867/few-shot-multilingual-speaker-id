import os
import numpy as np
import torch
import json
from flask import Flask, render_template, request, jsonify, session, redirect, url_for, flash
from werkzeug.utils import secure_filename
from scipy.spatial.distance import cosine
import io
import traceback

# Import your custom modules
from preprocess import preprocess_audio_from_bytes, SAMPLE_RATE
from ecapa_tdnn import ECAPA_TDNN # Ensure class defaults/uses spkrec model

# --- Configuration ---
app = Flask(__name__)
app.config['SECRET_KEY'] = os.urandom(24)
app.config['EMBEDDING_DIR'] = "saved_embeddings"
app.config['PRETRAINED_MODEL_DIR'] = "pretrained_models"
app.config['USER_DATA_FILE'] = "user_data.json"

# !!! ADJUST THIS THRESHOLD BASED ON TESTING - critical for this approach !!!
# The threshold determines the maximum allowed distance for a "match".
# Start with a value like 0.4 and test EXTENSIVELY with different speakers.
app.config['COSINE_THRESHOLD'] = 0.4 # <<<< TUNE THIS VALUE CAREFULLY
print(f"!!! Speaker Verification Threshold: {app.config['COSINE_THRESHOLD']}. Tuning Required! !!!")

app.config['DEVICE'] = "cuda" if torch.cuda.is_available() else "cpu"

# --- Language Configuration (Keep as before) ---
SUPPORTED_LANGUAGES = {"en": "English", "hi": "Hindi (हिन्दी)", "ta": "Tamil (தமிழ்)"}
LANGUAGE_PROMPTS = { # Tuples: (Enroll1, Enroll2, Login)
    "en": ("The weather is lovely today, perfect for a walk.", "Reading books opens up doors to new worlds.", "Music often brings joy and lifts the spirit."),
    "hi": ("आज मौसम बहुत सुहावना है, टहलने के लिए बढ़िया।", "किताबें पढ़ना नई दुनिया के दरवाजे खोलता है।", "संगीत अक्सर खुशी लाता है और आत्मा को उत्साहित करता है।"),
    "ta": ("இன்று வானிலை மிகவும் அழகாக இருக்கிறது, நடைப்பயிற்சிக்கு ஏற்றது.", "புத்தகங்கள் வாசிப்பது புதிய உலகங்களுக்கான கதவுகளைத் திறக்கிறது.", "இசை பெரும்பாலும் மகிழ்ச்சியைத் தருகிறது மற்றும் உற்சாகமூட்டுகிறது.")
}

# Create necessary directories
os.makedirs(app.config['EMBEDDING_DIR'], exist_ok=True)
os.makedirs(app.config['PRETRAINED_MODEL_DIR'], exist_ok=True)

# --- Initialize Model (Keep as before - ensure using spkrec model) ---
try:
    model = ECAPA_TDNN(
        model_source="speechbrain/spkrec-ecapa-voxceleb", # <<< Speaker Recognition
        device=app.config['DEVICE'],
        savedir=app.config['PRETRAINED_MODEL_DIR']
    )
    EMBEDDING_DIM = model.embedding_size # Should be 192
    print(f"App confirmed embedding dimension: {EMBEDDING_DIM}")
    if EMBEDDING_DIM != 192:
        print(f"ERROR: App expected EMBEDDING_DIM 192, got {EMBEDDING_DIM}!")
        exit()
except Exception as model_init_error:
    print(f"FATAL: Could not initialize model: {model_init_error}")
    exit()

# --- User Metadata Handling (Keep load/save as before) ---
def load_user_data():
    filepath = app.config['USER_DATA_FILE']
    if not os.path.exists(filepath): return {}
    try:
        with open(filepath, 'r', encoding='utf-8') as f: data = json.load(f)
        if not isinstance(data, dict): return {}
        return data
    except Exception as e: print(f"Error loading user data: {e}"); return {}

def save_user_data(data):
    filepath = app.config['USER_DATA_FILE']
    try:
        with open(filepath, 'w', encoding='utf-8') as f: json.dump(data, f, indent=4, ensure_ascii=False)
    except Exception as e: print(f"Error saving user data: {e}")

# --- Helper Function (get_enrolled_speakers_embeddings - Keep as before) ---
def get_enrolled_speakers_embeddings():
    speakers = {}
    user_data = load_user_data()
    for username, data in user_data.items():
        if 'embedding_file' in data:
            path = os.path.join(app.config['EMBEDDING_DIR'], data['embedding_file'])
            embedding = model.load_embedding(path) #load_embedding does shape check
            if embedding is not None: speakers[username] = embedding
            else: print(f"Could not load embedding file for {username} despite existing metadata.") # Should be handeled within load
    print(f"Loaded {len(speakers)} valid speaker embeddings (dim={EMBEDDING_DIM}).")
    return speakers

# --- Routes (/, enroll GET, login GET, dashboard, logout - Keep as before) ---
@app.route('/')
def index():
    if 'logged_in' in session: return redirect(url_for('dashboard'))
    return redirect(url_for('enroll'))

@app.route('/enroll', methods=['GET'])
def enroll():
     if 'logged_in' in session: return redirect(url_for('dashboard'))
     return render_template('enroll.html', languages=SUPPORTED_LANGUAGES, prompts_json=json.dumps(LANGUAGE_PROMPTS))

@app.route('/login', methods=['GET'])
def login():
    if 'logged_in' in session: return redirect(url_for('dashboard'))
    return render_template('login.html', languages=SUPPORTED_LANGUAGES, prompts_json=json.dumps(LANGUAGE_PROMPTS))

@app.route('/dashboard')
def dashboard():
    if 'logged_in' not in session: flash("Please log in.", "warning"); return redirect(url_for('login'))
    username=session.get('username','User'); lang_code=session.get('language','en'); lang_name=SUPPORTED_LANGUAGES.get(lang_code,'Unknown')
    return render_template('dashboard.html', username=username, language_name=lang_name)

@app.route('/logout')
def logout():
    session.pop('logged_in', None); session.pop('username', None); session.pop('language', None)
    flash("Logged out.", "info"); return redirect(url_for('login'))


# --- Routes (process_enrollment - Keep as before) ---
@app.route('/process_enrollment', methods=['POST'])
def process_enrollment():
    # ...(Keep robust enrollment logic)...
    if 'audio_data' not in request.files or 'username' not in request.form or 'language' not in request.form: return jsonify({"status":"error", "message":"Missing fields."}), 400
    audio_file=request.files['audio_data']; username=request.form['username'].strip().lower(); language=request.form['language'].strip()
    if not username: return jsonify({"status":"error", "message":"Username empty."}), 400
    safe_username=secure_filename(username)
    if not safe_username: return jsonify({"status":"error", "message":"Invalid username."}), 400
    if language not in SUPPORTED_LANGUAGES: return jsonify({"status":"error", "message":"Invalid language."}), 400
    user_data=load_user_data()
    if safe_username in user_data: return jsonify({"status":"error", "message":"Username exists."}), 400
    embedding_filename=f"{safe_username}.npy"; embedding_path=os.path.join(app.config['EMBEDDING_DIR'], embedding_filename)
    try:
        audio_bytes=audio_file.read(); print(f"Enrollment audio: {len(audio_bytes)} bytes")
        audio_waveform, sr=preprocess_audio_from_bytes(audio_bytes)
        if audio_waveform is None: raise ValueError("Preprocessing failed.")
        min_duration_sec=1.0
        if len(audio_waveform) < SAMPLE_RATE * min_duration_sec: return jsonify({"status":"error", "message":"Audio too short."}), 400
        print(f"Extracting embedding for {safe_username}...")
        embedding=model.extract_embedding(audio_waveform, sr) # model
        if embedding is None: raise ValueError("Embedding extraction failed.")
        model.save_embedding(embedding, embedding_path)
        user_data[safe_username]={"embedding_file": embedding_filename, "language": language}
        save_user_data(user_data); print(f"User '{safe_username}' enrolled.")
        return jsonify({"status":"success", "message":f"User '{username}' enrolled."})
    except ValueError as ve: print(f"Enroll ValueError: {ve}"); return jsonify({"status":"error", "message":f"Enrollment failed: {ve}"}), 500
    except Exception as e: print(f"Enroll Exception: {e}"); traceback.print_exc(); return jsonify({"status":"error", "message":"Internal error."}), 500

@app.route('/process_login', methods=['POST'])
def process_login():
    # Ensure audio data is present
    if 'audio_data' not in request.files:
        return jsonify({"status": "error", "message": "Missing audio data."}), 400

    audio_file = request.files['audio_data']

    # Load enrolled speakers
    enrolled_embeddings = get_enrolled_speakers_embeddings()
    if not enrolled_embeddings:
         return jsonify({"status": "error", "message": "No users enrolled yet. Please enroll first."}), 400

    try:
        # Process Audio
        audio_bytes = audio_file.read()
        print(f"Received login audio: {len(audio_bytes)} bytes")
        audio_waveform, sr = preprocess_audio_from_bytes(audio_bytes)
        if audio_waveform is None: raise ValueError("Audio preprocessing failed.")
        min_duration_sec = 1.0
        if len(audio_waveform) < SAMPLE_RATE * min_duration_sec: return jsonify({"status": "error", "message": "Audio recording is too short."}), 400

        # Extract Embedding
        print("Extracting embedding for login attempt...")
        login_embedding = model.extract_embedding(audio_waveform, sr)
        if login_embedding is None: raise ValueError("Embedding extraction failed.")

        # --- Modified LOGIN LOGIC - Speaker Identification ---
        min_distance = float('inf')
        best_match_user = None
        all_distances = {}

        print(f"--- LOGIN: Comparing against enrolled users (Threshold = {app.config['COSINE_THRESHOLD']:.3f}) ---")
        for username, enrolled_embedding in enrolled_embeddings.items():
            try:
                dist = cosine(login_embedding.astype(np.float32), enrolled_embedding.astype(np.float32))
                all_distances[username] = dist # Store distance for logging
                print(f"    Compared with '{username}': Distance = {dist:.4f}")
                if dist < min_distance:
                    min_distance = dist
                    best_match_user = username
            except Exception as calc_e: print(f"    Distance calculation error for {username}: {calc_e}")

        threshold = app.config['COSINE_THRESHOLD']
        # The speaker IS recognized
        if best_match_user and min_distance <= threshold:
            session['logged_in'] = True
            session['username'] = best_match_user
            user_data = load_user_data()
            session['language'] = user_data.get(best_match_user, {}).get('language', 'en')

            print(f"✅ SUCCESS: Recognized user '{best_match_user}' (Distance = {min_distance:.4f} <= Threshold = {threshold:.3f})")
            return jsonify({
                "status": "success",
                "message": f"Welcome, {best_match_user}!", # Note: change this welcome message!
                "user": best_match_user,
                "distance": f"{min_distance:.4f}"
            })
        # The speaker is NOT recognized
        else:
            print(f"❌ FAILED: No match or all distances above threshold. Closest match was '{best_match_user}', Dist: {min_distance:.4f}, Threshold: {threshold:.3f}")
            reason = f"Speaker not recognized. New user? Signup" # Directing to sign up for new user if it coudnt recognise from any of existing users
            return jsonify({"status": "error", "message": reason}), 401
    except ValueError as ve:
        print(f"Login ValueError: {ve}"); return jsonify({"status": "error", "message": f"Login failed: {ve}"}), 500
    except Exception as e:
        print(f"Login Exception: {e}"); traceback.print_exc(); return jsonify({"status": "error", "message": "Internal server error."}), 500

# --- Run Application ---
if __name__ == '__main__':
    if not os.path.exists(app.config['USER_DATA_FILE']): save_user_data({})
    app.run(debug=True, host='0.0.0.0', port=5001)