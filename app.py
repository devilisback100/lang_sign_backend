import google.generativeai as genai
import re
import os
import json
import imageio
import numpy as np
from PIL import Image
import requests
import time
from flask import Flask, request, jsonify, send_file
from dotenv import load_dotenv
from flask_cors import CORS
import io

load_dotenv()

app = Flask(__name__)
CORS(app, expose_headers=['X-Original-Text', 'X-ASL-Translation', 'X-Fingerspell-Words'])


TARGET_HEIGHT = 480
GIF_BASE_URL = os.getenv('GIF_BASE_URL')  # Get from .env


def resize_frame(frame):
    img = Image.fromarray(frame)
    aspect_ratio = img.width / img.height
    new_width = int(TARGET_HEIGHT * aspect_ratio)
    resized = img.resize((new_width, TARGET_HEIGHT), Image.Resampling.LANCZOS)
    return np.array(resized)


def create_video_from_asl(text, fingerspell_words):
    frames = []
    words = text.split()

    for word in words:
        try:
            if word in fingerspell_words:
                # Fingerspell the word
                for letter in word.lower():
                    response = requests.get(
                        f"{GIF_BASE_URL}/{letter}.gif", stream=True)
                    if response.status_code == 200:
                        gif = imageio.get_reader(response.content, '.gif')
                        letter_frames = [resize_frame(frame) for frame in gif]
                        for frame in letter_frames:
                            frames.extend([frame] * 4)
                        # Pause between letters
                        frames.extend([frames[-1]] * 6)
            else:
                # Regular word sign
                response = requests.get(
                    f"{GIF_BASE_URL}/{word.lower()}.gif", stream=True)
                if response.status_code == 200:
                    gif = imageio.get_reader(response.content, '.gif')
                    word_frames = [resize_frame(frame) for frame in gif]
                    for frame in word_frames:
                        frames.extend([frame] * 3)
                    frames.extend([frames[-1]] * 10)  # Pause between words
        except Exception as e:
            print(f"Error processing {word}: {e}")
            continue

    if not frames:
        return None

    try:
        # Create bytes buffer instead of saving to file
        buffer = io.BytesIO()
        writer = imageio.get_writer(
            buffer, format='mp4', fps=10, codec='libx264',
            pixelformat='yuv420p', bitrate='8M'
        )

        for frame in frames:
            writer.append_data(frame)

        writer.close()
        buffer.seek(0)
        return buffer
    except Exception as e:
        print(f"Error creating video: {e}")
        if 'writer' in locals():
            writer.close()
        return None


def levenshtein_distance(s1, s2):
    if len(s1) < len(s2):
        return levenshtein_distance(s2, s1)
    if len(s2) == 0:
        return len(s1)
    previous_row = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        current_row = [i + 1]
        for j, c2 in enumerate(s2):
            insertions = previous_row[j + 1] + 1
            deletions = current_row[j] + 1
            substitutions = previous_row[j] + (c1 != c2)
            current_row.append(min(insertions, deletions, substitutions))
        previous_row = current_row
    return previous_row[-1]


def load_asl_patterns():
    try:
        with open("asl_patterns.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        return {
            "question_patterns": [
                {"english": "do you want", "asl": "YOU WANT"},
                {"english": "can you", "asl": "YOU CAN"},
                {"english": "will you", "asl": "YOU WILL"},
                {"english": "have you", "asl": "YOU HAVE"},
                {"english": "are you", "asl": "YOU"},
            ],
            "time_phrases": [
                {"english": "tomorrow", "asl": "TOMORROW"},
                {"english": "yesterday", "asl": "YESTERDAY"},
                {"english": "today", "asl": "TODAY"},
                {"english": "next week", "asl": "NEXT-WEEK"},
                {"english": "last week", "asl": "LAST-WEEK"},
            ],
            "common_phrases": [
                {"english": "want to go", "asl": "WANT GO"},
                {"english": "go with me", "asl": "GO-WITH ME"},
                {"english": "for a date", "asl": "DATE"},
            ]
        }


def preprocess_text(text):
    text = re.sub(r'\s+', ' ', text).strip()
    filler_words = ["a", "an", "the", "is",
                    "are", "am", "was", "were", "be", "been"]
    for word in filler_words:
        text = re.sub(rf'\b{word}\b', '', text, flags=re.IGNORECASE)
    return text.strip()


def pattern_match(text, patterns):
    matched = []
    text_lower = text.lower()
    for ptype, plist in patterns.items():
        for pattern in plist:
            if re.search(rf'\b{re.escape(pattern["english"].lower())}\b', text_lower):
                matched.append({
                    "type": ptype,
                    "english": pattern["english"],
                    "asl": pattern["asl"]
                })
    return matched


def generate_sign_grammar(text, api_key, word_list="words_list.txt", model_name="gemini-2.0-flash"):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    try:
        with open(word_list, "r") as f:
            vocabulary = set(line.strip().upper() for line in f)
    except FileNotFoundError:
        return None, None

    preprocessed = preprocess_text(text)
    patterns = load_asl_patterns()
    matched = pattern_match(preprocessed, patterns)

    pattern_info = "\n".join(
        [f"- {p['english']} → {p['asl']}" for p in matched])
    is_question = any(
        [text.endswith("?"), text.lower().startswith(('do ', 'can ', 'will ', 'wh'))])

    prompt = f"""Translate this English text to ASL grammar following these rules:

# ASL Grammar Rules
1. TOPIC-COMMENT structure (Subject/Object first)
2. Time expressions first
3. Questions: Put question words (WHAT/WHERE/etc) LAST
4. Omit articles (a/an/the)
5. Use base verbs
6. Use ONLY these signs: {', '.join(sorted(vocabulary))}

# Examples
English: "Do you want coffee?"
ASL: "COFFEE YOU WANT?"

English: "Tomorrow I go store"
ASL: "TOMORROW STORE I GO"

English: "What is your name?"
ASL: "YOUR NAME WHAT"

# Patterns Detected
{pattern_info if pattern_info else "No common patterns detected"}

# Translate this:
English: "{text}"
ASL:"""

    try:
        response = model.generate_content(
            prompt, generation_config={"temperature": 0.1})
        raw_asl = response.text.strip().upper()

        # Clean response
        clean_asl = re.sub(r'(ASL|ENGLISH|TRANSLATION)[:\s]*', '', raw_asl)
        clean_asl = re.sub(r'["“”]', '', clean_asl)
        return clean_asl, vocabulary
    except Exception as e:
        print(f"Generation error: {e}")
        return None, None


def get_closest_word(word, vocabulary, max_dist=2):
    if word in vocabulary:
        return word
    if word.isupper() and any(c.isalpha() for c in word):
        return word  # Proper noun

    closest = None
    min_dist = float('inf')
    for v in vocabulary:
        dist = levenshtein_distance(word, v)
        if dist < min_dist and dist <= max_dist:
            min_dist = dist
            closest = v
    return closest or word


def validate_translation(asl_text, vocabulary):
    if not asl_text:
        return "", []

    fingerspell = []
    validated = []

    for token in asl_text.split():
        if '-' in token:
            parts = token.split('-')
            valid_parts = []
            for p in parts:
                closest = get_closest_word(p, vocabulary)
                if closest != p:
                    fingerspell.append(p)
                    valid_parts.append(closest)
                else:
                    valid_parts.append(p)
            validated.append('-'.join(valid_parts))
        else:
            closest = get_closest_word(token, vocabulary)
            if closest != token:
                fingerspell.append(token)
                validated.append(closest)
            else:
                validated.append(token)

    return ' '.join(validated), list(set(fingerspell))


@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing text'}), 400

    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        return jsonify({'error': 'API key missing'}), 500

    text = data['text'].strip()
    if not text:
        return jsonify({'error': 'Empty text'}), 400

    asl_grammar, vocab = generate_sign_grammar(text, api_key)
    if not asl_grammar:
        return jsonify({'error': 'Translation failed'}), 500

    validated, fingerspell = validate_translation(asl_grammar, vocab)

    # Generate video and return as stream
    video_buffer = create_video_from_asl(validated, fingerspell)

    response = {
        'original': text,
        'translation': validated,
        'fingerspell': fingerspell,
        'status': 'success'
    }

    if video_buffer:
        response = send_file(
            video_buffer,
            mimetype='video/mp4',
            as_attachment=True,
            download_name=f"sign_{int(time.time())}.mp4"
        )
        # Add custom headers and ensure they are exposed
        response.headers['Access-Control-Expose-Headers'] = 'X-Original-Text, X-ASL-Translation, X-Fingerspell-Words'
        response.headers['X-Original-Text'] = text
        response.headers['X-ASL-Translation'] = validated
        response.headers['X-Fingerspell-Words'] = ','.join(fingerspell)
        return response

    return jsonify(response)


@app.route('/health')
def health_check():
    return jsonify({'status': 'ok'})





if __name__ == '__main__':
    port = int(os.getenv("PORT", 8080))
    app.run(host='0.0.0.0', port=port)
