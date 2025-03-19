import google.generativeai as genai
import re
import os
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enables CORS for all routes


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


def generate_sign_grammar(text, api_key, word_list_file="words_list.txt", model_name="gemini-2.0-flash"):
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)
    try:
        with open(word_list_file, "r") as f:
            vocabulary = set(line.strip().upper() for line in f)
    except FileNotFoundError:
        return None, None

    prompt = f"""
    Translate the following English sentence into an ASL-compatible grammar. 
    - Use a **subject-verb-object** structure where applicable.
    - Use words from the provided vocabulary: {", ".join(vocabulary)}.
    - If a word is missing from the vocabulary, replace it with a **similar meaning word**.
    - Preserve **essential context** while simplifying the sentence.

    **English:** {text}  
    **ASL Grammar:**
    """

    try:
        response = model.generate_content(prompt)
        sign_grammar = response.text.strip().upper()
        return sign_grammar, vocabulary
    except Exception as e:
        return None, None


def check_vocabulary(sign_grammar, vocabulary):
    if sign_grammar is None or vocabulary is None:
        return False, []
    words = re.findall(r'\b\w+\b', sign_grammar)
    out_of_vocab = [word for word in words if word not in vocabulary]
    return not out_of_vocab, out_of_vocab


def correct_vocabulary(sign_grammar, vocabulary):
    if sign_grammar is None or vocabulary is None:
        return None
    words = re.findall(r'\b\w+\b', sign_grammar)
    corrected_words = []
    for word in words:
        if word in vocabulary:
            corrected_words.append(word)
        else:
            closest_match = None
            min_distance = float('inf')
            for vocab_word in vocabulary:
                distance = levenshtein_distance(word, vocab_word)
                if distance < min_distance and distance <= 4:
                    min_distance = distance
                    closest_match = vocab_word
            if closest_match:
                corrected_words.append(closest_match)
            else:
                corrected_words.append(word)
    return " ".join(corrected_words)


def process_long_text(long_text, api_key, vocabulary):
    """
    Splits long text while maintaining sentence structure and converts each part separately.
    """
    sentences = re.split(
        r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', long_text)
    sign_grammars = []

    for sentence in sentences:
        if sentence.strip():
            sign_grammar, _ = generate_sign_grammar(sentence.strip(), api_key)
            if sign_grammar:
                sign_grammar = correct_vocabulary(sign_grammar, vocabulary)
                sign_grammars.append(sign_grammar)

    return " ".join(sign_grammars)


@app.route('/translate', methods=['POST'])
def translate_text():
    print("Received request method:", request.method)
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing text in request'}), 400

    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        return jsonify({'error': 'GOOGLE_API_KEY not set in environment variables'}), 500

    long_english_text = data['text']

    try:
        with open("words_list.txt", "r") as f:
            vocabulary = set(line.strip().upper() for line in f)
    except FileNotFoundError:
        return jsonify({'error': 'words_list.txt not found'}), 500

    final_sign_grammar = process_long_text(
        long_english_text, api_key, vocabulary)

    if final_sign_grammar:
        is_valid, out_of_vocab_words = check_vocabulary(
            final_sign_grammar, vocabulary)
        if is_valid:
            return jsonify({'original_text': long_english_text, 'sign_grammar': final_sign_grammar, 'status': 'success'})
        else:
            return jsonify({'original_text': long_english_text, 'sign_grammar': final_sign_grammar, 'status': 'corrected', 'out_of_vocab': out_of_vocab_words})
    else:
        return jsonify({'error': 'Translation failed'}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200


@app.route('/favicon.ico')
def favicon():
    return '', 204  # No Content


if __name__ == '__main__':
    port = int(os.getenv("PORT", 8080))  # Use Railway’s assigned port
    app.run(debug=False, host='0.0.0.0', port=port)
