import google.generativeai as genai
import re
import os
import json
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS
import requests
from flask import send_file
from io import BytesIO

load_dotenv()

app = Flask(__name__)
CORS(app)  # Enables CORS for all routes


def levenshtein_distance(s1, s2):
    """Calculate the Levenshtein distance between two strings."""
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
    """Load predefined ASL grammar patterns from a JSON file."""
    try:
        with open("asl_patterns.json", "r") as f:
            return json.load(f)
    except FileNotFoundError:
        # Default patterns if file not found
        return {
            "question_patterns": [
                {"english": "do you want", "asl": "YOU WANT"},
                {"english": "can you", "asl": "YOU CAN"},
                {"english": "will you", "asl": "YOU WILL"},
                {"english": "have you", "asl": "YOU HAVE"},
                {"english": "are you", "asl": "YOU"},
            ],
            "time_patterns": [
                {"english": "tomorrow", "asl": "TOMORROW"},
                {"english": "yesterday", "asl": "YESTERDAY"},
                {"english": "today", "asl": "TODAY"},
                {"english": "next week", "asl": "NEXT WEEK"},
                {"english": "last week", "asl": "LAST WEEK"},
            ],
            "common_phrases": [
                {"english": "want to go", "asl": "WANT GO"},
                {"english": "go with me", "asl": "GO-WITH ME"},
                {"english": "for a date", "asl": "DATE"},
            ]
        }


def preprocess_text(text):
    """Preprocess the input text for better translation."""
    # Convert to lowercase for processing
    text = text.lower().strip()

    # Remove filler words
    filler_words = ["a", "an", "the", "is",
                    "are", "am", "was", "were", "be", "been"]
    for word in filler_words:
        text = re.sub(r'\b' + word + r'\b', '', text)

    # Normalize spaces
    text = re.sub(r'\s+', ' ', text).strip()

    return text


def pattern_match(text, patterns):
    """Match input text against predefined ASL patterns."""
    matched_patterns = []
    for pattern_type, pattern_list in patterns.items():
        for pattern in pattern_list:
            if pattern["english"].lower() in text.lower():
                matched_patterns.append({
                    "type": pattern_type,
                    "english": pattern["english"],
                    "asl": pattern["asl"]
                })
    return matched_patterns


def generate_sign_grammar(text, api_key, word_list_file="words_list.txt", model_name="gemini-2.0-flash"):
    """Generate ASL grammar from English text using Gemini API with enhanced prompting."""
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(model_name)

    try:
        with open(word_list_file, "r") as f:
            vocabulary = set(line.strip().upper() for line in f)
    except FileNotFoundError:
        return None, None

    # Preprocess the input text
    preprocessed_text = preprocess_text(text)

    # Load ASL patterns
    asl_patterns = load_asl_patterns()

    # Match patterns in the text
    matched_patterns = pattern_match(preprocessed_text, asl_patterns)
    matched_pattern_info = ""
    if matched_patterns:
        matched_pattern_info = "\n\nDetected patterns in the input:\n" + "\n".join([
            f"- '{p['english']}' typically translates to '{p['asl']}' in ASL"
            for p in matched_patterns
        ])

    # Add examples for common question structures
    examples = """
    # Examples of ASL Grammar Translation:
    1. English: "Do you want to go to the store?"
       ASL: "STORE YOU WANT GO"
       
    2. English: "Can you help me with my homework?"
       ASL: "HOMEWORK YOU HELP ME"
       
    3. English: "What time is the meeting tomorrow?"
       ASL: "TOMORROW MEETING TIME WHAT"
       
    4. English: "I want to go on a date with you."
       ASL: "ME WANT DATE YOU"
    """

    # Determine if the text is a question
    is_question = "?" in text or text.lower().startswith(
        ("do ", "can ", "will ", "what ", "when ", "where ", "why ", "how "))
    question_guidance = """
    # Question Structure in ASL:
    - Yes/No questions: Non-manual markers (raised eyebrows) with question at the end
    - WH-questions: Question word typically at the end (WHAT, WHO, WHERE, WHEN, WHY, HOW)
    - Question format: [TOPIC] [COMMENT] [QUESTION-WORD]
    """

    prompt = f"""
    Translate the following English text into accurate American Sign Language (ASL) grammar.
    
    # Input Analysis
    Original text: "{text}"
    Preprocessed: "{preprocessed_text}"
    Type: {("Question" if is_question else "Statement")}
    {matched_pattern_info}
    
    # ASL Grammar Rules
    1. TOPIC-COMMENT structure (not Subject-Verb-Object)
    2. Time expressions come at the beginning
    3. Questions have question words at the END
    4. Establish subjects in space
    5. No articles (a, an, the)
    6. Limited use of prepositions
    7. No verb conjugations (use base form)
    8. Use repetition for plurality
    
    {examples}
    
    {question_guidance if is_question else ""}
    
    # Vocabulary Constraints
    Only use words from this vocabulary: {", ".join(sorted(list(vocabulary)[:50]))}... (and others from the full vocabulary list)
    
    # Translation Process
    1. Identify time references (if any) → place at beginning
    2. Identify main topic → place next
    3. Identify main action/verb → use base form
    4. For questions, move question word to the end
    5. Remove unnecessary words
    6. Ensure proper ASL word order
    7. Verify all words are in vocabulary
    
    # Output Format
    Provide ONLY the final ASL grammar translation in ALL CAPS with proper spacing.
    """

    try:
        response = model.generate_content(prompt)
        sign_grammar = response.text.strip().upper()

        # Clean up the output
        # Remove [non-manual markers]
        sign_grammar = re.sub(r'\[.*?\]', '', sign_grammar)
        sign_grammar = re.sub(r'ASL( GRAMMAR| TRANSLATION)?:',
                              '', sign_grammar).strip()  # Remove headers
        sign_grammar = re.sub(r'#.*?$', '', sign_grammar,
                              flags=re.MULTILINE).strip()  # Remove comments

        # Apply post-processing rules for common ASL structures
        if is_question and "?" not in sign_grammar:
            # For questions, ensure question mark is included
            sign_grammar = sign_grammar + " ?"

        # Remove any words not in vocabulary
        words = sign_grammar.split()
        sign_grammar = " ".join(
            [word for word in words if word in vocabulary or word == "?"])

        return sign_grammar, vocabulary
    except Exception as e:
        return None, None


def get_closest_vocabulary_word(word, vocabulary, max_distance=3):
    """Find the closest word in vocabulary using semantic and phonetic similarity."""
    if word in vocabulary:
        return word

    closest_match = None
    min_distance = float('inf')

    # Check for prefix matches first (more semantically relevant)
    prefix_matches = [v for v in vocabulary if v.startswith(
        word[:3]) and len(v) >= len(word) - 2]

    if prefix_matches:
        for vocab_word in prefix_matches:
            distance = levenshtein_distance(word, vocab_word)
            if distance < min_distance and distance <= max_distance:
                min_distance = distance
                closest_match = vocab_word
    else:
        # Fall back to full vocabulary search
        for vocab_word in vocabulary:
            distance = levenshtein_distance(word, vocab_word)
            if distance < min_distance and distance <= max_distance:
                min_distance = distance
                closest_match = vocab_word

    return closest_match


def correct_vocabulary(sign_grammar, vocabulary):
    """Ensure all words in the sign grammar are in the vocabulary."""
    if sign_grammar is None or vocabulary is None:
        return None

    words = sign_grammar.split()
    corrected_words = []

    for word in words:
        if word in vocabulary or word == "?":
            corrected_words.append(word)
        else:
            closest_match = get_closest_vocabulary_word(word, vocabulary)
            if closest_match:
                corrected_words.append(closest_match)

    return " ".join(corrected_words)


def process_long_text(long_text, api_key, vocabulary):
    """Process longer text by breaking it into sentences and translating each."""
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


def post_process_asl_grammar(asl_grammar, original_text, vocabulary):
    """Apply ASL-specific post-processing rules."""
    # Special handling for questions
    if "?" in original_text and "?" not in asl_grammar:
        asl_grammar += " ?"

    # Ensure proper ASL word order for questions
    if "?" in original_text or original_text.lower().startswith(("do ", "can ", "will ", "what ", "when ", "where ", "why ", "how ")):
        words = asl_grammar.split()

        # Move question words to the end if they're at the beginning
        question_words = ["WHAT", "WHO", "WHERE", "WHEN", "WHY", "HOW"]
        for q_word in question_words:
            if words and words[0] == q_word:
                words.append(words.pop(0))

        asl_grammar = " ".join(words)

    # Handle time expressions
    time_words = ["TODAY", "TOMORROW", "YESTERDAY", "MORNING",
                  "AFTERNOON", "NIGHT", "WEEK", "MONTH", "YEAR"]
    words = asl_grammar.split()
    for time_word in time_words:
        if time_word in words and words.index(time_word) > 0:
            words.remove(time_word)
            words.insert(0, time_word)

    return " ".join(words)


@app.route('/translate', methods=['POST'])
def translate_text():
    data = request.get_json()
    if not data or 'text' not in data:
        return jsonify({'error': 'Missing text in request'}), 400

    api_key = os.getenv('GOOGLE_API_KEY')
    if not api_key:
        return jsonify({'error': 'GOOGLE_API_KEY not set in environment variables'}), 500

    english_text = data['text']

    try:
        with open("words_list.txt", "r") as f:
            vocabulary = set(line.strip().upper() for line in f)
    except FileNotFoundError:
        return jsonify({'error': 'words_list.txt not found'}), 500

    # For short texts, process directly
    if len(english_text.split()) <= 15:
        sign_grammar, _ = generate_sign_grammar(english_text, api_key)
        if sign_grammar:
            sign_grammar = correct_vocabulary(sign_grammar, vocabulary)
            # Apply post-processing
            sign_grammar = post_process_asl_grammar(
                sign_grammar, english_text, vocabulary)
    else:
        # For longer texts, process sentence by sentence
        sign_grammar = process_long_text(english_text, api_key, vocabulary)

    if sign_grammar:
        # Final validation check
        words = sign_grammar.split()
        valid_words = [w for w in words if w in vocabulary or w == "?"]
        out_of_vocab = [w for w in words if w not in vocabulary and w != "?"]

        if out_of_vocab:
            status = "corrected"
        else:
            status = "success"

        return jsonify({
            'original_text': english_text,
            'sign_grammar': " ".join(valid_words),
            'status': status,
            'out_of_vocab': out_of_vocab if out_of_vocab else []
        })
    else:
        return jsonify({'error': 'Translation failed'}), 500


@app.route('/proxy/gif/<word>', methods=['GET'])
def proxy_gif(word):
    """Proxy endpoint to fetch GIFs and handle CORS."""
    try:
        gif_url = f"{os.getenv('GIF_BASE_URL')}/{word}.gif"
        response = requests.get(gif_url)
        if response.status_code == 200:
            return send_file(
                BytesIO(response.content),
                mimetype='image/gif',
                as_attachment=False
            )
        return jsonify({'error': 'GIF not found'}), 404
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200


@app.route('/favicon.ico')
def favicon():
    return '', 204  # No Content


if __name__ == '__main__':
    port = int(os.getenv("PORT", 8080))
    app.run(debug=False, host='0.0.0.0', port=port)
