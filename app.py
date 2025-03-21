import google.generativeai as genai
import re
import os
import json
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from flask_cors import CORS

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
                {"english": "are you", "asl": "YOU"}
            ],
            "time_patterns": [
                {"english": "tomorrow", "asl": "TOMORROW"},
                {"english": "yesterday", "asl": "YESTERDAY"},
                {"english": "today", "asl": "TODAY"},
                {"english": "next week", "asl": "NEXT WEEK"},
                {"english": "last week", "asl": "LAST WEEK"}
            ],
            "common_phrases": [
                {"english": "want to go", "asl": "WANT GO"},
                {"english": "go with me", "asl": "GO-WITH ME"},
                {"english": "for a date", "asl": "DATE"}
            ]
        }


def extract_proper_nouns(text):
    """Extract proper nouns from text more reliably."""
    words = text.split()
    proper_nouns = []

    for i, word in enumerate(words):
        # Strip punctuation for checking
        clean_word = re.sub(r'[^\w\s]', '', word)
        if not clean_word:
            continue

        # Check if word starts with uppercase (not at sentence beginning) or has internal caps
        if (clean_word[0].isupper() and i > 0) or \
           (i == 0 and len(clean_word) > 1 and clean_word[1].isupper()) or \
           any(c.isupper() for c in clean_word[1:]):
            proper_nouns.append(clean_word)

    return proper_nouns


def preprocess_text(text):
    """Preprocess the input text for better translation."""
    # Get proper nouns first
    proper_nouns = extract_proper_nouns(text)

    # Convert to lowercase for processing
    text_lower = text.lower().strip()

    # Don't remove proper nouns during preprocessing
    filler_words = ["a", "an", "the", "is",
                    "are", "am", "was", "were", "be", "been"]
    for word in filler_words:
        text_lower = re.sub(r'\b' + word + r'\b', '', text_lower)

    # Normalize spaces
    text_lower = re.sub(r'\s+', ' ', text_lower).strip()
    return text_lower, proper_nouns


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


def generate_sign_grammar(text, api_key, proper_nouns, vocabulary, model_name="gemini-2.0-flash"):
    """Generate ASL grammar from English text using Gemini API."""
    try:
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(model_name)

        # Preprocess but preserve proper nouns
        preprocessed_text, _ = preprocess_text(
            text)  # We already have proper_nouns

        # Get ASL patterns
        asl_patterns = load_asl_patterns()
        matched_patterns = pattern_match(preprocessed_text, asl_patterns)
        matched_pattern_info = ""
        if matched_patterns:
            matched_pattern_info = "\n\nDetected patterns in the input:\n" + "\n".join([
                f"- '{p['english']}' typically translates to '{p['asl']}' in ASL"
                for p in matched_patterns
            ])

        # Add proper nouns to prompt context
        proper_nouns_info = ""
        if proper_nouns:
            proper_nouns_info = "\n\nPreserve these proper nouns EXACTLY AS IS: " + \
                ", ".join(proper_nouns)

        prompt = f"""
        Translate the following English text into accurate American Sign Language (ASL) grammar.
        Keep all proper nouns and names exactly as they appear in the input.
        DO NOT modify, remove, or transform proper nouns in any way.
        
        # Input Analysis
        Original text: "{text}"
        Preprocessed: "{preprocessed_text}"{proper_nouns_info}
        Type: {("Question" if "?" in text else "Statement")}
        {matched_pattern_info}
        
        # Translation Rules
        1. Keep ALL proper nouns unchanged (e.g., names like {', '.join(proper_nouns) if proper_nouns else 'John, Mary'})
        2. Place proper nouns in the correct ASL grammar position
        3. Use TIME + TOPIC + COMMENT structure for longer sentences
        4. DO NOT abbreviate or shorten any proper noun
        
        # Output Requirements
        1. Output MUST include all proper nouns from input EXACTLY as they appear
        2. Use correct ASL word order
        3. Output in ALL CAPS with proper spacing
        4. Do not remove, shorten or modify names like SURESH, MICHAEL, NEWYORK, etc.
        
        # Example Translations
        - "hello John" → "HELLO JOHN" (not "HELLO J")
        - "mary is happy" → "MARY HAPPY" (not "M HAPPY")
        - "see you tomorrow Bob" → "TOMORROW BOB SEE" (not "TOMORROW B SEE")
        
        # Output Format
        Return ONLY the final ASL translation in CAPS, preserving all proper nouns intact.
        """

        is_question = "?" in text or text.lower().startswith(
            ("do ", "can ", "will ", "what ", "when ", "where ", "why ", "how "))

        response = model.generate_content(prompt)
        if not response or not response.text:
            print("Empty response from Gemini API")
            return None

        sign_grammar = response.text.strip().upper()

        # Clean up output
        sign_grammar = re.sub(r'\[.*?\]', '', sign_grammar)
        sign_grammar = re.sub(
            r'ASL( GRAMMAR| TRANSLATION)?:', '', sign_grammar).strip()
        sign_grammar = re.sub(r'#.*?$', '', sign_grammar,
                              flags=re.MULTILINE).strip()

        if is_question and "?" not in sign_grammar:
            sign_grammar = sign_grammar + " ?"

        # Add question mark if needed and return
        # No vocabulary filtering or "closest match" replacements
        return sign_grammar

    except Exception as e:
        print(f"Error in generate_sign_grammar: {str(e)}")
        return None


def process_long_text(long_text, api_key, vocabulary):
    """Process longer text by breaking it into sentences and translating each."""
    # Extract proper nouns from the entire text
    proper_nouns = extract_proper_nouns(long_text)

    # For short texts, process the whole thing at once
    if long_text.count('.') <= 1 and long_text.count('?') <= 1 and len(long_text.split()) <= 15:
        return generate_sign_grammar(long_text, api_key, proper_nouns, vocabulary)

    # For longer texts, split into sentences
    sentences = re.split(
        r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', long_text)
    sign_grammars = []

    for sentence in sentences:
        if sentence.strip():
            # Get proper nouns for this sentence
            sentence_proper_nouns = extract_proper_nouns(sentence)
            sign_grammar = generate_sign_grammar(sentence.strip(), api_key,
                                                 sentence_proper_nouns, vocabulary)
            if sign_grammar:
                sign_grammars.append(sign_grammar)

    return " ".join(sign_grammars)


@app.route('/translate', methods=['POST'])
def translate_text():
    try:
        data = request.get_json()
        if not data or 'text' not in data:
            return jsonify({'error': 'Missing text in request'}), 400
        english_text = data['text']
        if not english_text.strip():
            return jsonify({'error': 'Empty text provided'}), 400

        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            return jsonify({'error': 'GOOGLE_API_KEY not set in environment variables'}), 500


        try:
            with open("words_list.txt", "r", encoding='utf-8') as f:
                vocabulary = set(line.strip().upper() for line in f)
        except FileNotFoundError:
            print("words_list.txt not found")
            return jsonify({'error': 'words_list.txt not found'}), 500
        except Exception as e:
            print(f"Error loading vocabulary: {str(e)}")
            return jsonify({'error': f'Error loading vocabulary: {str(e)}'}), 500

        # Extract proper nouns from text
        proper_nouns = extract_proper_nouns(english_text)
        proper_nouns_upper = [noun.upper() for noun in proper_nouns]

        # Add proper nouns to vocabulary
        extended_vocabulary = vocabulary.copy()
        for noun in proper_nouns_upper:
            extended_vocabulary.add(noun)

        # Process text with extended vocabulary
        sign_grammar = process_long_text(
            english_text, api_key, extended_vocabulary)
        if not sign_grammar:
            return jsonify({'error': 'Translation failed - empty response'}), 500


        # No additional processing or filtering - use the direct output
        final_sign_grammar = sign_grammar

        # Check for any out of vocab words excluding proper nouns
        words = sign_grammar.split()
        out_of_vocab = [
            w for w in words if w not in extended_vocabulary and w != "?"]
        status = "corrected" if out_of_vocab else "success"

        return jsonify({
            'original_text': english_text,
            'sign_grammar': final_sign_grammar,
            'status': status,
            'out_of_vocab': out_of_vocab if out_of_vocab else [],
            'proper_nouns_detected': proper_nouns
        })

    except Exception as e:
        print(f"Error in translate_text: {str(e)}")
        return jsonify({'error': f'Translation failed: {str(e)}'}), 500


@app.route('/health', methods=['GET'])
def health():
    return jsonify({'status': 'ok'}), 200


@app.route('/favicon.ico')
def favicon():
    return '', 204  # No Content


if __name__ == '__main__':
    port = int(os.getenv("PORT", 8080))
    app.run(debug=False, host='0.0.0.0', port=port)
