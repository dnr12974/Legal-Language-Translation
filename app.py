from flask import Flask, request, jsonify, render_template
from transformers import MarianMTModel, MarianTokenizer
from evaluate import load
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import torch

app = Flask(__name__)

# Declare global variables for feedback data and BLEU scoring
feedback_data = []
bleu = load("bleu")
similarity_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

# Load the models only when required (on-demand)
tokenizer_en_fr = None
model_en_fr = None
tokenizer_fr_en = None
model_fr_en = None

# Load models when the translation endpoint is called
def load_translation_models(source, target):
    global tokenizer_en_fr, model_en_fr, tokenizer_fr_en, model_fr_en
    if source == "en" and target == "fr":
        if tokenizer_en_fr is None or model_en_fr is None:
            tokenizer_en_fr = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
            model_en_fr = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-en-fr")
    elif source == "fr" and target == "en":
        if tokenizer_fr_en is None or model_fr_en is None:
            tokenizer_fr_en = MarianTokenizer.from_pretrained("Helsinki-NLP/opus-mt-fr-en")
            model_fr_en = MarianMTModel.from_pretrained("Helsinki-NLP/opus-mt-fr-en")

# Route for the home page
@app.route('/')
def home():
    return render_template('index.html')

# Route for translation
@app.route('/translate', methods=['POST'])
def translate():
    data = request.json
    text, source, target = data['text'], data['source'], data['target']
    
    # Load models dynamically based on source and target languages
    load_translation_models(source, target)

    # Select tokenizer and model based on the language pair
    tokenizer = tokenizer_en_fr if source == "en" and target == "fr" else tokenizer_fr_en
    model = model_en_fr if source == "en" and target == "fr" else model_fr_en

    # Tokenize input text and generate translation
    input_ids = tokenizer(text, return_tensors="pt", truncation=True, padding=True).input_ids

    # Use no_grad() to reduce memory consumption during inference
    with torch.no_grad():
        output_ids = model.generate(input_ids)
    
    translated_text = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    return jsonify({"translatedText": translated_text})

# Route for receiving feedback
@app.route('/feedback', methods=['POST'])
def feedback():
    try:
        data = request.json
        feedback_data.append(data)
        
        # Log feedback data for debugging purposes
        print("Received Feedback:", data)
        
        # If there are 3 feedback entries, fine-tune the model
        if len(feedback_data) >= 3:
            fine_tune_model()

        return jsonify({"message": "Feedback received"})
    except Exception as e:
        print("Error handling feedback:", e)
        return jsonify({"error": "An error occurred while processing feedback."}), 500

# Function to fine-tune the translation model based on feedback
def fine_tune_model():
    if not model_en_fr or not tokenizer_en_fr:
        print("Model not loaded properly, cannot fine-tune.")
        return
    
    optimizer = torch.optim.AdamW(model_en_fr.parameters(), lr=5e-5)
    model_en_fr.train()

    for feedback in feedback_data:
        input_ids = tokenizer_en_fr(feedback["source"], return_tensors="pt", truncation=True, padding=True).input_ids
        reference = feedback["target"]

        outputs = model_en_fr.generate(input_ids)
        hypothesis = tokenizer_en_fr.decode(outputs[0], skip_special_tokens=True)
        reward = calculate_reward(reference, hypothesis)

        logits = model_en_fr(input_ids=input_ids, decoder_input_ids=outputs[:, :-1]).logits
        log_probs = torch.log_softmax(logits, dim=-1)
        loss = -torch.mean(log_probs) * reward

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Function to calculate reward based on BLEU score and similarity
def calculate_reward(reference, hypothesis):
    bleu_score = bleu.compute(predictions=[hypothesis], references=[[reference]])["bleu"]
    bleu_score = max(0.5, min(1.0, bleu_score))

    ref_embedding = similarity_model.encode([reference])
    hyp_embedding = similarity_model.encode([hypothesis])
    cosine_sim = cosine_similarity(ref_embedding, hyp_embedding)[0][0]
    return 0.5 * bleu_score + 0.5 * ((cosine_sim + 1) / 2)

# Run the Flask app
if __name__ == '__main__':
    app.run(debug=False)
