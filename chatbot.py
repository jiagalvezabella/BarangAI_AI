from sentence_transformers import SentenceTransformer, util

# Load the multilingual model
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Sample intents in English and Cebuano
intents = {
    "greet": ["hello", "hi", "good morning", "kamusta", "maayong buntag"],
    "ms_word": ["how to use word", "paano gamitin ang word", "unsaon paggamit sa word"],
    "ms_excel": ["how to use excel", "paano gamitin ang excel", "unsaon paggamit sa excel"],
    "email": ["how to send email", "paano mag padala ng email", "unsaon pag padala ug email"],
}

def get_intent(user_input):
    best_match = None
    best_score = 0
    
    user_embedding = model.encode(user_input)
    
    for intent, examples in intents.items():
        for example in examples:
            example_embedding = model.encode(example)
            score = util.cos_sim(user_embedding, example_embedding).item()
            if score > best_score:
                best_score = score
                best_match = intent
    
    return best_match, best_score

# Test it
user_input = "unsaon paggamit sa microsoft word"
intent, score = get_intent(user_input)
print(f"Intent: {intent}, Score: {score:.2f}")