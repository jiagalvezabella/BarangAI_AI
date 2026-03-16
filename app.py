from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from typing import List, Dict, Optional
import numpy as np
import uuid

app = FastAPI(title="BarangAI Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the multilingual model
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Conversation sessions storage (Dialogue State Tracker)
sessions: Dict[str, List[Dict]] = {}

# User skill profiles
user_profiles: Dict[str, Dict] = {}

# Intents in English and Cebuano/Bisaya
intents = {
    "greet": {
        "examples": ["hello", "hi", "good morning", "kamusta", "maayong buntag", "maayong hapon", "maayong gabii", "hey"],
        "response": "Hello! How can I help you today? / Kamusta! Unsaon nako pagtabang kanimo?",
        "difficulty": "basic"
    },
    "basic_computer": {
        "examples": ["how to turn on computer", "how to restart", "unsaon pag-on sa computer", "unsaon pag-restart", "how to shut down", "unsaon pag-off sa computer", "computer not working"],
        "response": "To turn on your computer, press the power button. To restart, click Start > Restart. To shut down, click Start > Shut Down. / Aron i-on ang computer, pindota ang power button. Aron i-restart, i-click ang Start > Restart. Aron i-off, i-click ang Start > Shut Down.",
        "difficulty": "basic"
    },
    "save_file": {
        "examples": ["how to save file", "unsaon pagluwas sa file", "how to save document", "save file help", "unsaon pag save", "dili ma save ang file"],
        "response": "To save a file, press Ctrl+S or click File > Save. To save as a new file, press Ctrl+Shift+S or click File > Save As. / Aron magluwas sa file, pindota ang Ctrl+S o i-click ang File > Save.",
        "difficulty": "basic"
    },
    "ms_word": {
        "examples": ["how to use word", "paano gamitin ang word", "unsaon paggamit sa word", "microsoft word help", "word document", "how to format document", "word tutorial"],
        "response": "I can help you with Microsoft Word! You can create documents, format text, insert tables and images. What specific help do you need? / Makatabang ko sa Microsoft Word! Unsay imong gikinahanglan?",
        "difficulty": "basic"
    },
    "ms_word_table": {
        "examples": ["how to insert table in word", "unsaon pagsal-ot ug table sa word", "add table word", "table in document", "unsaon paghimo ug table"],
        "response": "To insert a table in Word: Click Insert > Table > select rows and columns. / Aron magsal-ot ug table sa Word: I-click ang Insert > Table > pilia ang gidaghanon sa rows ug columns.",
        "difficulty": "intermediate"
    },
    "ms_word_format": {
        "examples": ["how to change font", "unsaon pagbag-o sa font", "how to bold text", "unsaon pag bold", "how to change font size", "text formatting word"],
        "response": "To format text in Word: Select the text first, then use the toolbar to change font, size, bold (Ctrl+B), italic (Ctrl+I), or underline (Ctrl+U). / Aron mag-format ug teksto sa Word: Pilion una ang teksto, unya gamiton ang toolbar.",
        "difficulty": "intermediate"
    },
    "ms_excel": {
        "examples": ["how to use excel", "paano gamitin ang excel", "unsaon paggamit sa excel", "spreadsheet help", "excel help", "excel tutorial"],
        "response": "I can help you with Microsoft Excel! Excel is used for spreadsheets, calculations, and data management. / Makatabang ko sa Microsoft Excel! Ang Excel gamiton sa spreadsheets, kalkulasyon, ug pagdumala sa datos.",
        "difficulty": "basic"
    },
    "ms_excel_formula": {
        "examples": ["how to use formula in excel", "unsaon paggamit sa formula sa excel", "sum formula", "excel calculation", "unsaon pag calculate sa excel", "excel sum", "average formula"],
        "response": "To use formulas in Excel: Start with = sign. Common formulas: =SUM(A1:A5) to add, =AVERAGE(A1:A5) for average. / Aron mogamit ug formula sa Excel: Sugdi sa = sign. =SUM(A1:A5) aron magdugang, =AVERAGE(A1:A5) alang sa average.",
        "difficulty": "intermediate"
    },
    "ms_excel_chart": {
        "examples": ["how to create chart in excel", "unsaon paghimo ug chart sa excel", "graph excel", "chart excel", "data visualization excel"],
        "response": "To create a chart in Excel: Select your data, click Insert > Chart, choose chart type (bar, line, pie). / Aron maghimo ug chart sa Excel: Pilion ang imong datos, i-click ang Insert > Chart.",
        "difficulty": "advanced"
    },
    "email": {
        "examples": ["how to send email", "paano mag padala ng email", "unsaon pag padala ug email", "email help", "compose email", "how to attach file in email"],
        "response": "To send an email: Open your email app > Click Compose > Enter recipient email > Add subject > Write message > Click Send. / Aron magpadala ug email: Ablihi ang email app > I-click ang Compose > Isulod ang email sa tigdawat > I-click ang Send.",
        "difficulty": "basic"
    },
    "internet": {
        "examples": ["how to search online", "unsaon pagpangita online", "how to use google", "unsaon paggamit sa google", "internet help", "how to browse internet"],
        "response": "To search online: Open your browser > Go to google.com > Type your search query > Press Enter. / Aron mangita online: Ablihi ang imong browser > Adto sa google.com > I-type ang imong pangutana > Pindota ang Enter.",
        "difficulty": "basic"
    },
    "video_call": {
        "examples": ["how to use zoom", "unsaon paggamit sa zoom", "google meet help", "video call help", "online meeting", "unsaon pagsalmot sa zoom meeting"],
        "response": "To join a Zoom meeting: Click the meeting link or open Zoom > Enter Meeting ID > Click Join. For Google Meet: Click the meeting link > Click Join Now. / Aron mosalmot sa Zoom: I-click ang meeting link > Isulod ang Meeting ID > I-click ang Join.",
        "difficulty": "intermediate"
    },
    "password": {
        "examples": ["how to change password", "unsaon pagbag-o sa password", "forgot password", "nakalimtan ang password", "password help"],
        "response": "To change your password: Go to Settings > Account > Change Password. Make sure your password is strong with letters, numbers, and symbols. / Aron mabag-o ang password: Adto sa Settings > Account > Change Password.",
        "difficulty": "basic"
    },
    "print": {
        "examples": ["how to print document", "unsaon pag-print sa dokumento", "print help", "printer not working", "unsaon paggamit sa printer"],
        "response": "To print a document: Press Ctrl+P or click File > Print. Select your printer and click Print. / Aron mag-print: Pindota ang Ctrl+P o i-click ang File > Print. Pilia ang imong printer unya i-click ang Print.",
        "difficulty": "basic"
    },
    "goodbye": {
        "examples": ["bye", "goodbye", "thank you", "salamat", "ok thanks", "paalam", "daghang salamat"],
        "response": "Goodbye! Feel free to ask again anytime! / Paalam! Ayaw kahadlok pangutana bisan kanus-a!",
        "difficulty": "basic"
    },
}

# ============================================================
# SVM INTENT CLASSIFIER - Train on startup
# ============================================================
print("Training SVM Intent Classifier...")
X_train = []
y_train = []

for intent_name, data in intents.items():
    for example in data["examples"]:
        embedding = model.encode(example)
        X_train.append(embedding)
        y_train.append(intent_name)

X_train = np.array(X_train)
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_train)

svm_classifier = SVC(kernel='rbf', probability=True)
svm_classifier.fit(X_train, y_encoded)
print("SVM Classifier trained successfully!")

# ============================================================
# RANDOM FOREST SKILL LEVEL CLASSIFIER
# ============================================================
print("Training Random Forest Skill Level Classifier...")

skill_data = {
    "basic": ["turn on computer", "save file", "open browser", "send email", "use keyboard"],
    "intermediate": ["insert table", "format text", "use formula", "join zoom", "change password", "attach file email"],
    "advanced": ["create chart", "use pivot table", "mail merge", "macro excel", "advanced formula"]
}

X_skill = []
y_skill = []

for level, examples in skill_data.items():
    for example in examples:
        embedding = model.encode(example)
        X_skill.append(embedding)
        y_skill.append(level)

X_skill = np.array(X_skill)
skill_encoder = LabelEncoder()
y_skill_encoded = skill_encoder.fit_transform(y_skill)

rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_skill, y_skill_encoded)
print("Random Forest Classifier trained successfully!")

# ============================================================
# K-MEANS PATTERN DETECTION
# ============================================================
print("Training K-Means Pattern Detection...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_train)
print("K-Means trained successfully!")

# ============================================================
# RECOMMENDATION SYSTEM
# ============================================================
recommendations = {
    "basic": [
        "Try learning how to save files (Ctrl+S)",
        "Learn how to send your first email",
        "Practice turning on and off your computer properly",
        "Learn how to search online using Google"
    ],
    "intermediate": [
        "Try inserting tables in Microsoft Word",
        "Learn basic Excel formulas like =SUM()",
        "Practice joining online meetings using Zoom",
        "Learn how to format documents properly"
    ],
    "advanced": [
        "Try creating charts in Excel",
        "Learn about pivot tables in Excel",
        "Practice mail merge in Microsoft Word",
        "Explore advanced Excel formulas"
    ]
}

def get_recommendation(skill_level: str) -> str:
    recs = recommendations.get(skill_level, recommendations["basic"])
    return recs[np.random.randint(0, len(recs))]

# ============================================================
# TOKENIZATION
# ============================================================
def tokenize(text: str) -> List[str]:
    tokens = text.lower().strip().split()
    tokens = [t for t in tokens if len(t) > 1]
    return tokens

# ============================================================
# NAMED ENTITY RECOGNITION (NER)
# ============================================================
digital_tools = {
    "microsoft word": "ms_word",
    "ms word": "ms_word",
    "word": "ms_word",
    "microsoft excel": "ms_excel",
    "ms excel": "ms_excel",
    "excel": "ms_excel",
    "zoom": "video_call",
    "google meet": "video_call",
    "email": "email",
    "gmail": "email",
    "google": "internet",
    "browser": "internet",
    "printer": "print",
    "password": "password",
}

def extract_entities(text: str) -> Optional[str]:
    text_lower = text.lower()
    for tool, intent in digital_tools.items():
        if tool in text_lower:
            return intent
    return None

# ============================================================
# MAIN RESPONSE FUNCTION
# ============================================================
def get_response(user_input: str, session_id: str):
    # Step 1: Tokenize
    tokens = tokenize(user_input)

    # Step 2: NER - check if specific tool mentioned
    ner_intent = extract_entities(user_input)

    # Step 3: SVM Intent Classification
    user_embedding = model.encode(user_input)
    svm_probs = svm_classifier.predict_proba([user_embedding])[0]
    svm_confidence = float(np.max(svm_probs))
    svm_intent_idx = np.argmax(svm_probs)
    svm_intent = label_encoder.inverse_transform([svm_intent_idx])[0]

    # Step 4: Combine NER + SVM (NER takes priority if found)
    final_intent = ner_intent if ner_intent else svm_intent

    # Step 5: Random Forest Skill Level Detection
    skill_probs = rf_classifier.predict_proba([user_embedding])[0]
    skill_idx = np.argmax(skill_probs)
    skill_level = skill_encoder.inverse_transform([skill_idx])[0]

    # Step 6: K-Means Pattern Detection
    cluster = int(kmeans.predict([user_embedding])[0])

    # Step 7: Get recommendation
    recommendation = get_recommendation(skill_level)

    # Step 8: Store in session
    if session_id not in sessions:
        sessions[session_id] = []

    if svm_confidence > 0.3 and final_intent in intents:
        response = intents[final_intent]["response"]
    else:
        response = "I'm sorry, I don't understand. Please try again. / Pasensya, wala ko kasabot. Palihug sulayi pag-usab."
        final_intent = "unknown"

    sessions[session_id].append({
        "user": user_input,
        "tokens": tokens,
        "bot": response,
        "intent": final_intent,
        "skill_level": skill_level,
        "confidence": svm_confidence,
        "cluster": cluster,
        "recommendation": recommendation
    })

    return response, final_intent, svm_confidence, skill_level, recommendation

# ============================================================
# API ENDPOINTS
# ============================================================
class ChatRequest(BaseModel):
    message: str
    session_id: str = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    intent: str
    confidence: float
    skill_level: str
    recommendation: str

@app.get("/health")
def health():
    return {
        "status": "running",
        "model": "paraphrase-multilingual-MiniLM-L12-v2",
        "classifiers": ["SVM", "RandomForest", "KMeans"]
    }

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    session_id = request.session_id or str(uuid.uuid4())
    response, intent, confidence, skill_level, recommendation = get_response(
        request.message, session_id
    )
    return ChatResponse(
        response=response,
        session_id=session_id,
        intent=intent,
        confidence=round(confidence, 2),
        skill_level=skill_level,
        recommendation=recommendation
    )

@app.get("/history/{session_id}")
def get_history(session_id: str):
    history = sessions.get(session_id, [])
    return {"session_id": session_id, "history": history}

@app.delete("/history/{session_id}")
def clear_history(session_id: str):
    if session_id in sessions:
        del sessions[session_id]
    return {"message": "History cleared"}

@app.get("/profile/{session_id}")
def get_profile(session_id: str):
    history = sessions.get(session_id, [])
    if not history:
        return {"session_id": session_id, "skill_level": "basic", "total_interactions": 0}
    skill_levels = [h["skill_level"] for h in history]
    most_common = max(set(skill_levels), key=skill_levels.count)
    return {
        "session_id": session_id,
        "skill_level": most_common,
        "total_interactions": len(history),
        "topics_asked": list(set([h["intent"] for h in history]))
    }