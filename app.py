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

# Intents specific to Barangay Officials' Digital Literacy tasks (English and Cebuano/Bisaya)
intents = {
    "greet": {
        "examples": ["hello", "hi", "good morning", "kamusta", "maayong buntag", "maayong hapon", "maayong gabii", "hey barangai"],
        "response": "Maayong adlaw! I am BarangAI, your digital assistant. I can help you with MS Word (Clearances, Certificates), MS Excel (Liquidations, Budget), Emails, and Printing. Unsaon nako pagtabang kanimo karon?",
        "difficulty": "basic"
    },
    "basic_computer": {
        "examples": ["how to turn on computer", "how to restart", "unsaon pag-on sa computer", "unsaon pag-restart", "how to shut down", "unsaon pag-off sa computer", "nag hang ang computer"],
        "response": "To turn on, press the power button. If the computer freezes (nag-hang), press Ctrl+Alt+Delete or hold the power button to restart. / Kung nag-hang ang computer, pindota ang Ctrl+Alt+Delete o hupti ang power button aron ma-restart.",
        "difficulty": "basic"
    },
    "save_document": {
        "examples": ["how to save file", "unsaon pag save sa clearance", "how to save document", "save indigency file", "unsaon pagluwas sa file", "dili ma save ang file"],
        "response": "Always save official documents! Press Ctrl+S or click File > Save. To save a new version of a clearance, use File > Save As. / Kanunay i-save ang mga dokumento sa barangay! Pindota ang Ctrl+S o i-click ang File > Save.",
        "difficulty": "basic"
    },
    "ms_word_clearance": {
        "examples": ["how to make barangay clearance", "unsaon paghimo ug barangay clearance sa word", "certificate of indigency", "unsaon pag buhat og certificate", "format clearance", "business permit word"],
        "response": "To make a Barangay Clearance/Indigency in Word: Open a blank document, center the alignment (Ctrl+E) for the Barangay Header, and use standard font like Arial or Times New Roman size 12. / Aron maghimo ug clearance: Pag-abli ug blangko nga dokumento, i-center (Ctrl+E) ang header sa barangay.",
        "difficulty": "intermediate"
    },
    "ms_word_table_blotter": {
        "examples": ["how to insert table in word", "unsaon pagsal-ot ug table sa word", "unsaon pag buhat og table sa word", "table for blotter report", "add table for attendance", "unsaon paghimo ug table"],
        "response": "For blotter logs or attendance sheets: Click Insert > Table > select the number of columns and rows you need for the report. / Aron maghimo ug table para sa blotter o attendance: I-click ang Insert > Table > pilia ang gidaghanon sa rows ug columns.",
        "difficulty": "intermediate"
    },
    "ms_word_format": {
        "examples": ["how to bold name in certificate", "unsaon pag bold", "change font size clearance", "how to change font", "unsaon pagbag-o sa font", "underline text"],
        "response": "To format names on certificates: Select the text, press Ctrl+B to make it Bold, Ctrl+U to Underline, or use the top menu to change font size. / Aron mag-format ug pangalan sa sertipiko: Pilia ang teksto, pindota ang Ctrl+B para mo-Bold, o Ctrl+U para Underline.",
        "difficulty": "intermediate"
    },
    "ms_excel_liquidation": {
        "examples": ["how to use excel for budget", "unsaon pag compute sa budget sa excel", "financial liquidation excel", "how to sum in excel", "unsaon pag total sa excel", "excel formula for expenses"],
        "response": "For barangay financial liquidations, use the AutoSum tool. Click the empty cell below your expenses, type =SUM(, select the numbers, and press Enter. / Para sa liquidation sa barangay, i-type ang =SUM(, pilia ang mga kantidad, ug pindota ang Enter aron makuha ang total.",
        "difficulty": "intermediate"
    },
    "ms_excel_table_inventory": {
        "examples": ["how to insert table in excel", "unsaon pagbuhat ug table sa excel", "unsaon pag buhat og table sa excel", "inventory of barangay assets", "create table in spreadsheet", "excel format table"],
        "response": "To make an inventory of barangay assets in Excel: Highlight your typed data, click Insert > Table, and make sure 'My table has headers' is checked. / Aron maghimo ug inventory table: Pilia ang imong datos, i-click ang Insert > Table, ug i-check ang 'My table has headers'.",
        "difficulty": "intermediate"
    },
    "ms_excel_chart_demographics": {
        "examples": ["how to create chart in excel", "unsaon paghimo ug chart sa excel", "graph for population", "chart for barangay demographics", "pie chart excel"],
        "response": "To visualize barangay population or demographics: Select your data table, click Insert > Chart, and choose a Pie Chart or Bar Graph. / Aron maghimo ug chart para sa populasyon: Pilia ang datos, i-click ang Insert > Chart.",
        "difficulty": "advanced"
    },
    "email_dilg_city": {
        "examples": ["how to send email to dilg", "unsaon pag padala ug email", "send report to city hall email", "attach file to email", "unsaon pag attach ug file", "compose email"],
        "response": "To send a report via Email: Click Compose, type the official email address (e.g., DILG or City Hall), add a Subject, click the Paperclip icon to attach your Word/Excel file, and click Send. / Aron magpadala ug report: I-click ang Compose, ibutang ang email, i-click ang Paperclip icon aron i-attach ang file, ug i-click ang Send.",
        "difficulty": "intermediate"
    },
    "internet_search_memos": {
        "examples": ["how to search online", "unsaon pagpangita online", "search dilg memo", "find official guidelines google", "how to use google", "unsaon pag search sa memo"],
        "response": "To search for official memos or guidelines: Open Google Chrome, type your keywords (e.g., 'DILG memorandum circular 2026') in the search bar, and press Enter. / Aron mangita ug memo: Ablihi ang Google, i-type ang imong gipangita sa search bar, ug pindota ang Enter.",
        "difficulty": "basic"
    },
    "video_call_seminar": {
        "examples": ["how to use zoom", "unsaon paggamit sa zoom", "join dilg online meeting", "google meet help", "unsaon pagsalmot sa zoom meeting", "online seminar video call"],
        "response": "To join an online DILG seminar or meeting: Click the Zoom/Google Meet link provided in your email or messenger, enter your name (e.g., Brgy. Tisa - Secretary), and click Join. / Aron mosalmot sa online meeting: I-click ang link, isulod ang imong ngalan, ug i-click ang Join.",
        "difficulty": "intermediate"
    },
    "password_security": {
        "examples": ["how to change password", "unsaon pagbag-o sa password", "forgot password email", "nakalimtan ang password sa portal", "secure barangay account"],
        "response": "For security, barangay accounts should have strong passwords! Go to Settings > Security > Change Password. Use a mix of letters, numbers, and symbols. / Alang sa seguridad, adto sa Settings > Security > Change Password. Paggamit ug sagol nga letra ug numero.",
        "difficulty": "basic"
    },
    "print_certificate": {
        "examples": ["how to print document", "unsaon pag-print sa clearance", "unsaon pag print sa blotter", "printer not working", "print certificate of indigency", "print setup"],
        "response": "To print a clearance or certificate: Press Ctrl+P or click File > Print. Make sure the correct barangay printer is selected and the paper size matches (A4 or Letter) before clicking Print. / Aron mag-print ug clearance: Pindota ang Ctrl+P. Siguruha nga sakto ang printer ug size sa papel una i-click ang Print.",
        "difficulty": "basic"
    }
}



# SVM INTENT CLASSIFIER - Train on startup

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


# RANDOM FOREST SKILL LEVEL CLASSIFIER

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


# K-MEANS PATTERN DETECTION

print("Training K-Means Pattern Detection...")
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
kmeans.fit(X_train.astype(np.float64))
print("K-Means trained successfully!")


# RECOMMENDATION SYSTEM

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


# TOKENIZATION
 
def tokenize(text: str) -> List[str]:
    tokens = text.lower().strip().split()
    tokens = [t for t in tokens if len(t) > 1]
    return tokens

 
# NAMED ENTITY RECOGNITION (NER)
 
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


# MAIN RESPONSE FUNCTION

def get_response(user_input: str, session_id: str):
    # Tokenize
    tokens = tokenize(user_input)

    # NER - check if specific tool mentioned
    ner_intent = extract_entities(user_input)

    # SVM Intent Classification
    user_embedding = model.encode(user_input)
    
    # 1. FIX K-MEANS: Create a strict 2D float64 numpy array right away
    user_embedding_2d = np.array([user_embedding], dtype=np.float64)

    svm_probs = svm_classifier.predict_proba(user_embedding_2d)[0]
    svm_confidence = float(np.max(svm_probs))
    svm_intent_idx = np.argmax(svm_probs)
    
    # 2. FIX PYDANTIC: Wrap outputs in str()
    svm_intent = str(label_encoder.inverse_transform([svm_intent_idx])[0])

    # Combine NER + SVM (SVM takes priority since it understands context!)
    if svm_confidence > 0.1:
        final_intent = str(svm_intent)
    else:
        final_intent = str(ner_intent if ner_intent else "unknown")

    # Random Forest Skill Level Detection
    skill_probs = rf_classifier.predict_proba(user_embedding_2d)[0]
    skill_idx = np.argmax(skill_probs)
    
    # FIX PYDANTIC: Wrap in str()
    skill_level = str(skill_encoder.inverse_transform([skill_idx])[0])

    # K-Means Pattern Detection
    cluster = int(kmeans.predict(user_embedding_2d.astype(np.float64))[0])

    recommendation = str(get_recommendation(skill_level))

    # Store in session
    if session_id not in sessions:
        sessions[session_id] = []

    if svm_confidence > 0.1 and final_intent in intents:
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


# API ENDPOINTS

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