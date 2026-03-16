from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util
from typing import List, Dict
import uuid

app = FastAPI(title="BarangAI Chatbot API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Conversation sessions storage (Dialogue State Tracker)
sessions: Dict[str, List[Dict]] = {}

# Intents in English and Cebuano/Bisaya
intents = {
    "greet": {
        "examples": ["hello", "hi", "good morning", "kamusta", "maayong buntag", "maayong hapon", "maayong gabii", "hey"],
        "response": "Hello! How can I help you today? / Kamusta! Unsaon nako pagtabang kanimo?"
    },
    "basic_computer": {
        "examples": ["how to turn on computer", "how to restart", "unsaon pag-on sa computer", "unsaon pag-restart", "how to shut down", "unsaon pag-off sa computer", "computer not working"],
        "response": "To turn on your computer, press the power button. To restart, click Start > Restart. To shut down, click Start > Shut Down. / Aron i-on ang computer, pindota ang power button. Aron i-restart, i-click ang Start > Restart. Aron i-off, i-click ang Start > Shut Down."
    },
    "save_file": {
        "examples": ["how to save file", "unsaon pagluwas sa file", "how to save document", "save file help", "unsaon pag save", "dili ma save ang file"],
        "response": "To save a file, press Ctrl+S or click File > Save. To save as a new file, press Ctrl+Shift+S or click File > Save As. / Aron magluwas sa file, pindota ang Ctrl+S o i-click ang File > Save. Aron mag-save as, pindota ang Ctrl+Shift+S."
    },
    "ms_word": {
        "examples": ["how to use word", "paano gamitin ang word", "unsaon paggamit sa word", "microsoft word help", "word document", "how to format document", "unsaon pag format sa dokumento", "word tutorial"],
        "response": "I can help you with Microsoft Word! You can create documents, format text, insert tables and images. What specific help do you need? / Makatabang ko sa Microsoft Word! Makabuhat ka ug dokumento, mag-format ug teksto, mag-insert ug table ug hulagway. Unsay imong gikinahanglan?"
    },
    "ms_word_table": {
        "examples": ["how to insert table in word", "unsaon pagsal-ot ug table sa word", "add table word", "table in document", "unsaon paghimo ug table"],
        "response": "To insert a table in Word: Click Insert > Table > select rows and columns. You can also draw a table by clicking Insert > Table > Draw Table. / Aron magsal-ot ug table sa Word: I-click ang Insert > Table > pilia ang gidaghanon sa rows ug columns."
    },
    "ms_word_format": {
        "examples": ["how to change font", "unsaon pagbag-o sa font", "how to bold text", "unsaon pag bold", "how to change font size", "text formatting word"],
        "response": "To format text in Word: Select the text first, then use the toolbar to change font, size, bold (Ctrl+B), italic (Ctrl+I), or underline (Ctrl+U). / Aron mag-format ug teksto sa Word: Pilion una ang teksto, unya gamiton ang toolbar aron mabag-o ang font, gidak-on, bold (Ctrl+B), italic (Ctrl+I), o underline (Ctrl+U)."
    },
    "ms_excel": {
        "examples": ["how to use excel", "paano gamitin ang excel", "unsaon paggamit sa excel", "spreadsheet help", "excel help", "excel tutorial"],
        "response": "I can help you with Microsoft Excel! Excel is used for spreadsheets, calculations, and data management. What do you need help with? / Makatabang ko sa Microsoft Excel! Ang Excel gamiton sa spreadsheets, kalkulasyon, ug pagdumala sa datos. Unsay imong gikinahanglan?"
    },
    "ms_excel_formula": {
        "examples": ["how to use formula in excel", "unsaon paggamit sa formula sa excel", "sum formula", "excel calculation", "unsaon pag calculate sa excel", "excel sum", "average formula"],
        "response": "To use formulas in Excel: Start with = sign. Common formulas: =SUM(A1:A5) to add, =AVERAGE(A1:A5) for average, =COUNT(A1:A5) to count. / Aron mogamit ug formula sa Excel: Sugdi sa = sign. Kasagarang formula: =SUM(A1:A5) aron magdugang, =AVERAGE(A1:A5) alang sa average, =COUNT(A1:A5) aron magihap."
    },
    "ms_excel_chart": {
        "examples": ["how to create chart in excel", "unsaon paghimo ug chart sa excel", "graph excel", "chart excel", "data visualization excel"],
        "response": "To create a chart in Excel: Select your data, click Insert > Chart, choose chart type (bar, line, pie). / Aron maghimo ug chart sa Excel: Pilion ang imong datos, i-click ang Insert > Chart, pilia ang klase sa chart (bar, line, pie)."
    },
    "email": {
        "examples": ["how to send email", "paano mag padala ng email", "unsaon pag padala ug email", "email help", "compose email", "how to attach file in email", "unsaon pag attach sa email"],
        "response": "To send an email: Open your email app > Click Compose > Enter recipient email > Add subject > Write message > Click Send. To attach a file, click the paperclip icon. / Aron magpadala ug email: Ablihi ang email app > I-click ang Compose > Isulod ang email sa tigdawat > Idugang ang subject > Isulat ang mensahe > I-click ang Send."
    },
    "internet": {
        "examples": ["how to search online", "unsaon pagpangita online", "how to use google", "unsaon paggamit sa google", "internet help", "how to browse internet", "unsaon paggamit sa internet"],
        "response": "To search online: Open your browser (Chrome, Edge) > Go to google.com > Type your search query > Press Enter. For reliable information, check government websites (.gov.ph) or educational sites (.edu). / Aron mangita online: Ablihi ang imong browser > Adto sa google.com > I-type ang imong pangutana > Pindota ang Enter."
    },
    "video_call": {
        "examples": ["how to use zoom", "unsaon paggamit sa zoom", "google meet help", "video call help", "online meeting", "unsaon pagsalmot sa zoom meeting"],
        "response": "To join a Zoom meeting: Click the meeting link or open Zoom > Enter Meeting ID > Enter Password if required > Click Join. For Google Meet: Click the meeting link > Click Join Now. / Aron mosalmot sa Zoom meeting: I-click ang meeting link o ablihi ang Zoom > Isulod ang Meeting ID > Isulod ang Password kung gikinahanglan > I-click ang Join."
    },
    "password": {
        "examples": ["how to change password", "unsaon pagbag-o sa password", "forgot password", "nakalimtan ang password", "password help"],
        "response": "To change your password: Go to Settings > Account > Change Password. Enter your current password, then your new password. Make sure your password is strong with letters, numbers, and symbols. / Aron mabag-o ang password: Adto sa Settings > Account > Change Password. Isulod ang imong kasamtangang password, unya ang bag-ong password."
    },
    "print": {
        "examples": ["how to print document", "unsaon pag-print sa dokumento", "print help", "printer not working", "unsaon paggamit sa printer"],
        "response": "To print a document: Press Ctrl+P or click File > Print. Select your printer, set the number of copies, and click Print. Make sure the printer is connected and has paper. / Aron mag-print ug dokumento: Pindota ang Ctrl+P o i-click ang File > Print. Pilia ang imong printer, itakda ang gidaghanon sa kopya, unya i-click ang Print."
    },
    "goodbye": {
        "examples": ["bye", "goodbye", "thank you", "salamat", "ok thanks", "paalam", "daghang salamat"],
        "response": "Goodbye! Feel free to ask again anytime! / Paalam! Ayaw kahadlok pangutana bisan kanus-a!"
    },
}

class ChatRequest(BaseModel):
    message: str
    session_id: str = None

class ChatResponse(BaseModel):
    response: str
    session_id: str
    intent: str
    confidence: float

def get_response(user_input: str, session_id: str):
    best_match = None
    best_score = 0

    user_embedding = model.encode(user_input)

    for intent, data in intents.items():
        for example in data["examples"]:
            example_embedding = model.encode(example)
            score = util.cos_sim(user_embedding, example_embedding).item()
            if score > best_score:
                best_score = score
                best_match = intent

    # Store conversation
    if session_id not in sessions:
        sessions[session_id] = []

    if best_score > 0.4:
        response = intents[best_match]["response"]
    else:
        response = "I'm sorry, I don't understand. Please try again. / Pasensya, wala ko kasabot. Palihug sulayi pag-usab."
        best_match = "unknown"

    # Save to conversation history
    sessions[session_id].append({
        "user": user_input,
        "bot": response,
        "intent": best_match,
        "confidence": best_score
    })

    return response, best_match, best_score

@app.get("/health")
def health():
    return {"status": "running", "model": "paraphrase-multilingual-MiniLM-L12-v2"}

@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest):
    # Generate session ID if not provided
    session_id = request.session_id or str(uuid.uuid4())
    response, intent, confidence = get_response(request.message, session_id)
    return ChatResponse(
        response=response,
        session_id=session_id,
        intent=intent,
        confidence=round(confidence, 2)
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