from fastapi import FastAPI
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer, util

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Load the multilingual model
model = SentenceTransformer('sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2')

# Intents in English and Cebuano/Bisaya
intents = {
    "greet": {
        "examples": ["hello", "hi", "good morning", "kamusta", "maayong buntag", "maayong hapon", "maayong gabii"],
        "response": "Hello! How can I help you today? / Kamusta! Unsaon nako pagtabang kanimo?"
    },
    "basic_computer": {
        "examples": ["how to turn on computer", "how to restart", "unsaon pag-on sa computer", "unsaon pag-restart", "how to shut down", "unsaon pag-off sa computer"],
        "response": "To turn on your computer, press the power button. To restart, click Start > Restart. / Aron i-on ang computer, pindota ang power button. Aron i-restart, i-click ang Start > Restart."
    },
    "save_file": {
        "examples": ["how to save file", "unsaon pagluwas sa file", "how to save document", "save file help", "unsaon pag save"],
        "response": "To save a file, press Ctrl+S or click File > Save. / Aron magluwas sa file, pindota ang Ctrl+S o i-click ang File > Save."
    },
    "ms_word": {
        "examples": ["how to use word", "paano gamitin ang word", "unsaon paggamit sa word", "microsoft word help", "word document", "how to format document"],
        "response": "I can help you with Microsoft Word! What do you need help with? / Makatabang ko sa Microsoft Word! Unsay imong gikinahanglan?"
    },
    "ms_word_table": {
        "examples": ["how to insert table in word", "unsaon pagsal-ot ug table sa word", "add table word", "table in document"],
        "response": "To insert a table in Word, click Insert > Table. / Aron magsal-ot ug table sa Word, i-click ang Insert > Table."
    },
    "ms_excel": {
        "examples": ["how to use excel", "paano gamitin ang excel", "unsaon paggamit sa excel", "spreadsheet help", "excel help"],
        "response": "I can help you with Microsoft Excel! / Makatabang ko sa Microsoft Excel!"
    },
    "ms_excel_formula": {
        "examples": ["how to use formula in excel", "unsaon paggamit sa formula sa excel", "sum formula", "excel calculation"],
        "response": "To use a formula in Excel, start with = sign. Example: =SUM(A1:A5) / Aron mogamit ug formula sa Excel, sugdi sa = sign. Pananglitan: =SUM(A1:A5)"
    },
    "email": {
        "examples": ["how to send email", "paano mag padala ng email", "unsaon pag padala ug email", "email help", "compose email"],
        "response": "To send an email, click Compose, enter recipient, subject, message then click Send. / Aron magpadala ug email, i-click ang Compose, isulod ang email sa tigdawat, subject, ug mensahe, unya i-click ang Send."
    },
    "internet": {
        "examples": ["how to search online", "unsaon pagpangita online", "how to use google", "unsaon paggamit sa google", "internet help"],
        "response": "To search online, open your browser, go to google.com and type your query. / Aron mangita online, ablihi ang imong browser, adto sa google.com ug i-type ang imong pangutana."
    },
    "video_call": {
        "examples": ["how to use zoom", "unsaon paggamit sa zoom", "google meet help", "video call help", "online meeting"],
        "response": "To join a Zoom meeting, click the meeting link or open Zoom and enter the Meeting ID. / Aron mosalmot sa Zoom meeting, i-click ang meeting link o ablihi ang Zoom ug isulod ang Meeting ID."
    },
    "goodbye": {
        "examples": ["bye", "goodbye", "thank you", "salamat", "ok thanks", "paalam"],
        "response": "Goodbye! Feel free to ask again anytime! / Paalam! Ayaw kahadlok pangutana bisan kanus-a!"
    },
}

class ChatRequest(BaseModel):
    message: str

def get_response(user_input):
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
    
    if best_score > 0.4:
        return intents[best_match]["response"]
    else:
        return "I'm sorry, I don't understand. Please try again. / Pasensya, wala ko kasabot. Palihug sulayi pag-usab."

@app.get("/health")
def health():
    return {"status": "running"}

@app.post("/chat")
def chat(request: ChatRequest):
    response = get_response(request.message)
    return {"response": response}