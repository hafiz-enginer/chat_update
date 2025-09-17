# --------------------------
# ✅ Shopping Chatbot API with CLI-style flow + Streamlit UI
# --------------------------
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any
import requests
import os
import json
from dotenv import load_dotenv
from openai import OpenAI
from starlette.middleware.cors import CORSMiddleware

load_dotenv()

app = FastAPI(title="Shopping Chatbot API - Unified Endpoint")
# ✅ Allow frontend origin(s)
origins = [
    "http://localhost:5173",   # local dev
    "https://chat-bot-fq96.vercel.app",  # your deployed frontend
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,          # or ["*"] for all (not recommended in prod)
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# -------- Environment --------
CATEGORY_API_URL = os.getenv("CATEGORY_API_URL")
ITEMS_API_BASE = os.getenv("ITEMS_API_BASE")
BILL_API_URL = os.getenv("BILL_API_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

client = OpenAI(api_key=OPENAI_API_KEY)

# -------- Models --------
class CartItem(BaseModel):
    name: str
    quantity: int = Field(..., gt=0)
    price: float = Field(..., gt=0)

class UserDetails(BaseModel):
    name: str
    phone: str
    address: str
    payment_method: Optional[str] = None

    @validator("phone")
    def phone_valid(cls, v):
        if not (v.isdigit() and len(v) in [10, 11]):
            raise ValueError("Phone must be 10 or 11 digits")
        return v

class ChatRequest(BaseModel):
    action: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    message: Optional[str] = None

# -------- Memory --------
user_session: Optional[UserDetails] = None
cart: List[CartItem] = []
pending_login: Dict[str, str] = {}
user_lang: str = "en"

categories_cache: List[str] = []
items_cache: Dict[str, List[Dict[str, Any]]] = {}

# -------- Helpers --------
def fetch_categories():
    global categories_cache
    if categories_cache:
        return categories_cache
    try:
        resp = requests.get(CATEGORY_API_URL)
        resp.raise_for_status()
        data = resp.json()
        categories_cache = [
            cat["categoryName"].strip() for cat in data if cat.get("isEnable")
        ]
        return categories_cache
    except Exception:
        return []

def fetch_items_by_category(cat_name: str):
    global items_cache
    if cat_name in items_cache:
        return items_cache[cat_name]
    try:
        url = f"{ITEMS_API_BASE}/{cat_name.strip()}"
        resp = requests.get(url)
        resp.raise_for_status()
        items = resp.json()
        items_cache[cat_name] = items
        return items
    except Exception:
        return []

def format_cart_summary():
    if not cart:
        return "🛒 Cart is empty." if user_lang == "en" else "🛒 Cart khaali hai."
    lines = []
    total = 0
    for idx, item in enumerate(cart, start=1):
        subtotal = item.quantity * item.price
        total += subtotal
        lines.append(f"{idx}. {item.name} x {item.quantity} = Rs{subtotal}")
    lines.append(f"\n➡️ Running Total: Rs{total}")
    return "\n".join(lines)

# -------- NLP Rules --------
def analyze_user_message(message: str) -> dict:
    lowered = message.lower()

    # ✅ Smart show_cart detection
    show_cart_keywords = ["cart", "basket", "order", "saman", "meri shopping", "mere saman"]
    if any(word in lowered for word in show_cart_keywords):
        return {"action": "show_cart", "payload": {}, "lang": "en"}

    # ✅ Smart checkout detection
    checkout_keywords = ["checkout", "payment", "order complete", "order kar do", "payment karni hai"]
    if any(word in lowered for word in checkout_keywords):
        return {"action": "checkout", "payload": {}, "lang": "en"}

    # ✅ Smart add_to_cart detection (simple rule)
    if "add" in lowered or "lo" in lowered or "dal" in lowered:
        return {"action": "add_to_cart", "payload": {}, "lang": "en"}

    # ✅ NEW FIX: detect categories directly from message
    categories = fetch_categories()
    for cat in categories:
        if cat.lower() in lowered:  # agar category ka naam message me ho
            return {
                "action": "list_items",
                "payload": {"category_name": cat},
                "lang": "en"
            }

    # ---- GPT based analysis (fallback) ----
    prompt = f"""
    Tum ek shopping chatbot ho. User ke message ko dekho aur decide karo ke konsa action lena hai.

    Possible actions:
    - greet
    - login_progress (jab user apna naam, phone ya address de raha ho)
    - login (jab teeno fields complete ho jayein)
    - list_categories
    - list_items
    - add_to_cart
    - show_cart
    - checkout
    - logout

    Saath hi user ki language detect karo:
    - Agar user english use kar raha hai → "lang": "en"
    - Agar user roman urdu use kar raha hai → "lang": "ur"

    Example:
    User: "mera naam hamid hai"
    Response:
    {{"action": "login_progress", "payload": {{"name": "Hamid"}}, "lang": "ur"}}

    User: "my phone number is 03124567896"
    Response:
    {{"action": "login_progress", "payload": {{"phone": "03124567896"}}, "lang": "en"}}

    User message: "{message}"
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2
    )
    try:
        return json.loads(response.choices[0].message.content)
    except:
        return {"action": "greet", "payload": {}, "lang": "en"}

# -------- API --------
@app.post("/chat")
def chat(request: ChatRequest):
    global user_session, cart, pending_login, user_lang

    # detect action
    if request.message:
        nlp = analyze_user_message(request.message)
        action = nlp.get("action", "greet").lower()
        payload = nlp.get("payload", {})
        user_lang = nlp.get("lang", user_lang)
    elif request.action:
        action = request.action.lower()
        payload = request.payload or {}
    else:
        raise HTTPException(400, "Invalid request")

    # ---- actions ----
    if action == "greet":
        return {"message": "🤖 Welcome! What's your name?"}

    if action == "login_progress":
        pending_login.update(payload)
        missing = [f for f in ["name", "phone", "address"] if f not in pending_login]
        if not missing:
            try:
                user = UserDetails(**pending_login)
                user_session = user
                cart.clear()
                pending_login.clear()
                return {"message": f"🎉 Welcome {user.name}! You're logged in."}
            except Exception as e:
                return {"message": f"❌ Error: {e}"}
        next_field = missing[0]
        q = {
            "name": "What's your name?",
            "phone": "📞 Enter phone number (10/11 digits):",
            "address": "🏠 Enter your address:",
        }
        return {"message": q[next_field]}

    if action == "list_categories":
        cats = fetch_categories()
        if not cats:
            raise HTTPException(503, "No categories available")
        text = "Please select a category:\n" + "\n".join(
            [f"{i+1}. {c}" for i, c in enumerate(cats)]
        )
        return {"categories": cats, "message": text}

    if action == "list_items":
        cat = payload.get("category_name")
        if not cat:
            raise HTTPException(400, "Missing 'category_name'")
        items = fetch_items_by_category(cat)
        mapped = [
            {"name": i.get("itemName", "Unknown"), "price": i.get("price") or 0}
            for i in items
        ]
        text = f"Items in '{cat}':\n" + "\n".join(
            [f"{i+1}. {x['name']} - Rs{x['price']}" for i, x in enumerate(mapped)]
        )
        text += "\n(type item number to add, 'back' for categories, 'checkout' to finish)"
        return {"items": mapped, "message": text}

    if action == "add_to_cart":
        if not user_session:
            raise HTTPException(401, "Login required")
        try:
            item = CartItem(**payload)
        except Exception as e:
            raise HTTPException(400, str(e))
        for existing in cart:
            if existing.name == item.name:
                existing.quantity += item.quantity
                return {"message": f"✅ Updated {item.name} qty = {existing.quantity}\n\n🧾 {format_cart_summary()}"}
        cart.append(item)
        return {"message": f"✅ Added {item.quantity} x {item.name}\n\n🧾 {format_cart_summary()}"}

    if action == "show_cart":
        return {"message": format_cart_summary()}

    if action == "checkout":
        if not user_session:
            raise HTTPException(401, "Login required")
        if not cart:
            raise HTTPException(400, "Cart is empty")
        pm = payload.get("payment_method")
        if not pm:
            return {"message": "💳 Choose payment: 1. Cash on Delivery  2. Online Transfer"}
        user_session.payment_method = pm
        bill_payload = {"user": user_session.dict(), "items": [i.dict() for i in cart]}
        try:
            resp = requests.post(BILL_API_URL, json=bill_payload)
            resp.raise_for_status()
            bill = resp.json()
        except Exception as e:
            raise HTTPException(502, f"Billing error: {e}")
        cart.clear()
        return {"message": f"✅ Checkout complete!\n\n🧾 Bill: {bill}"}

    if action == "logout":
        user_session = None
        cart.clear()
        pending_login.clear()
        return {"message": "👋 Logged out & cart cleared."}

    raise HTTPException(400, "Invalid action")

# --------------------------
# ✅ Streamlit Test UI
# --------------------------
import streamlit as st

def run_ui():
    st.set_page_config(page_title="🛒 Shopping Assistant Chatbot", layout="centered")
    st.title("🛒 Shopping Assistant Chatbot (Test UI)")

    if "messages" not in st.session_state:
        st.session_state.messages = []

    if st.sidebar.button("🔄 New Conversation"):
        st.session_state.messages = []

    for msg in st.session_state.messages:
        st.chat_message(msg["role"]).markdown(msg["content"])

    if prompt := st.chat_input("Type your message..."):
        st.chat_message("user").markdown(prompt)
        st.session_state.messages.append({"role": "user", "content": prompt})

        try:
            res = requests.post("http://localhost:8000/chat", json={"message": prompt})
            if res.status_code == 200:
                data = res.json()
                reply = data.get("message", str(data))
            else:
                reply = f"⚠️ Error {res.status_code}: {res.text}"
        except Exception as e:
            reply = f"❌ Backend error: {e}"

        st.chat_message("assistant").markdown(reply)
        st.session_state.messages.append({"role": "assistant", "content": reply})

if __name__ == "__main__":
    run_ui()
