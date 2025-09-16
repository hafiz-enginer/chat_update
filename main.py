from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field, validator
from typing import List, Optional, Dict, Any, Union
import requests
import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = FastAPI(title="Shopping Chatbot API - Unified Endpoint")

# Environment variables
CATEGORY_API_URL = os.getenv("CATEGORY_API_URL")
ITEMS_API_BASE = os.getenv("ITEMS_API_BASE")
BILL_API_URL = os.getenv("BILL_API_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# OpenAI client
client = OpenAI(api_key=OPENAI_API_KEY)

# ----- Models -----
class CartItem(BaseModel):
    name: str
    quantity: int = Field(..., gt=0)
    price: float = Field(..., gt=0)

class UserDetails(BaseModel):
    name: str
    phone: str
    address: str
    payment_method: Optional[str] = None

    @validator('phone')
    def phone_valid(cls, v):
        if not (v.isdigit() and len(v) in [10, 11]):
            raise ValueError('Phone must be 10 or 11 digits')
        return v

class ChatRequest(BaseModel):
    action: Optional[str] = None
    payload: Optional[Dict[str, Any]] = None
    message: Optional[str] = None  # for raw NLP input

# ----- In-memory session and cache -----
user_session: Optional[UserDetails] = None
cart: List[CartItem] = []

categories_cache: List[str] = []
items_cache: Dict[str, List[Dict[str, Any]]] = {}

# ----- Helper functions with caching -----
def fetch_categories():
    global categories_cache
    if categories_cache:
        return categories_cache
    try:
        resp = requests.get(CATEGORY_API_URL)
        resp.raise_for_status()
        data = resp.json()
        categories_cache = [cat["categoryName"].strip() for cat in data if cat.get("isEnable")]
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

# ----- NLP analysis -----
def analyze_user_message(message: str) -> dict:
    prompt = f"""
    Tum ek shopping chatbot ho. 
    User ke message ko dekho aur decide karo ke konsa action lena hai.

    Possible actions:
    - greet
    - login
    - list_categories
    - list_items
    - add_to_cart
    - show_cart
    - checkout
    - logout

    Agar payload zaroori ho to include karo (JSON ke form mein).

    Example output:
    {{
      "action": "add_to_cart",
      "payload": {{"name": "Apple", "quantity": 2, "price": 120}}
    }}

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
        return {"action": "greet", "payload": {}}

# ----- Unified endpoint (merged) -----
@app.post("/chat")
def chat(request: ChatRequest):
    global user_session, cart

    # Determine action and payload
    if request.message:  # NLP path
        nlp_result = analyze_user_message(request.message)
        action = nlp_result.get("action", "greet").lower()
        payload = nlp_result.get("payload", {})
    elif request.action:  # Direct action path
        action = request.action.lower()
        payload = request.payload or {}
    else:
        raise HTTPException(status_code=400, detail="Invalid request: provide 'action' or 'message'")

    # ----- Action processing -----
    if action == "greet":
        return {"message": "Assalamoalikum! 🙏\n🤖: Welcome! Please login to continue shopping.\n🤖: What's your name?"}

    if action == "login":
        try:
            user = UserDetails(**payload)
            user_session = user
            cart.clear()
            return {"message": f"Welcome {user.name}!", "user": user.dict()}
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    if action == "list_categories":
        categories = fetch_categories()
        if not categories:
            raise HTTPException(status_code=503, detail="Categories service unavailable")
        return {"categories": categories}

    if action == "list_items":
        category_name = payload.get("category_name")
        if not category_name:
            raise HTTPException(status_code=400, detail="Missing 'category_name' in payload")
        items = fetch_items_by_category(category_name)
        mapped_items = [{"name": i.get("itemName", "Unknown"), "price": i.get("price") or i.get("sales") or 0} for i in items]
        return {"items": mapped_items}

    if action == "add_to_cart":
        if not user_session:
            raise HTTPException(status_code=401, detail="User not logged in")
        try:
            item = CartItem(**payload)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        for existing_item in cart:
            if existing_item.name == item.name:
                existing_item.quantity += item.quantity
                return {"message": f"Updated {item.name} quantity to {existing_item.quantity}"}
        cart.append(item)
        return {"message": f"Added {item.quantity} x {item.name} to cart"}

    if action == "show_cart":
        if not cart:
            return {"message": "🛒 Your cart is empty."}
        total = sum(item.quantity * item.price for item in cart)
        items_summary = [{"name": i.name, "quantity": i.quantity, "price": i.price, "subtotal": i.quantity * i.price} for i in cart]
        return {"items": items_summary, "total": total}

    if action == "checkout":
        if not user_session:
            raise HTTPException(status_code=401, detail="User not logged in")
        if not cart:
            raise HTTPException(status_code=400, detail="Cart is empty")
        pm = payload.get("payment_method")
        if pm not in ["Cash on Delivery", "Online Transfer"]:
            raise HTTPException(status_code=400, detail="Invalid payment method")
        user_session.payment_method = pm
        payload_bill = {"user": user_session.dict(), "items": [item.dict() for item in cart]}
        try:
            resp = requests.post(BILL_API_URL, json=payload_bill)
            resp.raise_for_status()
            bill_response = resp.json()
        except Exception as e:
            raise HTTPException(status_code=502, detail=f"Billing API error: {e}")
        cart.clear()
        return {"message": "Checkout successful", "payment_method": pm, "bill": bill_response.get("bill", {})}

    if action == "logout":
        user_session = None
        cart.clear()
        return {"message": "Logged out and cart cleared."}

    raise HTTPException(status_code=400, detail="Invalid action")
