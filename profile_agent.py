import json
import logging
import re
from typing import Optional, Dict

from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq

# Logging setup
logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

# Initialize the LLM
llm = ChatGroq(
    model="llama3-70b-8192",
    api_key="Your Groq api key here"
)

# Define prompt to extract user details
prompt = ChatPromptTemplate.from_template("""
You're a friendly chatbot helg collect user details for government scheme recommendations.

Instructions:
- Ask questions one by one conversationally to collect: name, state, gender, caste, occupation, category (like SC/ST/OBC), income.
- Be warm and friendly.
- At the end, respond with a JSON like:
{{
    "name": "...",
    "state": "...",
    "gender": "...",
    "caste": "...",
    "occupation": "...",
    "category": "...",
    "income": "...",
    "additional_details": "..."  # optional
}}

Conversation so far:
{chat_history}
User: {user_input}
Bot:
""")

chain: Runnable = prompt | llm | StrOutputParser()

# Store user info
user_info: Dict[str, Optional[str]] = {
    "name": None,
    "state": None,
    "gender": None,
    "caste": None,
    "occupation": None,
    "category": None,
    "income": None,
    "additional_details": None
}

# Utility to parse JSON from mixed output
def extract_json(text: str) -> Dict[str, Optional[str]]:
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group())
        except json.JSONDecodeError:
            return {}
    return {}

def collect_user_info():
    print("👋 Welcome to the Government Scheme Finder chatbot!")
    print("Let's get to know you. You can say something like: 'I'm Priya from Tamil Nadu, my income is under 1 lakh.'")

    chat_history = ""
    additional_info_phase = False
    awaiting_additional_confirmation = False

    while True:
        # Check for missing required fields
        required_fields = ["name", "state", "gender", "caste", "occupation", "category", "income"]
        missing_fields = [field for field in required_fields if not user_info.get(field)]

        # If all required fields are collected and not already asking for additional details
        if not missing_fields and not additional_info_phase and not awaiting_additional_confirmation:
            print("Bot: Thanks! I’ve collected everything I need for the basic recommendation.")
            print("Bot: Would you like to add any additional details (like education, land, family background, etc.)? (yes/no)")
            awaiting_additional_confirmation = True

        # Prompt for input
        user_input = input("\nYou: ").strip().lower()

        # Handle additional info phase
        if additional_info_phase:
            if user_input in ["no", "nah", "nope", "not", "dont want", "i dont want to"]:
                break
            else:
                existing = user_info.get("additional_details", "")
                user_info["additional_details"] = f"{existing}, {user_input}".strip(", ")
                print("Bot: Got it! ✅ You can add more, or say 'no' to finish.")
                continue

        # Handle additional info confirmation
        if awaiting_additional_confirmation:
            if user_input in ["yes", "y", "sure"]:
                additional_info_phase = True
                awaiting_additional_confirmation = False
                print("Bot: Great! Go ahead and tell me more. You can add as much as you want, and say 'no' when done.")
                continue
            elif user_input in ["no", "nah", "nope", "not", "dont want", "i dont want to"]:
                break
            else:
                print("Bot: Please say 'yes' or 'no' (or something like 'not' or 'I don’t want to') to let me know if you want to add more details.")
                continue

        # Normal LLM interaction for required fields
        if missing_fields:
            response = chain.invoke({"user_input": user_input, "chat_history": chat_history})
            chat_history += f"\nUser: {user_input}\nBot: {response}"
            # Parse JSON response
            json_data = extract_json(response)
            if json_data:
                user_info.update({k: v for k, v in json_data.items() if v})
            print(f"Bot: {response.strip()}")

    # Save to file
    with open("user_details.json", "w") as f:
        json.dump(user_info, f, indent=2)

    print("\n✅ All details saved to user_details.json")
    print("Bot: You’re a star! I’ve got all I need to find you awesome schemes!")

def get_user_profile_via_chat() -> Dict[str, Optional[str]]:
    collect_user_info()
    return user_info

if __name__ == "__main__":
    get_user_profile_via_chat()