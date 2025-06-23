import json
import logging
import re
from typing import Optional, Dict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import Runnable
from langchain_groq import ChatGroq


# Logging setup
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize the LLM
llm = ChatGroq(
    model="llama3-70b-8192",
    api_key="Your Groq api key here"
)

# Define prompt to extract user details
prompt = ChatPromptTemplate.from_template("""
You're a friendly chatbot helping collect user details for government scheme recommendations.

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
        user_input = input("\nYou: ")
        logger.info(f"Processing user input: {user_input}")

        # Phase: User wants to add more info
        if additional_info_phase:
            if user_input.strip().lower() in ["no", "nah", "nope"]:
                break
            else:
                existing = user_info.get("additional_details", "")
                user_info["additional_details"] = f"{existing}, {user_input.strip()}".strip(", ")
                print("Bot: Got it! ✅ You can add more, or say 'no' to finish.")
                continue

        # Normal LLM interaction
        response = chain.invoke({"user_input": user_input, "chat_history": chat_history})
        logger.info(f"LLM raw response: {response}")

        chat_history += f"\nUser: {user_input}\nBot: {response}"

        # Try to extract structured info
        json_data = extract_json(response)
        if json_data:
            user_info.update({k: v for k, v in json_data.items() if v})
            print("Bot: Got it! ✅")

            # Check if we have all required fields
            required_fields = ["name", "state", "gender", "caste", "occupation", "category", "income"]
            if all(user_info.get(field) for field in required_fields):
                print("Bot: Thanks! I’ve collected everything I need for the basic recommendation.")
                print("Bot: Would you like to add any additional details (like education, land, family background, etc.)? (yes/no)")
                awaiting_additional_confirmation = True
        elif awaiting_additional_confirmation:
            if user_input.strip().lower() in ["yes", "y", "sure"]:
                additional_info_phase = True
                print("Bot: Great! Go ahead and tell me more. You can add as much as you want, and say 'no' when done.")
            else:
                break
        else:
            print(f"Bot: {response.strip()}")

    # Save to file
    with open("user_details.json", "w") as f:
        json.dump(user_info, f, indent=2)

    print("\n✅ All details saved to user_details.json")
    print("User Profile Collected:")
    print(json.dumps(user_info, indent=2))


# Function callable from orchestrator
def get_user_profile_via_chat() -> Dict[str, Optional[str]]:
    collect_user_info()
    return user_info
