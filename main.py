import os
import json
import logging
from typing import Dict, List
from profile_agent import get_user_profile_via_chat
from scheme_search_agent import search_schemes
from scheme_display_agent import SchemeDisplayAgent

# Logging setup: Console shows only ERROR, file captures all
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('chatbot.log', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
# Adjust console handler to ERROR only
for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
        handler.setLevel(logging.ERROR)

def validate_user_profile(profile: Dict) -> bool:
    """Validate that the user profile has all required fields."""
    required_fields = ['name', 'state', 'gender', 'caste', 'occupation', 'category', 'income']
    missing_fields = [field for field in required_fields if not profile.get(field)]
    if missing_fields:
        logger.error(f"Missing required fields in user profile: {missing_fields}")
        print(f"\nError: Missing required details ({', '.join(missing_fields)}). Please provide all required information.")
        return False
    return True

def validate_schemes(schemes: List[Dict]) -> List[Dict]:
    """Validate schemes to ensure they have required metadata."""
    valid_schemes = [
        scheme for scheme in schemes
        if scheme.get('metadata', {}).get('scheme_name')
    ]
    if len(valid_schemes) < len(schemes):
        logger.warning(f"Filtered out {len(schemes) - len(valid_schemes)} schemes with missing scheme_name")
    return valid_schemes

def run_conversational_chatbot():
    """Orchestrates the multi-turn conversational chatbot workflow."""
    print("🌟 Welcome to the Government Scheme Finder Chatbot! 🌟")
    logger.info("Starting the chatbot workflow")

    # Step 1: Collect user profile
    print("\nFirst, let’s get to know you a bit. I need your name, state, gender, caste, occupation, category (like SC/ST/OBC), and income.")
    try:
        user_profile = get_user_profile_via_chat()
        if not validate_user_profile(user_profile):
            logger.warning("Invalid user profile, aborting workflow")
            print("\nLet’s try again! Run the chatbot and provide all required details.")
            return
        logger.info(f"User profile collected successfully: {user_profile}")
        print("\nThanks for sharing your details! Now, let me find some schemes for you...")
    except Exception as e:
        logger.error(f"Failed to collect user profile: {str(e)}")
        print(f"\nOops, something went wrong while collecting your details: {str(e)}")
        return

    # Step 2: Search for schemes based on user profile
    json_path = "user_details.json"
    output_path = "recommended_schemes2.json"
    try:
        schemes = search_schemes(json_path, output_path)
        valid_schemes = validate_schemes(schemes)
        if not valid_schemes:
            logger.warning("No valid schemes found after validation")
            print("\nSorry, I couldn’t find any valid schemes matching your profile right now. Try again later or with different details.")
            return
        logger.info(f"Retrieved {len(valid_schemes)} valid schemes for the user")
        print(f"\nGreat news! I found {len(valid_schemes)} schemes that might work for you.")
    except FileNotFoundError:
        logger.error(f"User details file {json_path} not found")
        print(f"\nError: User details file not found. Please try again.")
        return
    except Exception as e:
        logger.error(f"Scheme search failed: {str(e)}")
        print(f"\nOops, something went wrong while searching for schemes: {str(e)}")
        return

    # Step 3: Display schemes conversationally
    print("\nNow, let’s explore the schemes I found for you!")
    try:
        agent = SchemeDisplayAgent(valid_schemes)
        agent.run()
        logger.info("Scheme display agent completed successfully")
    except Exception as e:
        logger.error(f"Scheme display agent failed: {str(e)}")
        print(f"\nOops, something went wrong while displaying the schemes: {str(e)}")

    print("\nThanks for using the Government Scheme Finder Chatbot! Have a great day! 😊")

if __name__ == "__main__":
    run_conversational_chatbot()