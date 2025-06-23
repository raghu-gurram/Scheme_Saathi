from flask import Flask, request, render_template, jsonify
import os
import json
import logging
from typing import Dict, List
from dotenv import load_dotenv
from profile_agent import get_user_profile_via_chat
from scheme_search_agent import search_schemes
from scheme_display_agent import SchemeDisplayAgent, fetch_scheme_details

app = Flask(__name__)

# Logging setup: File only, no console output
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('chatbot.log', encoding='utf-8')]
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()
if not os.getenv("PINECONE_API_KEY") or not os.getenv("GROQ_API_KEY"):
    logger.error("Missing PINECONE_API_KEY or GROQ_API_KEY in .env")
    raise ValueError("Missing API keys in .env file")

def validate_user_profile(profile: Dict) -> bool:
    required_fields = ['name', 'state', 'gender', 'caste', 'occupation', 'category', 'income']
    missing_fields = [field for field in required_fields if not profile.get(field)]
    if missing_fields:
        logger.error(f"Missing required fields in user profile: {missing_fields}")
        return False
    return True

def validate_schemes(schemes: List[Dict]) -> List[Dict]:
    valid_schemes = [
        scheme for scheme in schemes
        if scheme.get('metadata', {}).get('scheme_name')
    ]
    if len(valid_schemes) < len(schemes):
        logger.warning(f"Filtered out {len(schemes) - len(valid_schemes)} schemes with missing scheme_name")
    return valid_schemes

# Global agent state
agent_state = {
    'schemes': [],
    'agent': None,
    'selected_scheme': None
}

@app.route('/', methods=['GET'])
def index():
    logger.info("Rendering index page")
    return render_template('index.html', schemes=agent_state['schemes'])

@app.route('/submit', methods=['POST'])
def submit_profile():
    try:
        profile = request.form.to_dict()
        profile['category'] = profile.get('caste', '')  # Align category with caste
        if not validate_user_profile(profile):
            logger.error("Profile validation failed")
            return render_template('index.html', schemes=[], error="Missing required profile fields")

        with open('user_details.json', 'w', encoding='utf-8') as f:
            json.dump(profile, f, indent=4)
        logger.info("User profile saved to user_details.json")

        schemes = search_schemes('user_details.json', 'recommended_schemes2.json')
        valid_schemes = validate_schemes(schemes)
        if not valid_schemes:
            logger.warning("No valid schemes found")
            return render_template('index.html', schemes=[], error="No schemes found matching your profile")

        # Update agent state
        agent_state['schemes'] = valid_schemes
        agent_state['agent'] = SchemeDisplayAgent(valid_schemes)
        logger.info(f"Agent initialized with {len(valid_schemes)} schemes")

        return render_template('index.html', schemes=valid_schemes)
    except Exception as e:
        logger.error(f"Profile submission failed: {str(e)}")
        return render_template('index.html', schemes=[], error=f"Error: {str(e)}")

@app.route('/chat', methods=['POST'])
def chat():
    try:
        user_input = request.json['message'].strip()
        if not user_input:
            return jsonify({'response': 'Please provide a message.', 'action': 'none'})

        if not agent_state['agent']:
            logger.warning("Chat attempted before agent initialization")
            return jsonify({
                'response': 'Please submit your profile first.',
                'action': 'none'
            })

        response = agent_state['agent'].handle_input(user_input)
        action = 'none'
        details = None

        if user_input.lower() == 'show schemes':
            action = 'show_schemes'
        elif user_input.lower().startswith('show scheme'):
            try:
                scheme_num = int(user_input.split()[-1]) - 1
                if 0 <= scheme_num < len(agent_state['schemes']):
                    scheme_id = agent_state['schemes'][scheme_num]['metadata']['scheme_id']
                    details = fetch_scheme_details(scheme_id)
                    if details:
                        action = 'show_details'
                    else:
                        response = f"Sorry, no details found for scheme number {scheme_num + 1}."
            except (ValueError, IndexError):
                response = "Please specify a valid scheme number (e.g., 'show scheme 3')."
        elif user_input.lower() in ['quit', 'exit']:
            action = 'exit'
            agent_state['schemes'] = []
            agent_state['agent'] = None
            agent_state['selected_scheme'] = None

        return jsonify({
            'response': response,
            'action': action,
            'details': details
        })
    except Exception as e:
        logger.error(f"Chat processing failed: {str(e)}")
        return jsonify({'response':  f'Error: {str(e)}', 'action': 'none'}), 500

if __name__ == "__main__":
    app.run(debug=True)