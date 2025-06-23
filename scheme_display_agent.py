import json
import sqlite3
import os
import hashlib
import warnings
from typing import Dict, List
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory

# Suppress LangChain deprecation warnings
warnings.filterwarnings("ignore", category=DeprecationWarning, module="langchain")

# Load environment variables
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not GROQ_API_KEY:
    raise ValueError("GROQ_API_KEY not found in .env file")

# Initialize Groq LLM
try:
    llm = ChatGroq(
        model_name="llama3-70b-8192",
        api_key=GROQ_API_KEY,
        temperature=0.7
    )
except Exception as e:
    print(f"Failed to initialize ChatGroq: {str(e)}")
    raise

# Load recommended schemes from JSON
def load_recommended_schemes(json_path: str = "recommended_schemes.json") -> List[Dict]:
    """Load recommended schemes from JSON file and normalize structure."""
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            schemes = json.load(f)
        if not isinstance(schemes, list):
            raise ValueError("JSON file must contain a list of schemes")
        normalized_schemes = []
        for scheme in schemes:
            if 'metadata' in scheme:
                normalized_schemes.append(scheme)
            else:
                metadata = {
                    'scheme_id': scheme.get('scheme_id', ''),
                    'scheme_name': scheme.get('scheme_name', ''),
                    'brief_description': scheme.get('brief_description', ''),
                    'eligibility_criteria': scheme.get('eligibility_criteria', ''),
                    'state': scheme.get('state', ''),
                    'tags': scheme.get('tags', ''),
                    'category': scheme.get('category', '')
                }
                normalized_schemes.append({
                    'id': scheme.get('id', scheme.get('scheme_id', '')),
                    'score': scheme.get('pinecone_score', 0.0),
                    'llm_score': scheme.get('llm_score', 0),
                    'metadata': metadata
                })
        return normalized_schemes
    except FileNotFoundError:
        print(f"Error: File {json_path} not found.")
        raise
    except json.JSONDecodeError:
        print(f"Error: Invalid JSON format in {json_path}.")
        raise

# Fetch scheme details from SQLite database
def fetch_scheme_details(scheme_id: str, db_path: str = "new_schemes.db") -> Dict:
    """Fetch detailed information for a scheme from the SQLite database."""
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        query = """
        SELECT scheme_name, detailed_description, eligibility_criteria, application_process, documents_required
        FROM schemes
        WHERE scheme_id = ?
        """
        cursor.execute(query, (scheme_id,))
        result = cursor.fetchone()
        conn.close()
        if result:
            return {
                "scheme_name": result[0] or "Not available",
                "detailed_description": result[1] or "Not available",
                "eligibility_criteria": result[2] or "Not available",
                "application_process": result[3] or "Not available",
                "documents_required": result[4] or "Not available"
            }
        else:
            return None
    except sqlite3.Error as e:
        print(f"Database error: {str(e)}")
        return None

# Custom conversational agent
class SchemeDisplayAgent:
    def __init__(self, schemes: List[Dict]):
        self.schemes = [
            scheme for scheme in schemes
            if scheme.get('metadata', {}).get('scheme_name')
        ]
        for scheme in self.schemes:
            metadata = scheme['metadata']
            if not metadata.get('scheme_id'):
                scheme_name = metadata.get('scheme_name', 'unknown')
                state = metadata.get('state', 'unknown')
                combined = f"{scheme_name}_{state}".encode()
                hash_object = hashlib.md5(combined)
                hash_value = int(hash_object.hexdigest(), 16) % 10000
                metadata['scheme_id'] = f"SCH-{state[:3].upper()}-{hash_value:04d}"
        if not self.schemes:
            raise ValueError("No valid schemes provided with required metadata")
        self.selected_scheme = None
        self.scheme_details = None
        self.state = "list_schemes"
        self.memory = ConversationBufferMemory(return_messages=True)

        self.prompt_template = PromptTemplate(
            input_variables=["history", "input", "scheme_details"],
            template="""You are a helpful assistant guiding the user through government schemes in a natural, conversational way. Use the conversation history, user input, and scheme details to respond appropriately. Format the scheme details in a structured way as requested by the user.

Conversation History:
{history}

User Input: {input}

Scheme Details:
{scheme_details}

### Instructions:
- If the user is viewing the scheme list, display the schemes with their numbers, names, brief descriptions, and states. Ask, "Which scheme would you like to learn more about? Please enter its number, or you can say 'quit' to exit."
- If the user selects a scheme, present all the scheme details in a structured format:
  - **Detailed Description**: Format as bullet points. Split the description into multiple bullet points if it contains multiple sentences or logical parts.
  - **Eligibility Criteria**: Format as bullet points. Split the criteria into individual bullet points based on logical separation (e.g., different conditions or categories).
  - **Application Process**: Format as numbered steps. Split the process into individual steps based on logical separation (e.g., different actions or stages).
  - **Documents Required**: Format as bullet points. Split the documents into individual bullet points if multiple documents are listed.
  After presenting the details, ask, "Would you like to go back to the scheme list to explore another scheme, or would you like to quit?"
- If the user asks about eligibility (e.g., mentions "eligible", "eligibility", "caste", "OC", "SC", "ST", "OBC"), analyze the scheme details and provide a clear answer about their eligibility based on the criteria provided. If caste is mentioned, check if the scheme has caste-based restrictions or relaxations and respond accordingly. If the information is insufficient, suggest checking the official website for more details.
- If the user wants to go back to the scheme list (e.g., says "back" or "go back"), return to the scheme list and ask which scheme they'd like to learn about next.
- If the user says "quit" or "exit," say goodbye and end the conversation.
- If the input is unclear (e.g., "tell me"), ask for clarification (e.g., "Could you clarify what you'd like to know? I can help with eligibility, application process, or other details about the scheme.") while providing a helpful default response, such as summarizing the eligibility criteria.
- Respond in a friendly, conversational tone, and keep the dialogue natural.

Respond only with the assistant's reply, without repeating the user input or history.
"""
        )

    def display_schemes(self) -> str:
        """Generate a string representation of the scheme list."""
        if not self.schemes:
            return "I don't have any recommended schemes to show you right now. Would you like to quit?"
        
        scheme_list = "Here are the recommended schemes I found for you:\n"
        for i, scheme in enumerate(self.schemes, 1):
            scheme_list += f"{i}. {scheme['metadata']['scheme_name']}\n"
            scheme_list += f"   Brief Description: {scheme['metadata']['brief_description']}\n"
            scheme_list += f"   State: {scheme['metadata']['state']}\n\n"
        scheme_list += "Which scheme would you like to learn more about? Please enter its number, or you can say 'quit' to exit."
        return scheme_list

    def format_scheme_details(self) -> str:
        """Format the scheme details as a string to pass to the LLM."""
        details = (
            f"Scheme Name: {self.scheme_details['scheme_name']}\n"
            f"Detailed Description: {self.scheme_details['detailed_description']}\n"
            f"Eligibility Criteria: {self.scheme_details['eligibility_criteria']}\n"
            f"Application Process: {self.scheme_details['application_process']}\n"
            f"Documents Required: {self.scheme_details['documents_required']}"
        )
        return details

    def handle_input(self, user_input: str) -> str:
        """Handle the user's input and manage the conversation state."""
        user_input = user_input.strip().lower()
        history = "\n".join([str(msg) for msg in self.memory.chat_memory.messages])

        if self.state == "list_schemes":
            if user_input == "quit":
                self.state = "exit"
                return "Goodbye! If you need help later, feel free to come back."
            
            try:
                choice = int(user_input)
                if 1 <= choice <= len(self.schemes):
                    self.selected_scheme = self.schemes[choice - 1]
                    scheme_id = self.selected_scheme['metadata']['scheme_id']
                    self.scheme_details = fetch_scheme_details(scheme_id)
                    if not self.scheme_details:
                        self.selected_scheme = None
                        return (f"Sorry, I couldn't find more details for scheme ID '{scheme_id}' in the database.\n\n"
                                "Let's try another one. " + self.display_schemes())
                    self.state = "select_detail"
                    scheme_details_str = self.format_scheme_details()
                    prompt = self.prompt_template.format(history=history, input="Show the details of the selected scheme", scheme_details=scheme_details_str)
                    response = llm.invoke(prompt).content.strip()
                    self.memory.chat_memory.add_user_message(user_input)
                    self.memory.chat_memory.add_ai_message(response)
                    return response
                else:
                    return f"Hmm, please enter a number between 1 and {len(self.schemes)}.\n\n" + self.display_schemes()
            except ValueError:
                return "I didn't understand that. Please enter the number of the scheme you'd like to learn more about, or say 'quit' to exit.\n\n" + self.display_schemes()

        elif self.state == "select_detail":
            if user_input == "quit":
                self.state = "exit"
                return "Goodbye! If you need help later, feel free to come back."
            elif user_input in ["back", "go back", "return"]:
                self.selected_scheme = None
                self.scheme_details = None
                self.state = "list_schemes"
                return "Sure, let's go back to the scheme list.\n\n" + self.display_schemes()
            elif any(keyword in user_input for keyword in ["eligible", "eligibility", "caste", "oc", "sc", "st", "obc"]):
                scheme_details_str = self.format_scheme_details()
                prompt = self.prompt_template.format(history=history, input=user_input, scheme_details=scheme_details_str)
                response = llm.invoke(prompt).content.strip()
                self.memory.chat_memory.add_user_message(user_input)
                self.memory.chat_memory.add_ai_message(response)
                return response
            elif user_input == "tell me":
                scheme_details_str = self.format_scheme_details()
                prompt = self.prompt_template.format(history=history, input="Could you clarify what you'd like to know? For now, I'll summarize the eligibility criteria.", scheme_details=scheme_details_str)
                response = llm.invoke(prompt).content.strip()
                self.memory.chat_memory.add_user_message(user_input)
                self.memory.chat_memory.add_ai_message(response)
                return response
            else:
                return "I'm not sure what you'd like to do. Would you like to go back to the scheme list to explore another scheme, or would you like to quit?"

        else:
            return "Goodbye! If you need help later, feel free to come back."

    def run(self):
        """Run the conversational agent."""
        print(self.display_schemes())
        while True:
            user_input = input("\nYour response: ").strip()
            if not user_input:
                print("Please provide a response.")
                continue
            response = self.handle_input(user_input)
            print("\n" + response)
            if self.state == "exit":
                break

def scheme_display_agent_conversational():
    """Main function to run the conversational Scheme Display Agent."""
    try:
        schemes = load_recommended_schemes()
        agent = SchemeDisplayAgent(schemes)
        agent.run()
    except Exception as e:
        print(f"Failed to run scheme display agent: {str(e)}")
        raise

if __name__ == "__main__":
    scheme_display_agent_conversational()