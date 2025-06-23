import os
import json
import logging
import pandas as pd
from typing import List, Dict, Any
from dotenv import load_dotenv
from pinecone import Pinecone
from sentence_transformers import SentenceTransformer
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
import re
import sys
import hashlib

# Logging setup: Console shows only ERROR, file captures all
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('search_errors.log', encoding='utf-8'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)
# Adjust console handler to ERROR only
for handler in logging.getLogger().handlers:
    if isinstance(handler, logging.StreamHandler) and not isinstance(handler, logging.FileHandler):
        handler.setLevel(logging.ERROR)

# Load environment variables
load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
if not PINECONE_API_KEY or not GROQ_API_KEY:
    logger.error("PINECONE_API_KEY or GROQ_API_KEY not found in .env file")
    raise ValueError("PINECONE_API_KEY or GROQ_API_KEY not found in .env file")

# Initialize Pinecone
try:
    pc = Pinecone(api_key=PINECONE_API_KEY)
    index_name = "scheme-data" # pinecone db index name
    index = pc.Index(
        name=index_name,
        host="Pinecone DB hosting link"
    )
except Exception as e:
    logger.error(f"Failed to initialize Pinecone: {str(e)}")
    raise

# Initialize embedding model
try:
    model = SentenceTransformer('BAAI/bge-large-en-v1.5')
except Exception as e:
    logger.error(f"Failed to initialize SentenceTransformer: {str(e)}")
    raise

# Initialize Groq LLM
try:
    llm = ChatGroq(
        model_name="llama3-70b-8192",
        api_key=GROQ_API_KEY,
        temperature=0.7
    )
except Exception as e:
    logger.error(f"Failed to initialize ChatGroq: {str(e)}")
    raise

def sanitize_input(text: str) -> str:
    """Sanitize input to prevent injection attacks."""
    if not isinstance(text, str):
        return ""
    return re.sub(r'[^\w\s.,-]', '', text)

def load_user_details(json_path: str) -> Dict[str, Any]:
    """Load and validate user details from JSON file."""
    logger.info(f"Loading user details from {json_path}")
    try:
        with open(json_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        required_fields = ['state', 'gender', 'occupation', 'income']
        missing_fields = [field for field in required_fields if field not in data]
        if missing_fields:
            logger.error(f"Missing required fields in user_details.json: {missing_fields}")
            raise ValueError(f"Missing required fields: {missing_fields}")
        sanitized_data = {k: sanitize_input(str(v)) for k, v in data.items()}
        sanitized_data['caste'] = sanitized_data.get('caste', sanitized_data.get('category', ''))
        return sanitized_data
    except FileNotFoundError:
        logger.error(f"User details file not found: {json_path}")
        raise
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON format in {json_path}")
        raise

def generate_query(user_details: Dict[str, Any]) -> str:
    """Generate a natural language query using Groq LLM."""
    logger.info("Generating query with Groq LLM")
    prompt_template = PromptTemplate(
        input_variables=["state", "gender", "caste", "occupation", "income", "additional_details"],
        template="Generate a concise search query for government schemes, styled like: 'Scheme Name. Tags: [tags]. State: [state]. Eligibility: [criteria].' "
                 "Use the following user details: State: {state}, Gender: {gender}, Caste: {caste}, "
                 "Occupation: {occupation}, Income: {income}, Additional Details: {additional_details}. "
                 "If caste is empty, include schemes for all castes, including SC."
    )
    prompt = prompt_template.format(
        state=user_details.get('state', ''),
        gender=user_details.get('gender', ''),
        caste=user_details.get('caste', ''),
        occupation=user_details.get('occupation', ''),
        income=user_details.get('income', ''),
        additional_details=user_details.get('additional_details', '')
    )
    try:
        response = llm.invoke(prompt)
        query = response.content.strip()
        logger.info(f"Generated query: {query}")
        return query
    except Exception as e:
        logger.error(f"Failed to generate query with Groq: {str(e)}")
        fields = [
            str(user_details.get('state', '')),
            str(user_details.get('gender', '')),
            str(user_details.get('caste', 'all castes')),
            str(user_details.get('occupation', '')),
            str(user_details.get('income', '')),
            str(user_details.get('additional_details', ''))
        ]
        return ". ".join(filter(None, fields))

def search_pinecone(query: str, top_k: int = 20) -> List[Dict[str, Any]]:
    """Search Pinecone index for relevant schemes."""
    logger.info(f"Searching Pinecone with query: {query}")
    try:
        embedding = model.encode(query, normalize_embeddings=True).tolist()
        results = index.query(
            vector=embedding,
            top_k=top_k,
            include_metadata=True
        )
        schemes = []
        for match in results['matches']:
            metadata = match.get('metadata', {})
            if not metadata.get('scheme_name'):
                logger.warning(f"Skipping scheme with missing scheme_name: {match}")
                continue
            scheme_id = None
            possible_keys = ['scheme_id', 'schemeId', 'SchemeID', 'id', 'scheme_id ']
            for key in possible_keys:
                if key in metadata:
                    scheme_id = metadata[key]
                    logger.info(f"Found scheme_id with key '{key}' for scheme '{metadata.get('scheme_name')}': {scheme_id}")
                    break
            if scheme_id is None:
                scheme_name = metadata.get('scheme_name', 'unknown')
                state = metadata.get('state', 'unknown')
                combined = f"{scheme_name}_{state}".encode()
                hash_object = hashlib.md5(combined)
                hash_value = int(hash_object.hexdigest(), 16) % 10000
                scheme_id = f"SCH-{state[:3].upper()}-{hash_value:04d}"
                logger.warning(f"No scheme_id found in metadata for scheme '{scheme_name}' (state: {state}). Generated placeholder: {scheme_id}")
            schemes.append({
                'id': match['id'],
                'score': match['score'],
                'metadata': {
                    'scheme_id': str(scheme_id),
                    'scheme_name': metadata.get('scheme_name', ''),
                    'brief_description': metadata.get('brief_description', ''),
                    'eligibility_criteria': metadata.get('eligibility_criteria', ''),
                    'state': metadata.get('state', ''),
                    'tags': metadata.get('tags', ''),
                    'category': metadata.get('category', '')
                }
            })
        logger.info(f"Retrieved {len(schemes)} schemes from Pinecone")
        return schemes
    except Exception as e:
        logger.error(f"Pinecone query failed: {str(e)}")
        return []

def rerank_with_llm(
    schemes: List[Dict[str, Any]],
    user_details: Dict[str, Any],
    top_k: int = 20,
    batch_size: int = 5
) -> List[Dict[str, Any]]:
    """Re-rank schemes using Groq LLM in batches based on user profile similarity."""
    logger.info(f"Re-ranking {len(schemes)} schemes with Groq LLM in batches of {batch_size}")
    prompt_template = PromptTemplate(
        input_variables=["user_details", "schemes"],
        template="""Score the relevance of each government scheme for the user (0-100). 
                 User profile: {user_details}. 
                 Schemes: {schemes}. 
                 Consider state, gender, caste (if specified), occupation, income, and additional details. 
                 If caste is not specified, assume all castes are eligible. 
                 Return a JSON array of integer scores (0-100) in the same order as the schemes, e.g., [95, 80, 70, 60, 12]."""
    )
    ranked_schemes = []
    user_details_str = json.dumps(user_details, indent=2, ensure_ascii=False)

    for i in range(0, len(schemes), batch_size):
        batch = schemes[i:i + batch_size]
        batch_schemes = [json.dumps(scheme['metadata'], indent=2, ensure_ascii=False) for scheme in batch]
        schemes_str = "\n\n".join([f"Scheme {j+1}: {s}" for j, s in enumerate(batch_schemes)])
        prompt = prompt_template.format(
            user_details=user_details_str,
            schemes=schemes_str
        )
        try:
            response = llm.invoke(prompt)
            response_text = response.content.strip()
            try:
                scores = json.loads(response_text)
                if not isinstance(scores, list) or len(scores) != len(batch):
                    raise ValueError(f"Invalid scores format or count: {scores}")
            except json.JSONDecodeError:
                logger.warning(f"Invalid JSON response from LLM")
                scores = [0] * len(batch)  # Fallback: assign 0 scores
        except Exception as e:
            logger.warning(f"Failed to re-rank batch {i+1}-{i+len(batch)}: {str(e)}")
            scores = [0] * len(batch)  # Fallback: assign 0 scores
        for scheme, score in zip(batch, scores):
            ranked_schemes.append({
                'id': scheme['id'],
                'llm_score': min(max(int(score), 0), 100),
                'pinecone_score': scheme['score'],
                'metadata': scheme['metadata']
            })
        logger.info(f"Successfully re-ranked batch {i+1}-{i+len(batch)}")

    ranked_schemes.sort(key=lambda x: (x['llm_score'], x['pinecone_score']), reverse=True)
    logger.info(f"Re-ranked {len(ranked_schemes)} schemes, selecting top {top_k}")
    return ranked_schemes[:top_k]

def save_recommended_schemes(schemes: List[Dict[str, Any]], output_path: str):
    """Save recommended schemes to JSON file."""
    logger.info(f"Saving {len(schemes)} schemes to {output_path}")
    output = [
        {
            'scheme_id': scheme['metadata']['scheme_id'],
            'scheme_name': scheme['metadata']['scheme_name'],
            'brief_description': scheme['metadata']['brief_description'],
            'eligibility_criteria': scheme['metadata']['eligibility_criteria'],
            'state': scheme['metadata']['state'],
            'tags': scheme['metadata']['tags'],
            'category': scheme['metadata']['category'],
            'llm_score': int(scheme['llm_score']),
            'pinecone_score': float(scheme['pinecone_score'])
        }
        for scheme in schemes
    ]
    try:
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(output, f, indent=4, ensure_ascii=False)
        logger.info(f"Successfully saved schemes to {output_path}")
    except Exception as e:
        logger.error(f"Failed to save schemes to {output_path}: {str(e)}")

def search_schemes(json_path: str, output_path: str = "recommended_schemes2.json") -> List[Dict[str, Any]]:
    """Main function to execute search for schemes based on user details."""
    user_details = load_user_details(json_path)
    query = generate_query(user_details)
    schemes = search_pinecone(query, top_k=20)
    if not schemes:
        logger.warning("No schemes found in Pinecone search")
        save_recommended_schemes([], output_path)
        return []
    user_state = str(user_details.get('state', '')).lower()
    filtered_schemes = [
        scheme for scheme in schemes
        if user_state == str(scheme['metadata'].get('state', '')).lower() or
           any(term in str(scheme['metadata'].get('state', '')).lower() for term in ['all india', 'central', 'nationwide'])
    ]
    logger.info(f"Filtered to {len(filtered_schemes)} schemes with matching state or Central")
    if not filtered_schemes:
        logger.warning("No schemes found after state filtering")
        save_recommended_schemes([], output_path)
        return []
    ranked_schemes = rerank_with_llm(filtered_schemes, user_details, top_k=20)
    save_recommended_schemes(ranked_schemes, output_path)
    return ranked_schemes

def main():
    """Test the scheme search agent with filtered query."""
    json_path = "user_details.json"
    output_path = "recommended_schemes2.json"
    try:
        schemes = search_schemes(json_path, output_path)
        if not schemes:
            print("No schemes found for your profile.")
            return
        print(f"Successfully fetched {len(schemes)} recommended schemes:")
        for i, scheme in enumerate(schemes, 1):
            metadata = scheme['metadata']
            print(f"\n{i}. {metadata['scheme_name']}")
            print(f"   Description: {metadata['brief_description']}")
            print(f"   LLM Score: {scheme['llm_score']}, Pinecone Score: {scheme['pinecone_score']}")
    except FileNotFoundError:
        print(f"Error: User details file {json_path} not found.")
        logger.error(f"User details file {json_path} not found.")
    except Exception as e:
        print(f"Error: {str(e)}")
        logger.error(f"Main function failed: {str(e)}")

if __name__ == "__main__":
    main()