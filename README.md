# Scheme Saathi

Scheme Saathi is an AI-powered chatbot and web application that helps users discover and get recommendations for government schemes based on their personal profile. It leverages LLMs, vector search, and a user-friendly interface to match users with the most relevant schemes.

## Features
- **Conversational Chatbot**: Collects user details in a friendly, step-by-step manner.
- **Personalized Recommendations**: Suggests government schemes tailored to user profile (state, gender, caste, occupation, category, income, etc.).
- **Scheme Search**: Uses semantic search and LLMs to find the best-matching schemes from a dataset.
- **Web Interface**: Simple and modern UI for easy interaction.
- **Database Integration**: Uses SQLite for storing and querying scheme data.

## Project Structure
```
Scheme_Saathi/
├── app.py                  # Flask web app
├── main.py                 # CLI chatbot entry point
├── databse_setup.py        # Script to set up SQLite DB from CSV
├── dataset.csv             # Source data for schemes
├── new_schemes.db          # SQLite database
├── profile_agent.py        # Handles user profile collection
├── scheme_search_agent.py  # Handles scheme search logic
├── scheme_display_agent.py # Handles scheme display and details
├── query_agent.py          # (Optional) Query logic
├── requirements.txt        # Python dependencies
├── templates/
│   └── index.html          # Web UI template
├── user_details.json       # Stores user profiles
├── user_info.json          # (Optional) User info
├── recommended_schemes.json# Stores recommendations
```

## Setup Instructions
1. **Clone the repository**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Set up environment variables**:
   - Create a `.env` file with your API keys (PINECONE_API_KEY, GROQ_API_KEY, etc.)
4. **Set up the database**:
   ```bash
   python databse_setup.py
   ```
5. **Run the web app**:
   ```bash
   python app.py
   ```
   The app will be available at `http://localhost:5000`.

6. **(Optional) Run the CLI chatbot**:
   ```bash
   python main.py
   ```

## Requirements
- Python 3.8+
- See `requirements.txt` for all dependencies

## Notes
- Make sure to provide valid API keys in your `.env` file for all LLM and vector search services.
- The dataset (`dataset.csv`) should be formatted as expected by `databse_setup.py`.

## License
MIT License
