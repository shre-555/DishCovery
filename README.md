# DishCovery - AI-Powered Indian Recipe Discovery

DishCovery is an advanced AI recipe recommendation agent powered by LangGraph, FastAPI, and ChromaDB. It takes ingredients you have on hand, respects your dietary restrictions, and provides structured culinary guidance using state-of-the-art LLMs (Gemma 3 and Llama 3.3).

## 📁 Data Organization

- **`IndianFoodDatasetXLS (1).csv`**: The source dataset containing raw Indian recipes, ingredients, instructions, and metadata.
- **`chroma.sqlite3/`**: The local ChromaDB vector database directory. It stores the cleaned, embedded recipes in the `indian_recipes` collection for semantic search (RAG).

## 💻 Code Organization

- **`create_vector.py`**: The data ingestion pipeline. It reads the CSV dataset, handles data cleaning (encoding fixes, translating Hindi to English, deduplication), generates text embeddings using `sentence-transformers/all-MiniLM-L6-v2` (configured for CPU), and indexes the data into ChromaDB.
- **`dishcovery.py`**: The primary FastAPI application and LangGraph agent workflow (`Dishcovery v4.2`). It contains the multi-node agent architecture (Safety Gate, Retrieval, Critique/Substitute, Generation, Fixer) and serves the API endpoints.
- **`project.py`**: A streamlined/reference version of the local FastAPI server implementation.
- **`dishcovery_notebook.ipynb`**: A Jupyter Notebook version of the pipeline for exploratory data analysis and prototyping.
- **`index.html`**: A front-end web interface served at the root URL to interactively test and use the Dishcovery API.
- **`.env`**: (User-created) Configuration file storing necessary API keys and local paths.

## 🚀 Instructions to Reproduce the Results

### 1. Prerequisites
- Python 3.12+ (tested with Python 3.12.6)
- API Keys for Gemini (Google AI Studio) and Groq.
- (Optional) Spoonacular API key for fallback functionalities.

### 2. Environment Setup
Create a new virtual environment and activate it:
```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate
```

Create a `.env` file in the root of the project with the following variables:
```env
GEMINI_API_KEY=your_gemini_api_key_here
GROQ_API_KEY=your_groq_api_key_here
SPOONACULAR_API_KEY=your_spoonacular_api_key_here
CHROMADB_PATH=./chroma.sqlite3
```

### 3. Install Dependencies
Install the required dependencies. *Note: The local setup is optimized to run embeddings on CPU, so specific CPU-compatible PyTorch versions are recommended to prevent compatibility issues:*

```bash
pip install torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cpu
pip install fastapi uvicorn chromadb sentence-transformers langgraph google-genai groq python-dotenv requests pandas
```

### 4. Create the Vector Database
Process the dataset and index the recipes into ChromaDB. This step is required before running the API to avoid "Collection does not exist" errors:
```bash
python create_vector.py
```
*Wait for this script to finish. It will clean the dataset, translate fields, generate embeddings, and populate `.chroma.sqlite3/`.*

### 5. Run the DishCovery Target API
Start the FastAPI server via the provided script:
```bash
python dishcovery.py
```

### 6. Access the Application
Once the server displays 🚀 `Starting Dishcovery API`, open your web browser and navigate to:
🔗 **http://localhost:8000**

You can now interact with the Dishcovery API via the local UI!
