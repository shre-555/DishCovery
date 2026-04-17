import re
import os
import time
import ftfy                        # fixes mojibake / encoding artifacts
import pandas as pd
import chromadb
from tqdm.auto import tqdm
from deep_translator import GoogleTranslator
from langdetect import detect, LangDetectException

# ── Force CPU-only execution ──────────────────────────────────────────────────
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from sentence_transformers import SentenceTransformer
 
 
# ============================================================
# 0.  INSTALL GUARD  (run once in Kaggle)
# ============================================================
# !pip install ftfy langdetect deep-translator sentence-transformers chromadb -q
 
 
# ============================================================
# 1.  LOAD  — try multiple encodings so ??? never happens
# ============================================================
 
def load_recipe_csv(path: str) -> pd.DataFrame:
    """
    Try encodings in order until the file loads cleanly.
    ftfy then repairs any remaining mojibake at the cell level.
    """
    encodings = ["utf-8", "utf-8-sig", "cp1252", "iso-8859-1", "latin-1"]
    df = None
    for enc in encodings:
        try:
            df = pd.read_csv(path, encoding=enc)
            print(f"✅ Loaded with encoding: {enc}  — {len(df)} rows")
            break
        except (UnicodeDecodeError, pd.errors.ParserError):
            continue
 
    if df is None:
        # Last resort: read bytes, replace bad chars, then parse
        with open(path, "rb") as f:
            raw = f.read().decode("utf-8", errors="replace")
        from io import StringIO
        df = pd.read_csv(StringIO(raw))
        print(f"⚠️  Loaded with errors='replace'  — {len(df)} rows")
 
    # Apply ftfy on every string column to fix remaining mojibake
    for col in df.select_dtypes(include="object").columns:
        df[col] = df[col].apply(lambda x: ftfy.fix_text(str(x)) if pd.notna(x) else x)
 
    return df
 
 
# ============================================================
# 2.  DETECT & TRANSLATE  (Hindi → English)
# ============================================================
 
# Cache so we don't re-translate the same string in different rows
_translation_cache: dict[str, str] = {}
 
def _is_hindi(text: str) -> bool:
    """
    Two-stage check:
      1. Fast regex — Devanagari Unicode block (U+0900–U+097F)
      2. langdetect fallback for romanised Hindi / mixed scripts
    """
    if not text or not isinstance(text, str):
        return False
    # Any Devanagari character → definitely Hindi/Sanskrit
    if re.search(r'[\u0900-\u097F]', text):
        return True
    # Unreadable ??? artifacts — these are broken Devanagari
    if text.count('?') / max(len(text), 1) > 0.3:
        return True
    return False
 
 
def translate_to_english(text: str, retries: int = 3) -> str:
    """
    Translate a single string to English using GoogleTranslator.
    Returns original text if translation fails or isn't needed.
    """
    if not text or not isinstance(text, str):
        return text
    text = text.strip()
    if not _is_hindi(text):
        return text
    if text in _translation_cache:
        return _translation_cache[text]
 
    for attempt in range(retries):
        try:
            translated = GoogleTranslator(source="auto", target="en").translate(text)
            result = translated if translated else text
            _translation_cache[text] = result
            return result
        except Exception as e:
            if attempt < retries - 1:
                time.sleep(2 ** attempt)   # exponential back-off: 1s, 2s, 4s
            else:
                print(f"    ⚠️  Translation failed after {retries} attempts: {e}")
                _translation_cache[text] = text
                return text
 
 
def translate_column(series: pd.Series, col_name: str) -> pd.Series:
    """Translate an entire Series with a progress bar."""
    print(f"  Translating column: {col_name}")
    results = []
    hindi_count = 0
    for val in tqdm(series, desc=f"  {col_name}", leave=False):
        if _is_hindi(str(val)):
            hindi_count += 1
            results.append(translate_to_english(str(val)))
        else:
            results.append(val)
    print(f"    → {hindi_count} Hindi values translated in '{col_name}'")
    return pd.Series(results, index=series.index)
 
 
# ============================================================
# 3.  FIELD-LEVEL CLEANING
# ============================================================
 
# --- Ingredients ---
 
def clean_ingredients(raw: str) -> str:
    """
    Input:  "1 cup rice,2 tsp oil,salt - to taste"
    Output: "1 cup rice, 2 tsp oil, salt"
    """
    if not raw or not isinstance(raw, str):
        return ""
    raw = raw.strip()
 
    # Split on commas
    parts = [p.strip() for p in raw.split(",") if p.strip()]
    cleaned = []
    for part in parts:
        # Remove "- to taste", "- as needed", "- for garnish" suffixes
        part = re.sub(r'\s*-\s*(to taste|as needed|as required|for garnish|'
                      r'for seasoning|adjust to taste)', '', part, flags=re.I)
        # Normalise whitespace
        part = re.sub(r'\s+', ' ', part).strip()
        # Drop empty or single-char fragments
        if len(part) > 1:
            cleaned.append(part)
 
    return ", ".join(cleaned)
 
 
# --- Instructions ---
 
def clean_instructions(raw: str) -> str:
    """
    Strips HTML tags, excessive whitespace, and truncates very long
    instructions to 1000 chars (embedding models have token limits).
    """
    if not raw or not isinstance(raw, str):
        return ""
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', raw)
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Truncate — keep first 1000 chars (roughly 200 tokens)
    return text[:1000]
 
 
# --- Recipe name ---
 
def clean_recipe_name(raw: str) -> str:
    if not raw or not isinstance(raw, str):
        return "Unknown Recipe"
    name = raw.strip()
    # Remove leading index numbers like "13994  Tadkewali..."
    name = re.sub(r'^\d+\s+', '', name)
    # Title-case
    return name.strip().title()
 
 
# --- Numeric fields ---
 
def clean_numeric(val, default: int = 0) -> int:
    try:
        return int(float(str(val).strip()))
    except (ValueError, TypeError):
        return default
 
 
# --- Diet / Cuisine / Course normalisation ---
 
DIET_ALIASES = {
    "high protein vegetarian": "High Protein Vegetarian",
    "vegetarian":              "Vegetarian",
    "vegan":                   "Vegan",
    "non vegetarian":          "Non Vegetarian",
    "eggetarian":              "Eggetarian",
    "diabetic friendly":       "Diabetic Friendly",
    "gluten free":             "Gluten Free",
    "no onion no garlic":      "No Onion No Garlic",
    "sugar free diet":         "Sugar Free",
    "high fiber vegetarian":   "High Fiber Vegetarian",
    "":                        "Unspecified",
}
 
def normalise_diet(raw: str) -> str:
    key = str(raw).strip().lower()
    return DIET_ALIASES.get(key, str(raw).strip().title())
 
 
# ============================================================
# 4.  MASTER CLEANING FUNCTION
# ============================================================
 
REQUIRED_COLS = [
    "TranslatedRecipeName",
    "TranslatedIngredients",
    "TranslatedInstructions",
    "Cuisine",
    "Course",
    "Diet",
    "Servings",
    "PrepTimeInMins",
    "CookTimeInMins",
    "TotalTimeInMins",
]
 
def clean_recipe_df(raw_df: pd.DataFrame) -> pd.DataFrame:
    """
    Full cleaning pipeline:
      1. Column standardisation
      2. Drop rows with no usable content
      3. Translate Hindi fields
      4. Field-level cleaning
      5. Deduplication
    """
    print("\n" + "="*60)
    print("STEP 1/5 — Column standardisation")
    print("="*60)
 
    df = raw_df.copy()
 
    # Rename Srno → index if present
    if "Srno" in df.columns:
        df = df.rename(columns={"Srno": "original_index"})
 
    # Ensure all required columns exist
    for col in REQUIRED_COLS:
        if col not in df.columns:
            print(f"  ⚠️  Missing column '{col}' — filling with empty string")
            df[col] = ""
 
    # Fill NaN
    df = df.fillna("")
 
    print(f"  Rows loaded: {len(df)}")
 
    # ── Step 2: Drop unusable rows ──────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 2/5 — Dropping unusable rows")
    print("="*60)
 
    before = len(df)
 
    # Must have a recipe name
    df = df[df["TranslatedRecipeName"].str.strip().str.len() > 2]
 
    # Must have at least some ingredients
    df = df[df["TranslatedIngredients"].str.strip().str.len() > 5]
 
    # Rows where >50% of ingredient text is '?' are broken-encoding rows
    def is_mostly_garbage(text: str) -> bool:
        if not text:
            return False
        q_ratio = text.count('?') / max(len(text), 1)
        return q_ratio > 0.4
 
    # Flag broken rows — we'll translate these rather than drop them
    df["_has_hindi"] = (
        df["TranslatedIngredients"].apply(_is_hindi) |
        df["TranslatedRecipeName"].apply(_is_hindi) |
        df["TranslatedInstructions"].apply(_is_hindi)
    )
 
    garbage_mask = df["TranslatedIngredients"].apply(is_mostly_garbage)
    df_garbage = df[garbage_mask & ~df["_has_hindi"]]   # broken AND not translatable
    df = df[~garbage_mask | df["_has_hindi"]]           # keep translatable ones
 
    print(f"  Rows dropped (unrecoverable garbage): {len(df_garbage)}")
    print(f"  Rows with Hindi content to translate:  {df['_has_hindi'].sum()}")
    print(f"  Rows remaining: {len(df)}  (dropped {before - len(df)} total)")
 
    # ── Step 3: Translate Hindi ─────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 3/5 — Hindi → English translation")
    print("="*60)
 
    translate_cols = [
        "TranslatedRecipeName",
        "TranslatedIngredients",
        "TranslatedInstructions",
    ]
 
    for col in translate_cols:
        df[col] = translate_column(df[col], col)
 
    df = df.drop(columns=["_has_hindi"])
 
    # ── Step 4: Field-level cleaning ────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 4/5 — Field-level cleaning")
    print("="*60)
 
    df["TranslatedRecipeName"]    = df["TranslatedRecipeName"].apply(clean_recipe_name)
    df["TranslatedIngredients"]   = df["TranslatedIngredients"].apply(clean_ingredients)
    df["TranslatedInstructions"]  = df["TranslatedInstructions"].apply(clean_instructions)
    df["Diet"]                    = df["Diet"].apply(normalise_diet)
    df["Cuisine"]                 = df["Cuisine"].str.strip().str.title()
    df["Course"]                  = df["Course"].str.strip().str.title()
    df["Servings"]                = df["Servings"].apply(lambda x: clean_numeric(x, default=4))
    df["PrepTimeInMins"]          = df["PrepTimeInMins"].apply(lambda x: clean_numeric(x, default=0))
    df["CookTimeInMins"]          = df["CookTimeInMins"].apply(lambda x: clean_numeric(x, default=0))
    df["TotalTimeInMins"]         = df["TotalTimeInMins"].apply(lambda x: clean_numeric(x, default=0))
 
    # Recompute TotalTime if it's 0 but Prep+Cook are known
    mask = (df["TotalTimeInMins"] == 0) & (df["PrepTimeInMins"] + df["CookTimeInMins"] > 0)
    df.loc[mask, "TotalTimeInMins"] = df.loc[mask, "PrepTimeInMins"] + df.loc[mask, "CookTimeInMins"]
 
    # Drop rows where ingredients are now empty after cleaning
    df = df[df["TranslatedIngredients"].str.strip().str.len() > 5]
 
    print(f"  Rows after field cleaning: {len(df)}")
 
    # ── Step 5: Deduplication ───────────────────────────────────────────
    print("\n" + "="*60)
    print("STEP 5/5 — Deduplication")
    print("="*60)
 
    before_dedup = len(df)
    # Exact name + ingredient duplicates
    df = df.drop_duplicates(
        subset=["TranslatedRecipeName", "TranslatedIngredients"],
        keep="first"
    )
    df = df.reset_index(drop=True)
    print(f"  Duplicates removed: {before_dedup - len(df)}")
    print(f"  ✅ Final clean dataset: {len(df)} recipes")
 
    return df
 
 
# ============================================================
# 5.  EMBEDDING TEXT CONSTRUCTION
# ============================================================
 
def build_embedding_text(row: pd.Series) -> str:
    """
    Constructs a rich, structured string for embedding.
    Weights the recipe name and ingredients more heavily by repeating them,
    since these are the most semantically important fields for retrieval.
 
    Example output:
        "Tadka Dal. Ingredients: red lentils, cumin, turmeric, garlic, ghee,
         tomato, onion. Diet: High Protein Vegetarian. Cuisine: North Indian.
         Course: Lunch. Ready in 30 minutes. Serves 4."
    """
    name        = str(row.get("TranslatedRecipeName", "")).strip()
    ingredients = str(row.get("TranslatedIngredients", "")).strip()
    diet        = str(row.get("Diet", "")).strip()
    cuisine     = str(row.get("Cuisine", "")).strip()
    course      = str(row.get("Course", "")).strip()
    total_time  = clean_numeric(row.get("TotalTimeInMins", 0))
    servings    = clean_numeric(row.get("Servings", 4))
 
    # Shorten ingredients list for embedding — first 15 items only
    # (avoids token overflow; the full list is stored in metadata)
    short_ingredients = ", ".join(
        [i.strip() for i in ingredients.split(",")][:15]
    )
 
    parts = [
        f"{name}.",                                    # name first (most important)
        f"Ingredients: {short_ingredients}.",
    ]
    if diet:        parts.append(f"Diet: {diet}.")
    if cuisine:     parts.append(f"Cuisine: {cuisine}.")
    if course:      parts.append(f"Course: {course}.")
    if total_time:  parts.append(f"Ready in {total_time} minutes.")
    if servings:    parts.append(f"Serves {servings}.")
 
    return " ".join(parts)
 
 
# ============================================================
# 6.  CHROMADB SETUP
# ============================================================
 
def setup_chromadb(
    recipe_df: pd.DataFrame,
    chroma_path: str = "./chroma_db",
    collection_name: str = "indian_recipes",
    model_name: str = "all-MiniLM-L6-v2",
    batch_size: int = 2000,
) -> tuple[chromadb.PersistentClient, chromadb.Collection]:
    """
    Embeds the cleaned recipe DataFrame and indexes it into ChromaDB.
    Safe to re-run — always recreates the collection from scratch.
    """
    print("\n" + "="*60)
    print("CHROMADB INDEXING")
    print("="*60)
 
    # 1. Build embedding texts
    print("  Building embedding texts...")
    df = recipe_df.copy()
    df["embedding_text"] = df.apply(build_embedding_text, axis=1)
 
    # 2. Encode
    print(f"  Loading encoder: {model_name}")
    encoder = SentenceTransformer(model_name)
    print(f"  Encoding {len(df)} recipes...")
    embeddings = encoder.encode(
        df["embedding_text"].tolist(),
        batch_size=64,
        show_progress_bar=True,
        normalize_embeddings=True,   # unit vectors → cosine = dot product (faster)
    )
 
    # 3. Init ChromaDB
    chroma_client = chromadb.PersistentClient(path=chroma_path)
    try:
        chroma_client.delete_collection(collection_name)
        print(f"  🔄 Refreshed collection: {collection_name}")
    except Exception:
        print(f"  ✨ Creating new collection: {collection_name}")
 
    collection = chroma_client.create_collection(
        name=collection_name,
        metadata={"hnsw:space": "cosine"},
    )
 
    # 4. Prepare metadata — ChromaDB only accepts str/int/float/bool
    #    We keep only the fields Dishcovery actually queries on.
    METADATA_COLS = [
        "TranslatedRecipeName",
        "TranslatedIngredients",
        "Diet",
        "Cuisine",
        "Course",
        "Servings",
        "TotalTimeInMins",
        "PrepTimeInMins",
        "CookTimeInMins",
    ]
    meta_df = df[METADATA_COLS].copy()
 
    # Ensure no NaN/None in metadata (ChromaDB rejects them)
    for col in meta_df.select_dtypes(include="object").columns:
        meta_df[col] = meta_df[col].fillna("").astype(str)
    for col in meta_df.select_dtypes(include=["int64", "float64"]).columns:
        meta_df[col] = meta_df[col].fillna(0).astype(int)
 
    metadatas    = meta_df.to_dict("records")
    all_docs     = df["embedding_text"].tolist()
    all_embeddings = embeddings.tolist()
    all_ids      = [str(i) for i in range(len(df))]
 
    # 5. Batch insert
    total = len(all_ids)
    for i in range(0, total, batch_size):
        end = min(i + batch_size, total)
        collection.add(
            embeddings=all_embeddings[i:end],
            documents=all_docs[i:end],
            metadatas=metadatas[i:end],
            ids=all_ids[i:end],
        )
        print(f"  Indexed {end}/{total} recipes")
 
    print(f"\n✅ ChromaDB ready — {collection.count()} recipes indexed")
    return chroma_client, collection
 
 
# ============================================================
# 7.  RAG RETRIEVAL  (drop-in replacement for Dishcovery)
# ============================================================
 
def retrieve_similar_recipes(
    collection: chromadb.Collection,
    encoder: SentenceTransformer,
    query: str,
    dietary_restrictions: list[str] | None = None,
    n_results: int = 3,
) -> list[str]:
    """
    Retrieve top-N relevant recipes.
 
    Optionally filter by dietary tag so retrieved examples are always
    compliant — this makes the Sous Chef prompt more useful.
 
    Returns a list of embedding_text strings (what Gemini sees).
    """
    query_embedding = encoder.encode(
        [query], normalize_embeddings=True
    ).tolist()
 
    # Build a where-filter if dietary restrictions are provided
    where = None
    if dietary_restrictions:
        # Map our restriction tags to the Diet field values in the dataset
        diet_map = {
            "vegan":                  "Vegan",
            "vegetarian":             "Vegetarian",
            "gluten_free":            "Gluten Free",
            "no_onion_garlic":        "No Onion No Garlic",
            "high_protein":           "High Protein Vegetarian",
            "diabetic_friendly":      "Diabetic Friendly",
        }
        matched_diets = [
            diet_map[r] for r in dietary_restrictions if r in diet_map
        ]
        if matched_diets:
            # ChromaDB $in operator — filter to compliant diet rows only
            where = {"Diet": {"$in": matched_diets}}
 
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"],
    )
 
    docs      = results.get("documents", [[]])[0]
    distances = results.get("distances", [[]])[0]
    metas     = results.get("metadatas", [[]])[0]
 
    # Log retrieval quality
    print(f"\n  RAG retrieved {len(docs)} results for: '{query[:60]}'")
    for i, (doc, dist, meta) in enumerate(zip(docs, distances, metas)):
        print(f"    [{i+1}] {meta.get('TranslatedRecipeName', '?')} "
              f"| diet={meta.get('Diet','?')} "
              f"| similarity={1-dist:.3f}")
 
    return docs
 
 
# ============================================================
# 8.  DIAGNOSTIC  — run this to verify RAG is working
# ============================================================
 
def run_rag_diagnostic(collection: chromadb.Collection, encoder: SentenceTransformer):
    """
    Quick sanity checks to confirm retrieval is non-trivial.
    Prints similarity scores — you want these above 0.3 for good retrieval.
    """
    print("\n" + "="*60)
    print("RAG DIAGNOSTIC")
    print("="*60)
    print(f"  Total recipes indexed: {collection.count()}")
 
    test_queries = [
        ("vegan curry coconut milk spinach",         ["vegan"]),
        ("quick vegetarian dal lentils",             ["vegetarian"]),
        ("gluten free Indian flatbread rice flour",  ["gluten_free"]),
        ("butter chicken tomato cream",              None),
        ("paneer palak spinach",                     ["vegetarian"]),
    ]
 
    for query, restrictions in test_queries:
        print(f"\n  Query: '{query}' | restrictions: {restrictions}")
        retrieve_similar_recipes(
            collection, encoder, query,
            dietary_restrictions=restrictions,
            n_results=3,
        )
 
 
# ============================================================
# 9.  EXECUTION  (Kaggle cells)
# ============================================================
 
if __name__ == "__main__":
 
    # ── Cell 1: Load & Clean ─────────────────────────────────────────────
    # Local CSV file
    DATASET_PATH = "IndianFoodDatasetXLS (1).csv"
 
    if not os.path.exists(DATASET_PATH):
        print(f"❌ Dataset not found: {DATASET_PATH}")
        print(f"   Please ensure '{DATASET_PATH}' exists in the current directory.")
        exit(1)
 
    print(f"\n📂 Loading dataset from: {DATASET_PATH}\n")
    raw_df    = load_recipe_csv(DATASET_PATH)
    clean_df  = clean_recipe_df(raw_df)
 
    # Optional: inspect a sample of previously-broken rows
    print("\nSample of cleaned rows (previously Hindi):")
    sample = clean_df[
        clean_df["TranslatedIngredients"].str.contains("lentil|dal|masoor", case=False, na=False)
    ].head(3)[["TranslatedRecipeName", "TranslatedIngredients", "Diet"]]
    print(sample.to_string())
 
    # ── Cell 2: Embed & Index ────────────────────────────────────────────
    print("\n🚀 Initializing SentenceTransformer (CPU-only mode)...")
    encoder = SentenceTransformer("all-MiniLM-L6-v2", device="cpu")
    chroma_client, collection = setup_chromadb(
        clean_df,
        chroma_path="./chroma.sqlite3",
        collection_name="indian_recipes",
        model_name="all-MiniLM-L6-v2",
    )
 
    # ── Cell 3: Diagnostic ───────────────────────────────────────────────
    run_rag_diagnostic(collection, encoder)