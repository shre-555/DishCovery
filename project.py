import os
import json
import logging
import time
from typing import TypedDict, Optional, List, Dict, Any
import requests
from datetime import datetime

# LangGraph & State Management
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# LLM Providers
from google import genai
from groq import Groq

# Vector DB
import chromadb

# FastAPI
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ================================================================================
# CONFIGURATION & CONSTANTS
# ================================================================================

INDIAN_TO_ENGLISH_MAP = {
    "hing": "asafoetida",
    "jeera": "cumin",
    "zeera": "cumin",
    "haldi": "turmeric",
    "methi": "fenugreek",
    "kasuri methi": "dried fenugreek leaves",
    "ajwain": "carom seeds",
    "dalchini": "cinnamon",
    "tej patta": "bay leaf",
    "laung": "clove",
    "elaichi": "cardamom",
    "amchoor": "mango powder",
    "amchur": "mango powder",
    "kali mirch": "black pepper",
    "saunf": "fennel",
    "kalonji": "nigella seeds",
    "makhana": "lotus seeds",
    "pudina": "mint",
    "dhaniya": "coriander",
}

# All supported dietary tags — the only valid values Gemini may emit
# in an ingredient's "violates" list.
ALL_DIETARY_TAGS = [
    "vegan", "vegetarian", "gluten_free", "dairy_free", "nut_free",
    "egg_free", "soy_free", "keto", "paleo", "low_carb",
    "no_onion_garlic", "halal", "kosher", "jain",
]

MAX_SUBSTITUTION_ATTEMPTS = 3


# ================================================================================
# STATE SCHEMA
# ================================================================================

class AgentState(TypedDict):
    user_profile: Dict[str, Any]
    inventory: List[str]
    current_recipe_json: Optional[Dict[str, Any]]
    current_step_index: int
    error_log: List[str]
    substitutions_needed: List[Dict[str, str]]
    user_query: str
    fix_request: Optional[str]
    messages: List[str]
    next_action: str
    substitution_attempts: int


# ================================================================================
# UTILITY FUNCTIONS
# ================================================================================

def normalize_ingredients(ingredients: List[str]) -> List[str]:
    normalized = []
    for ingredient in ingredients:
        key = ingredient.lower().strip()
        normalized.append(INDIAN_TO_ENGLISH_MAP.get(key, key))
    return normalized


def get_spoonacular_substitute(ingredient: str, api_key: str) -> Optional[Dict[str, Any]]:
    """Spoonacular API lookup. Strips noise adjectives before querying."""
    try:
        noise_words = {
            "fresh", "dried", "raw", "organic", "refined", "unrefined",
            "large", "small", "medium", "whole", "chopped", "sliced",
            "minced", "ground", "frozen", "canned", "unsweetened",
        }
        clean_name = " ".join(
            w for w in ingredient.lower().split() if w not in noise_words
        ).strip() or ingredient

        url = "https://api.spoonacular.com/food/ingredients/substitutes"
        params = {"ingredientName": clean_name, "apiKey": api_key}
        response = requests.get(url, params=params, timeout=10)
        logger.info(
            f"Spoonacular [{response.status_code}] for '{clean_name}': "
            f"{response.text[:200]}"
        )
        response.raise_for_status()
        data = response.json()
        if data.get("status") == "success" and data.get("substitutes"):
            return {"ingredient": ingredient, "substitutes": data["substitutes"]}
        return None
    except requests.exceptions.Timeout:
        logger.error(f"Spoonacular timeout for: {ingredient}")
        return None
    except requests.exceptions.RequestException as e:
        logger.error(f"Spoonacular request error for '{ingredient}': {e}")
        return None
    except Exception as e:
        logger.error(f"Unexpected Spoonacular error: {e}")
        return None


def print_recipe(result: dict):
    """
    Pretty-print the full recipe response including ingredients,
    substitutions, steps and tips.
    """
    if not result.get("success"):
        print(f"❌ Failed: {result.get('error')}")
        return

    recipe = result.get("recipe", {})
    print(f"\n✅ {recipe.get('title', 'Untitled')}")
    print(f"📝 {recipe.get('description', '')}")
    print(
        f"🍽️  Serves {recipe.get('servings')} | "
        f"Prep: {recipe.get('prep_time')} | "
        f"Cook: {recipe.get('cook_time')}"
    )

    print("\n── INGREDIENTS ──────────────────────────────────────────")
    for ing in recipe.get("ingredients", []):
        violates = ing.get("violates", [])
        flag = "⚠️ " if violates else "✅ "
        sub_note = ""
        # Use 'or' to ensure we always have a string even if notes is None
        notes_text = ing.get("notes") or ""
        if "Substituted" in notes_text:
            sub_note = f"  ← {notes_text}"
        print(f"  {flag}{ing.get('amount', ''):15s} {ing.get('name', '')}{sub_note}")
        if violates:
            print(f"              violates: {violates}")

    print("\n── STEPS ────────────────────────────────────────────────")
    for step in recipe.get("steps", []):
        duration = f"({step.get('duration', '')})" if step.get("duration") else ""
        print(f"\n  Step {step.get('step_number', '?')} {duration}")
        print(f"  {step.get('instruction', '')}")

    if recipe.get("tips"):
        print("\n── TIPS ─────────────────────────────────────────────────")
        for tip in recipe.get("tips", []):
            print(f"  💡 {tip}")

    print(f"\n  Session ID: {result.get('session_id', '')}")


# ================================================================================
# RECIPE JSON SCHEMA  (used in the Sous Chef prompt)
# ================================================================================

# Every ingredient carries a "violates" list filled by Gemini using its
# semantic food knowledge. The gatekeeper reads these tags — no string matching.
RECIPE_SCHEMA = """
{
    "title": "Recipe Name",
    "description": "Brief description",
    "servings": 4,
    "prep_time": "15 minutes",
    "cook_time": "30 minutes",
    "ingredients": [
        {
            "name": "ingredient name",
            "amount": "1 cup",
            "notes": "optional preparation notes",
            "violates": ["vegan", "dairy_free"]
        }
    ],
    "steps": [
        {"step_number": 1, "instruction": "Detailed instruction", "duration": "5 minutes"}
    ],
    "tips": ["helpful tip 1"]
}

Rules for the "violates" field on EVERY ingredient:
- Must be a JSON array (can be empty []).
- Only use tags from this exact list: """ + json.dumps(ALL_DIETARY_TAGS) + """
- Use your full semantic knowledge of what the ingredient IS — not substrings in its name.
- Examples:
    "butter"            → ["vegan", "dairy_free"]
    "vegan butter"      → []           plant-based, violates nothing
    "coconut milk"      → []           plant-based milk, not dairy
    "cow's milk"        → ["vegan", "dairy_free"]
    "eggs"              → ["vegan", "vegetarian", "egg_free"]
    "chicken"           → ["vegan", "vegetarian"]
    "wheat flour"       → ["gluten_free"]
    "rice flour"        → []
    "white rice"        → ["keto", "low_carb"]
    "garlic"            → ["no_onion_garlic", "jain"]
    "pork"              → ["vegan", "vegetarian", "halal", "kosher"]
    "tofu"              → ["soy_free"]
    "olive oil"         → []
- NEVER flag an ingredient because its name contains a substring of a prohibited
  food. Judge the actual ingredient: "coconut milk" is NOT dairy.
"""


# ================================================================================
# AGENT SYSTEM
# ================================================================================

class DishcoveryAgents:

    def __init__(
        self,
        gemini_api_key: str,
        groq_api_key: str,
        spoonacular_api_key: str,
        chromadb_path: str = ".",
    ):
        self.gemini_client = genai.Client(
            api_key=gemini_api_key, http_options={"api_version": "v1"}
        )
        self.groq_client = Groq(api_key=groq_api_key)
        self.spoonacular_api_key = spoonacular_api_key

        try:
            self.chroma_client = chromadb.PersistentClient(path=chromadb_path)
            self.recipe_collection = self.chroma_client.get_or_create_collection(
                name="indian_recipes",
                metadata={"hnsw:space": "cosine"},
            )
            logger.info(f"Connected to ChromaDB at {chromadb_path}")
        except Exception as e:
            logger.error(f"ChromaDB connection failed: {e}")
            raise

    # ------------------------------------------------------------------
    # GEMINI HELPER  — centralised call with retry on 429
    # ------------------------------------------------------------------
    def _rewrite_steps_with_substitutions(
            self, recipe: Dict[str, Any], subs_made: List[Dict[str, str]]) -> Dict[str, Any]:
            """
            After substitutions are applied to ingredients, rewrite the steps
            so they reference the new ingredient names instead of the originals.
            """
            if not subs_made or not recipe.get("steps"):
                return recipe
        
            sub_summary = "\n".join(
                f'  - "{s["original"]}" → "{s["chosen"]}"'
                for s in subs_made
                if s.get("chosen")
            )
        
            steps_json = json.dumps(recipe["steps"], indent=2)
        
            prompt = f"""The following ingredient substitutions were made in a recipe:
        {sub_summary}
        
        Here are the original recipe steps:
        {steps_json}
        
        Rewrite the steps so they reference the NEW ingredient names instead of the old ones.
        Keep all instructions, timings, and structure identical — only replace the ingredient
        names where they appear in the instruction text.
        
        Return ONLY the updated steps JSON array, no markdown, no extra text."""
        
            try:
                result = self._call_gemini(prompt)
                updated_steps = json.loads(self._clean_json(result))
                recipe["steps"] = updated_steps
                logger.info("✅ Steps rewritten to reflect substitutions")
            except Exception as e:
                logger.error(f"Failed to rewrite steps: {e}. Keeping original steps.")
        
            return recipe
    def _call_gemini(self, prompt: str, max_retries: int = 3) -> str:
        """
        Call Gemini with exponential backoff on 429 RESOURCE_EXHAUSTED.
        Raises on non-retryable errors or after max_retries exceeded.
        """
        for attempt in range(max_retries):
            try:
                response = self.gemini_client.models.generate_content(
                    model="gemini-2.5-flash", contents=prompt
                )
                return response.text.strip()
            except Exception as e:
                err_str = str(e)
                if "429" in err_str and attempt < max_retries - 1:
                    # Parse the retryDelay from the error message if present
                    try:
                        wait = int(err_str.split("retryDelay': '")[1].split("s'")[0]) + 5
                    except Exception:
                        wait = 60 * (attempt + 1)   # fallback: 60s, 120s
                    logger.warning(
                        f"Rate limited (attempt {attempt + 1}/{max_retries}). "
                        f"Waiting {wait}s..."
                    )
                    time.sleep(wait)
                else:
                    raise
        raise RuntimeError("Max Gemini retries exceeded")

    @staticmethod
    def _clean_json(text: str) -> str:
        """Strip markdown fences from a JSON string."""
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]
        return text.strip()

    # ------------------------------------------------------------------
    # SOUS CHEF NODE
    # ------------------------------------------------------------------

    def sous_chef_node(self, state: AgentState) -> AgentState:
        """
        RAG-driven recipe generation using ChromaDB + Gemini.

        Each ingredient is tagged with a "violates" list by Gemini using its
        semantic food knowledge — no string matching anywhere.
        """
        logger.info("👨‍🍳 Sous Chef: Generating recipe...")

        inventory = state.get("inventory", [])
        user_query = state.get("user_query", "")
        user_profile = state.get("user_profile", {})
        dietary_restrictions = user_profile.get("dietary_restrictions", [])
        normalized_inventory = normalize_ingredients(inventory)
        search_query = f"{user_query} {' '.join(normalized_inventory[:5])}"

        try:
            results = self.recipe_collection.query(
                query_texts=[search_query], n_results=3
            )
            top_recipes = []
            if results and results.get("documents"):
                top_recipes = results["documents"][0]

            prompt = f"""You are a professional chef AND a nutrition expert creating a customized recipe.

User Request: {user_query}
Available Ingredients: {', '.join(normalized_inventory) if normalized_inventory else 'Use whatever is appropriate'}
Dietary Restrictions to respect: {', '.join(dietary_restrictions) if dietary_restrictions else 'None'}

Similar Recipes from Database:
{chr(10).join([f"{i+1}. {r}" for i, r in enumerate(top_recipes)]) if top_recipes else 'None available'}

Create a detailed recipe using EXACTLY this JSON schema:
{RECIPE_SCHEMA}

CRITICAL INSTRUCTIONS:
1. The "name" field must contain ONLY the plain ingredient name — no preparation notes.
2. The "violates" field is MANDATORY on every ingredient. Use your semantic knowledge
   of what the ingredient actually IS — not substrings in its name.
   "coconut milk"  → violates []       (plant-based, not dairy)
   "vegan butter"  → violates []       (plant-based)
   "almond milk"   → violates []       (plant-based)
   "cow's milk"    → violates ["vegan","dairy_free"]
   "butter"        → violates ["vegan","dairy_free"]
3. Try to generate a recipe that respects the dietary restrictions.
   If impossible, include the ingredient and tag it correctly — the system
   will handle substitutions automatically.
4. Return ONLY the JSON, no markdown, no extra text."""

            recipe_text = self._call_gemini(prompt)
            recipe_json = json.loads(self._clean_json(recipe_text))

            # Defensive: ensure every ingredient has a "violates" key
            for ing in recipe_json.get("ingredients", []):
                if "violates" not in ing:
                    ing["violates"] = []

            state["current_recipe_json"] = recipe_json
            state["current_step_index"] = 0
            state["substitution_attempts"] = 0
            state["messages"] = state.get("messages", []) + [
                f"Generated recipe: {recipe_json.get('title', 'Untitled')}"
            ]
            state["next_action"] = "gatekeeper"
            logger.info(f"✅ Recipe generated: {recipe_json.get('title')}")

        except Exception as e:
            logger.error(f"Error in Sous Chef: {e}")
            state["error_log"] = state.get("error_log", []) + [
                f"Recipe generation failed: {e}"
            ]
            state["next_action"] = "end"

        return state

    # ------------------------------------------------------------------
    # GATEKEEPER NODE  — tag-based, zero string matching
    # ------------------------------------------------------------------

    def gatekeeper_node(self, state: AgentState) -> AgentState:
        """
        Reads the LLM-supplied "violates" tags on every ingredient and checks
        them against the user's dietary restrictions.

        No substring matching. No false positives from ingredient names.
        "coconut milk", "vegan butter", "almond milk" will never be flagged
        for a vegan/dairy-free user because Gemini tagged them with violates=[].
        """
        logger.info("🔒 Gatekeeper: Checking dietary compliance (tag-based)...")

        recipe = state.get("current_recipe_json")
        user_profile = state.get("user_profile", {})
        dietary_restrictions = set(user_profile.get("dietary_restrictions", []))

        if not recipe or not dietary_restrictions:
            state["next_action"] = "proceed"
            return state

        # Cycle-break guard
        if state.get("substitution_attempts", 0) >= MAX_SUBSTITUTION_ATTEMPTS:
            logger.warning(
                "Max substitution attempts reached — proceeding despite any "
                "remaining violations."
            )
            state["next_action"] = "proceed"
            return state

        violations = []
        substitutions_needed = []

        for ing in recipe.get("ingredients", []):
            ing_name = ing.get("name", "")
            ing_violations = set(ing.get("violates", []))
            conflicts = ing_violations & dietary_restrictions

            if conflicts:
                for conflict in conflicts:
                    violations.append(f"'{ing_name}' violates {conflict}")
                substitutions_needed.append({
                    "original": ing_name,
                    "reason": ", ".join(conflicts),
                })

        if violations:
            logger.warning(
                f"Found {len(violations)} dietary violation(s): "
                f"{'; '.join(violations)}"
            )
            state["error_log"] = state.get("error_log", []) + [
                f"Dietary violations: {'; '.join(violations)}"
            ]
            state["substitutions_needed"] = substitutions_needed
            state["next_action"] = "substitute"
        else:
            logger.info("✅ All ingredients pass dietary checks")
            state["next_action"] = "proceed"

        return state

    # ------------------------------------------------------------------
    # SUBSTITUTION NODE
    # ------------------------------------------------------------------

    def substitution_node(self, state: AgentState) -> AgentState:
        """
        Hybrid substitution: Spoonacular first, LLM fallback.

        After substitution the new ingredient's "violates" tags are re-generated
        by Gemini so the gatekeeper can correctly evaluate the replacement on
        the next pass — no string matching needed.
        """
        logger.info("🔄 Substitution Expert: Finding alternatives...")

        state["substitution_attempts"] = state.get("substitution_attempts", 0) + 1

        substitutions_needed = state.get("substitutions_needed", [])
        recipe = state.get("current_recipe_json", {})
        user_profile = state.get("user_profile", {})
        dietary_restrictions = user_profile.get("dietary_restrictions", [])

        if not substitutions_needed:
            state["next_action"] = "proceed"
            return state

        successful_subs = []

        for sub_request in substitutions_needed:
            original = sub_request["original"]
            reason = sub_request["reason"]

            # ── Step 1: Try Spoonacular ──────────────────────────────────
            api_result = get_spoonacular_substitute(original, self.spoonacular_api_key)

            if api_result and api_result.get("substitutes"):
                candidates = api_result["substitutes"][:5]
                successful_subs.append({
                    "original": original,
                    "candidates": candidates,
                    "reason": reason,
                    "source": "spoonacular",
                })
                logger.info(f"✅ Spoonacular candidates for '{original}': {candidates}")
            else:
                # ── Step 2: LLM fallback — get candidates + re-tag in ONE call ──
                prompt = f"""You are a culinary and nutrition expert.

Find the best substitute for "{original}" that is fully compliant with: {reason}.

Return ONLY this JSON (no markdown):
{{
    "chosen": "the best substitute (short plain ingredient name)",
    "alternatives": ["second option", "third option"],
    "violates": ["dietary_tag1", "dietary_tag2"]
}}

Rules:
- "chosen" must genuinely comply with: {reason}
- "violates" lists tags from {json.dumps(ALL_DIETARY_TAGS)} that the CHOSEN ingredient
  conflicts with. Use semantic knowledge — judge by what the ingredient IS.
  Example: "coconut milk" → violates [] even though it contains the word "milk".
- If the chosen ingredient complies with everything, return "violates": []
- Prefer substitutes with similar flavor and texture to "{original}"
- Use common, accessible ingredients with short plain names"""

                try:
                    result_text = self._call_gemini(prompt)
                    pick_result = json.loads(self._clean_json(result_text))

                    chosen_name = pick_result.get("chosen", "")
                    chosen_violates = pick_result.get("violates", [])
                    alternatives = pick_result.get("alternatives", [])

                    if chosen_name:
                        successful_subs.append({
                            "original": original,
                            "candidates": [chosen_name] + alternatives,
                            "chosen": chosen_name,
                            "chosen_violates": chosen_violates,
                            "reason": reason,
                            "source": "llm",
                        })
                        logger.info(
                            f"✅ LLM substitute for '{original}': '{chosen_name}' "
                            f"(violates={chosen_violates})"
                        )
                except Exception as e:
                    logger.error(f"Failed to find substitute for '{original}': {e}")

        # ── Step 3: Apply substitutions to recipe ────────────────────────
        # For Spoonacular results we still need a separate pick/re-tag call.
        # For LLM results the chosen+violates are already in the sub dict.
        if successful_subs and "ingredients" in recipe:
            for sub in successful_subs:
                if not sub.get("candidates"):
                    continue

                # LLM path: chosen and violates already resolved in one call
                if sub["source"] == "llm" and sub.get("chosen"):
                    chosen_name = sub["chosen"]
                    chosen_violates = sub.get("chosen_violates", [])
                else:
                    # Spoonacular path: need a pick/re-tag call
                    candidates_str = json.dumps(sub["candidates"])
                    pick_prompt = f"""From this list of candidate substitutes for "{sub['original']}":
{candidates_str}

Pick the BEST one that:
1. Is fully compliant with: {sub['reason']}
2. Has the most similar flavor and texture to "{sub['original']}"
3. Is a common accessible ingredient

Return ONLY this JSON (no markdown):
{{
    "chosen": "the best substitute name (short, plain)",
    "violates": ["dietary_tag1"]
}}

"violates" lists tags from {json.dumps(ALL_DIETARY_TAGS)} that the chosen ingredient
conflicts with. Use semantic knowledge — judge by what the ingredient IS.
If fully compliant with everything, return "violates": []"""

                    try:
                        pick_text = self._call_gemini(pick_prompt)
                        pick_result = json.loads(self._clean_json(pick_text))
                        chosen_name = pick_result.get("chosen", sub["candidates"][0])
                        chosen_violates = pick_result.get("violates", [])
                    except Exception as e:
                        logger.error(
                            f"Failed to pick/re-tag substitute for '{sub['original']}': {e}"
                        )
                        chosen_name = sub["candidates"][0]
                        chosen_violates = []

                logger.info(
                    f"Replacing '{sub['original']}' → '{chosen_name}' "
                    f"(violates={chosen_violates}, source={sub['source']})"
                )

                # Update the ingredient in the recipe
                for ing in recipe["ingredients"]:
                    if ing.get("name", "").lower() == sub["original"].lower():
                        ing["notes"] = (
                            f"Substituted from: {ing['name']} ({sub['source']})"
                        )
                        ing["name"] = chosen_name
                        ing["violates"] = chosen_violates   # re-tagged by LLM
                        break
        # After the loop that applies substitutions to recipe["ingredients"]:

        # Build the list of what was actually substituted for the rewrite
        subs_made = [
            {"original": s["original"], "chosen": s.get("chosen") or chosen_name}
            for s in successful_subs
        ]
        
        # Rewrite steps to match the new ingredient names
        recipe = self._rewrite_steps_with_substitutions(recipe, subs_made)
        state["current_recipe_json"] = recipe
        state["messages"] = state.get("messages", []) + [
            f"Applied {len(successful_subs)} substitution(s) "
            f"(attempt {state['substitution_attempts']}/{MAX_SUBSTITUTION_ATTEMPTS})"
        ]
        state["next_action"] = "gatekeeper"

        return state

    # ------------------------------------------------------------------
    # FIXER NODE
    # ------------------------------------------------------------------

    def fixer_node(self, state: AgentState) -> AgentState:
        """Deep reasoning agent using Groq/Llama to fix cooking mistakes."""
        logger.info("🔧 Fixer: Analysing mistake and creating recovery plan...")

        fix_request = state.get("fix_request", "")
        recipe = state.get("current_recipe_json", {})
        current_step = state.get("current_step_index", 0)

        if not fix_request:
            state["next_action"] = "proceed"
            return state

        recipe_context = json.dumps(recipe, indent=2)
        current_step_info = ""
        if "steps" in recipe and current_step < len(recipe["steps"]):
            current_step_info = recipe["steps"][current_step].get("instruction", "")

        prompt = f"""You are an expert chef troubleshooting a cooking mistake.

Recipe Context:
{recipe_context}

Current Step ({current_step + 1}): {current_step_info}

User's Problem: {fix_request}

Analyse the situation and provide a recovery plan in JSON format:
{{
    "analysis": "Brief analysis of what went wrong",
    "recovery_steps": [
        {{"step_number": 1, "instruction": "Recovery action", "duration": "time needed"}}
    ],
    "insert_before_step": 3,
    "modifications": [
        {{"original_step": 2, "modification": "Adjust this step by..."}}
    ],
    "tips": ["Prevention tip for future"]
}}

Focus on practical, immediate solutions."""

        try:
            chat_completion = self.groq_client.chat.completions.create(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are an expert chef specialising in troubleshooting "
                            "cooking mistakes. Provide practical, immediate solutions."
                        ),
                    },
                    {"role": "user", "content": prompt},
                ],
                model="llama-3.3-70b-versatile",
                temperature=0.3,
                max_tokens=2000,
            )

            response_text = chat_completion.choices[0].message.content.strip()
            if "```json" in response_text:
                response_text = response_text.split("```json")[1].split("```")[0]
            elif "```" in response_text:
                response_text = response_text.split("```")[1].split("```")[0]

            fix_plan = json.loads(response_text.strip())

            if fix_plan.get("recovery_steps"):
                recovery_steps = fix_plan["recovery_steps"]
                insert_before = fix_plan.get("insert_before_step")
                if insert_before and "steps" in recipe:
                    for i, step in enumerate(recovery_steps):
                        recipe["steps"].insert(insert_before - 1 + i, step)
                    for idx, step in enumerate(recipe["steps"]):
                        step["step_number"] = idx + 1

            if fix_plan.get("modifications") and "steps" in recipe:
                for mod in fix_plan["modifications"]:
                    step_num = mod.get("original_step", 0) - 1
                    if 0 <= step_num < len(recipe["steps"]):
                        recipe["steps"][step_num]["instruction"] += (
                            f" [Modified: {mod.get('modification')}]"
                        )

            state["current_recipe_json"] = recipe
            state["messages"] = state.get("messages", []) + [
                f"Fix applied: {fix_plan.get('analysis', 'Recovery plan created')}"
            ]
            state["error_log"] = state.get("error_log", []) + [
                f"Fixed issue: {fix_request}"
            ]
            state["fix_request"] = None
            state["next_action"] = "proceed"
            logger.info(f"✅ Fix plan: {fix_plan.get('analysis')}")

        except Exception as e:
            logger.error(f"Error in Fixer: {e}")
            state["error_log"] = state.get("error_log", []) + [f"Fix failed: {e}"]
            state["next_action"] = "proceed"

        return state


# ================================================================================
# LANGGRAPH WORKFLOW
# ================================================================================

def route_after_gatekeeper(state: AgentState) -> str:
    if state.get("next_action") == "substitute":
        return "substitute"
    return "proceed"


def create_dishcovery_graph(agents: DishcoveryAgents) -> StateGraph:
    workflow = StateGraph(AgentState)

    workflow.add_node("sous_chef", agents.sous_chef_node)
    workflow.add_node("gatekeeper", agents.gatekeeper_node)
    workflow.add_node("substitute", agents.substitution_node)
    workflow.add_node("fixer", agents.fixer_node)

    workflow.set_entry_point("sous_chef")
    workflow.add_edge("sous_chef", "gatekeeper")
    workflow.add_conditional_edges(
        "gatekeeper",
        route_after_gatekeeper,
        {"substitute": "substitute", "proceed": END},
    )
    workflow.add_edge("substitute", "gatekeeper")
    workflow.add_edge("fixer", END)

    memory = MemorySaver()
    app = workflow.compile(checkpointer=memory)
    logger.info("✅ LangGraph workflow compiled successfully")
    return app


# ================================================================================
# FASTAPI APPLICATION
# ================================================================================

agent_system: Optional[DishcoveryAgents] = None
graph_app = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global agent_system, graph_app

    from dotenv import load_dotenv
    load_dotenv()

    # NOTE: Set these in your .env file
    gemini_key = os.getenv("GEMINI_API_KEY")
    groq_key = os.getenv("GROQ_API_KEY")
    spoonacular_key = os.getenv("SPOONACULAR_API_KEY")
    
    # chromadb.PersistentClient path accepts the directory containing chroma.sqlite3
    chromadb_path = os.getenv("CHROMADB_PATH", ".")

    missing = [
        name
        for name, val in [
            ("GEMINI_API_KEY", gemini_key),
            ("GROQ_API_KEY", groq_key),
            ("SPOONACULAR_API_KEY", spoonacular_key),
        ]
        if not val or "YOUR_" in val
    ]
    if missing:
        raise ValueError(
            f"Required API keys not configured: {', '.join(missing)}"
        )

    agent_system = DishcoveryAgents(
        gemini_api_key=gemini_key,
        groq_api_key=groq_key,
        spoonacular_api_key=spoonacular_key,
        chromadb_path=chromadb_path,
    )
    graph_app = create_dishcovery_graph(agent_system)
    logger.info("🚀 Dishcovery system initialised!")
    yield
    logger.info("👋 Dishcovery system shutting down.")


api = FastAPI(title="Dishcovery API", version="3.0.0", lifespan=lifespan)
api.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Request / Response models ─────────────────────────────────────────────────

class DiscoverRequest(BaseModel):
    user_query: str
    inventory: List[str]
    dietary_restrictions: List[str] = []
    allergies: List[str] = []


class FixRequest(BaseModel):
    session_id: str
    fix_description: str
    current_step: int


class RecipeResponse(BaseModel):
    success: bool
    recipe: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    session_id: str


# ── Endpoints ─────────────────────────────────────────────────────────────────

from fastapi.responses import HTMLResponse

@api.get("/", response_class=HTMLResponse)
async def root():
    try:
        with open("index.html", "r", encoding="utf-8") as f:
            return f.read()
    except FileNotFoundError:
        return "<h1>index.html not found!</h1><p>Please ensure index.html is in the same directory.</p>"


@api.post("/discover", response_model=RecipeResponse)
async def discover_recipe(request: DiscoverRequest):
    try:
        if not graph_app:
            raise HTTPException(status_code=500, detail="System not initialised")

        session_id = f"session_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        initial_state: AgentState = {
            "user_profile": {
                "dietary_restrictions": request.dietary_restrictions,
                "allergies": request.allergies,
            },
            "inventory": request.inventory,
            "current_recipe_json": None,
            "current_step_index": 0,
            "error_log": [],
            "substitutions_needed": [],
            "user_query": request.user_query,
            "fix_request": None,
            "messages": [],
            "next_action": "start",
            "substitution_attempts": 0,
        }

        config = {"configurable": {"thread_id": session_id}}
        final_state = graph_app.invoke(initial_state, config)

        recipe = final_state.get("current_recipe_json")
        if recipe:
            return RecipeResponse(success=True, recipe=recipe, session_id=session_id)

        error_msg = (final_state.get("error_log") or ["Unknown error"])[-1]
        return RecipeResponse(success=False, error=error_msg, session_id=session_id)

    except Exception as e:
        logger.error(f"Error in /discover: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api.post("/fix", response_model=RecipeResponse)
async def fix_recipe(request: FixRequest):
    try:
        if not graph_app or not agent_system:
            raise HTTPException(status_code=500, detail="System not initialised")

        config = {"configurable": {"thread_id": request.session_id}}
        saved = graph_app.get_state(config)

        if not saved or not saved.values:
            raise HTTPException(
                status_code=404,
                detail=f"Session '{request.session_id}' not found. "
                       "Run /discover first.",
            )

        fix_state: AgentState = dict(saved.values)  # type: ignore[arg-type]
        fix_state["fix_request"] = request.fix_description
        fix_state["current_step_index"] = request.current_step

        fixed_state = agent_system.fixer_node(fix_state)

        return RecipeResponse(
            success=True,
            recipe=fixed_state.get("current_recipe_json"),
            session_id=request.session_id,
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error in /fix: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@api.get("/health")
async def health_check():
    return {
        "status": "running",
        "components": {
            "agents": agent_system is not None,
            "graph": graph_app is not None,
            "chromadb": (
                agent_system.recipe_collection is not None if agent_system else False
            ),
        },
    }


# ================================================================================
# MAIN EXECUTION
# ================================================================================

from fastapi.testclient import TestClient  # noqa: E402
import sys

if __name__ == "__main__":
    if "--test" in sys.argv:
        client = TestClient(api)

        # ── Health ────────────────────────────────────────────────────────
        print("\n==============================")
        print("🩺 /health")
        print("==============================")
        print(client.get("/health").json())

        # ── Test 1: Happy Path ────────────────────────────────────────────
        print("\n==============================")
        print("🧪 Test 1: Happy Path (vegetarian)")
        print("==============================")
        print_recipe(client.post("/discover", json={
            "user_query": "Make a healthy Indian dinner",
            "inventory": ["rice", "jeera", "tomato", "paneer"],
            "dietary_restrictions": ["vegetarian"],
            "allergies": [],
        }).json())

        time.sleep(5)

        # ── Test 2: Vegan — coconut milk & vegan butter must NOT be flagged
        print("\n==============================")
        print("🧪 Test 2: Vegan — no false positives")
        print("==============================")
        print_recipe(client.post("/discover", json={
            "user_query": "Rich vegan Indian curry using coconut milk and vegan butter",
            "inventory": ["coconut milk", "vegan butter", "tomato", "garlic", "onion", "spices"],
            "dietary_restrictions": ["vegan"],
            "allergies": [],
        }).json())

        time.sleep(5)

        # ── Test 3: Vegan rejection — real animal products must be flagged
        print("\n==============================")
        print("🧪 Test 3: Vegan rejection (real animal products)")
        print("==============================")
        print_recipe(client.post("/discover", json={
            "user_query": "Make butter chicken",
            "inventory": ["chicken", "dairy butter", "heavy cream", "tomato"],
            "dietary_restrictions": ["vegan"],
            "allergies": [],
        }).json())

        time.sleep(5)

        # ── Test 4: Conflicting constraints ───────────────────────────────
        print("\n==============================")
        print("🧪 Test 4: Conflicting constraints (Vegan + Keto)")
        print("==============================")
        print_recipe(client.post("/discover", json={
            "user_query": "Make dinner",
            "inventory": ["paneer", "rice", "coconut oil", "spinach"],
            "dietary_restrictions": ["vegan", "keto"],
            "allergies": [],
        }).json())

        time.sleep(5)

        # ── Test 5: Gluten-free — wheat flagged, rice flour not ──────────
        print("\n==============================")
        print("🧪 Test 5: Gluten-free precision")
        print("==============================")
        print_recipe(client.post("/discover", json={
            "user_query": "Make Indian flatbread",
            "inventory": ["wheat flour", "rice flour", "water", "salt", "oil"],
            "dietary_restrictions": ["gluten_free"],
            "allergies": [],
        }).json())

        time.sleep(5)

        # ── Test 6: Fix endpoint ──────────────────────────────────────────
        print("\n==============================")
        print("🧪 Test 6: Fix endpoint")
        print("==============================")
        discover_result = client.post("/discover", json={
            "user_query": "Quick vegetarian lunch",
            "inventory": ["tomato", "rice"],
            "dietary_restrictions": ["vegetarian"],
            "allergies": [],
        }).json()
        session_id = discover_result.get("session_id", "")
        print(f"Session: {session_id}")
        print_recipe(discover_result)

        if session_id:
            time.sleep(3)
            print("\n🔧 Applying fix: too much salt...")
            fix_result = client.post("/fix", json={
                "session_id": session_id,
                "fix_description": "I accidentally added too much salt",
                "current_step": 1,
            }).json()
            print("Fix success:", fix_result.get("success"))
            print(
                "Fix analysis:",
                (fix_result.get("recipe") or {})
                .get("steps", [{}])[0]
                .get("instruction", "")[:120]
            )
            print_recipe(fix_result)

        time.sleep(5)

        # ── Test 7: Stress loop ───────────────────────────────────────────
        print("\n==============================")
        print("🧪 Test 7: Stress Loop")
        print("==============================")
        stress_cases = [
            {
                "user_query": "Quick vegetarian lunch",
                "inventory": ["tomato", "rice"],
                "dietary_restrictions": ["vegetarian"],
                "allergies": [],
            },
            {
                "user_query": "High protein vegan meal",
                "inventory": ["tofu", "spinach", "coconut milk"],
                "dietary_restrictions": ["vegan"],
                "allergies": [],
            },
            {
                "user_query": "Comfort food",
                "inventory": ["butter", "paneer", "cream"],
                "dietary_restrictions": [],
                "allergies": [],
            },
        ]
        for i, payload in enumerate(stress_cases, 1):
            print(f"\n🔁 Stress Case {i}: {payload['user_query']}")
            print_recipe(client.post("/discover", json=payload).json())
            if i < len(stress_cases):
                time.sleep(5)
    else:
        import uvicorn
        logger.info("Starting Dishcovery Web Server on http://localhost:8000")
        uvicorn.run("project:api", host="0.0.0.0", port=8000, reload=True)