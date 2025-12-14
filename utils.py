"""
Utility functions for Master Chef
"""

import psycopg2
import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import pipeline, AutoTokenizer, AutoModelForCausalLM
import torch
import re
from typing import List, Tuple, Optional, Dict, Any

# Database configuration
DB_CONFIG = {
    'dbname': 'recipe_rag',
    'user': 'postgres',
    'password': 'root',
    'host': 'localhost',
    'port': '5432'
}

class Database:
    """Database connection and query handler"""

    def __init__(self, config: Dict[str, str] = None):
        self.config = config or DB_CONFIG
        self.connection = None

    def connect(self):
        """Connect to PostgreSQL database"""
        try:
            self.connection = psycopg2.connect(**self.config)
            return True
        except Exception as e:
            print(f"Database connection error: {e}")
            return False

    def disconnect(self):
        """Close database connection"""
        if self.connection:
            self.connection.close()
            self.connection = None

    def execute_query(self, query: str, params: tuple = None) -> List[Tuple]:
        """Execute a query and return results"""
        if not self.connection:
            if not self.connect():
                return []

        try:
            cursor = self.connection.cursor()
            cursor.execute(query, params)
            return cursor.fetchall()
        except Exception as e:
            print(f"Query error: {e}")
            return []

    def get_recipe_count(self) -> int:
        """Get total number of recipes"""
        results = self.execute_query("SELECT COUNT(*) FROM recipes")
        return results[0][0] if results else 0

    def get_database_info(self) -> Dict[str, Any]:
        """Get database information"""
        info = {}

        # Get recipe count
        info['recipe_count'] = self.get_recipe_count()

        # Get metadata
        results = self.execute_query('''
            SELECT model_name, embedding_dimension, total_recipes, dataset_version, created_at
            FROM embedding_metadata
            ORDER BY created_at DESC
            LIMIT 1
        ''')
        if results:
            info['model'] = results[0]

        return info


class EmbeddingModel:
    """Handle sentence embeddings for recipes"""

    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        self.model_name = model_name
        self.model = None
        self.dimension = 384  # Default for paraphrase-multilingual-MiniLM-L12-v2

    def load(self):
        """Load the embedding model"""
        try:
            self.model = SentenceTransformer(self.model_name)
            return True
        except Exception as e:
            print(f"Error loading embedding model: {e}")
            return False

    def encode(self, text: str | List[str]) -> np.ndarray:
        """Encode text(s) to embeddings"""
        if not self.model:
            raise ValueError("Model not loaded. Call load() first.")
        return self.model.encode(text)

    def list_to_vector_str(self, vector: np.ndarray) -> str:
        """Convert numpy array to PostgreSQL vector string format"""
        return '[' + ','.join(map(str, vector)) + ']'


class GenerationModel:
    """Handle text generation for responses"""

    def __init__(self, model_name: str = 'distilgpt2'):
        self.model_name = model_name
        self.generator = None
        self.tokenizer = None

    def load(self):
        """Load the generation model"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.tokenizer.pad_token = self.tokenizer.eos_token

            model = AutoModelForCausalLM.from_pretrained(self.model_name)
            device = 0 if torch.cuda.is_available() else -1

            self.generator = pipeline(
                "text-generation",
                model=model,
                tokenizer=self.tokenizer,
                device=device
            )
            return True
        except Exception as e:
            print(f"Error loading generation model: {e}")
            return False

    def generate(self, prompt: str, max_tokens: int = 150) -> str:
        """Generate text based on prompt"""
        if not self.generator:
            raise ValueError("Model not loaded. Call load() first.")

        try:
            result = self.generator(
                prompt,
                max_new_tokens=max_tokens,
                num_return_sequences=1,
                do_sample=True,
                top_k=50,
                top_p=0.95,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                truncation=True
            )

            response = result[0]['generated_text']
            # Extract only the generated part
            if prompt in response:
                response = response.replace(prompt, "").strip()

            return response
        except Exception as e:
            return f"Error generating response: {str(e)}"


class RecipeSearch:
    """Handle recipe search using vector similarity"""

    def __init__(self, db: Database, embedding_model: EmbeddingModel):
        self.db = db
        self.embedding_model = embedding_model

    def search(self, query: str, k: int = 3, min_similarity: float = 0.3) -> List[Dict]:
        """Search for similar recipes"""
        # Generate query embedding
        query_embedding = self.embedding_model.encode([query])[0]
        query_vector_str = self.embedding_model.list_to_vector_str(query_embedding)

        # Search in database
        sql = '''
            SELECT id, title, ingredients, directions, chunk_text,
                   1 - (embedding <=> %s::vector) as similarity
            FROM recipes
            WHERE 1 - (embedding <=> %s::vector) >= %s
            ORDER BY embedding <=> %s::vector
            LIMIT %s
        '''

        results = self.db.execute_query(sql, (query_vector_str, query_vector_str, min_similarity, query_vector_str, k))

        # Convert to list of dictionaries
        recipes = []
        for row in results:
            recipes.append({
                'id': row[0],
                'title': row[1],
                'ingredients': row[2],
                'directions': row[3],
                'chunk_text': row[4],
                'similarity': row[5]
            })

        return recipes

    def format_context(self, recipes: List[Dict]) -> str:
        """Format search results into context string"""
        if not recipes:
            return "No recipes found matching your query."

        context_parts = []
        for i, recipe in enumerate(recipes):
            context_parts.append(
                f"Recipe {i+1} (Similarity: {recipe['similarity']:.2f})\n"
                f"Title: {recipe['title']}\n"
                f"Content: {recipe['chunk_text']}"
            )

        return "\n\n".join(context_parts)


def format_response_prompt(query: str, context: str) -> str:
    """Create a prompt for response generation"""
    # Limit context length
    max_context_length = 2000
    if len(context) > max_context_length:
        context = context[:max_context_length] + "..."

    prompt = f"""You are a helpful cooking assistant. Based on the following recipe information, provide a clear and helpful answer to the user's question.

Context:
{context}

User Question: {query}

Helpful Answer:"""

    return prompt


def format_ingredients(ingredients: List[str]) -> str:
    """Format ingredients list for display"""
    if not ingredients:
        return "No ingredients listed"

    # Limit and format
    limited = ingredients[:10]
    if len(ingredients) > 10:
        limited.append(f"... and {len(ingredients) - 10} more ingredients")

    return "\n• " + "\n• ".join(str(ing) for ing in limited)


def format_directions(directions: str, max_length: int = 500) -> str:
    """Format directions for display"""
    if not directions:
        return "No directions provided"

    # Clean and limit
    directions = re.sub(r'\s+', ' ', directions).strip()
    if len(directions) > max_length:
        directions = directions[:max_length] + "..."

    return directions