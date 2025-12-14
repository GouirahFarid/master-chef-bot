"""
Streamlit Master Chef Application
Interactive web interface for recipe search and Q&A
"""

import streamlit as st
import sys
import os
import time
from datetime import datetime

# Add current directory to path for imports
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from utils import (
    Database, EmbeddingModel, GenerationModel, RecipeSearch,
    format_response_prompt, format_ingredients, format_directions
)

# Configure Streamlit page
st.set_page_config(
    page_title="Master Chef",
    page_icon="🍳",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better styling
st.markdown("""
<style>
    .recipe-card {
        background-color: #1b1e23;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .similarity-badge {
        background-color: #e1f5fe;
        color: #01579b;
        padding: 0.25rem 0.75rem;
        border-radius: 1rem;
        font-size: 0.875rem;
        display: inline-block;
        margin-bottom: 0.5rem;
    }
    .message-content {
        padding: 1rem;
    }
    .source-expander {
        background-color: #fafafa;
        border: 1px solid #e0e0e0;
        border-radius: 0.5rem;
        margin-top: 1rem;
    }
    .stats-card {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def init_session_state():
    """Initialize Streamlit session state"""
    if 'db' not in st.session_state:
        st.session_state.db = Database()
    if 'embedding_model' not in st.session_state:
        st.session_state.embedding_model = EmbeddingModel()
    if 'generation_model' not in st.session_state:
        st.session_state.generation_model = GenerationModel()
    if 'recipe_search' not in st.session_state:
        st.session_state.recipe_search = None
    if 'messages' not in st.session_state:
        st.session_state.messages = []
    if 'db_connected' not in st.session_state:
        st.session_state.db_connected = False
    if 'models_loaded' not in st.session_state:
        st.session_state.models_loaded = False

def render_sidebar():
    """Render the sidebar with controls and information"""
    with st.sidebar:
        st.title("🍳 Master Chef")
        st.markdown("---")

        # Connection status
        st.subheader("📊 System Status")

        # Database connection
        if st.button("🔌 Connect to Database", type="primary"):
            with st.spinner("Connecting to database..."):
                if st.session_state.db.connect():
                    st.session_state.db_connected = True
                    st.success("✅ Connected to PostgreSQL")
                    st.rerun()
                else:
                    st.error("❌ Failed to connect")

        # Model loading
        if st.session_state.db_connected:
            if st.button("🤖 Load Models", type="primary"):
                with st.spinner("Loading models..."):
                    # Load embedding model
                    if st.session_state.embedding_model.load():
                        st.success("✅ Embedding model loaded")
                    else:
                        st.error("❌ Failed to load embedding model")

                    # Load generation model
                    if st.session_state.generation_model.load():
                        st.success("✅ Generation model loaded")
                        st.session_state.models_loaded = True
                    else:
                        st.error("❌ Failed to load generation model")

                    # Initialize recipe search
                    st.session_state.recipe_search = RecipeSearch(
                        st.session_state.db,
                        st.session_state.embedding_model
                    )
                    st.rerun()

        # Status indicators
        st.write("---")
        st.write("**Connection Status:**")
        st.write(f"Database: {'✅' if st.session_state.db_connected else '❌'}")
        st.write(f"Models: {'✅' if st.session_state.models_loaded else '❌'}")

        # Database statistics
        if st.session_state.db_connected:
            st.write("---")
            st.write("**Database Statistics:**")
            info = st.session_state.db.get_database_info()

            if 'recipe_count' in info:
                st.metric("Total Recipes", f"{info['recipe_count']:,}")

            if 'model' in info:
                model_info = info['model']
                st.caption(f"Model: {model_info[0]}")
                st.caption(f"Dimension: {model_info[1]}")
                st.caption(f"Dataset: {model_info[3] or 'Unknown'}")

        # Search parameters
        if st.session_state.models_loaded:
            st.write("---")
            st.write("**Search Parameters:**")
            st.session_state.k_results = st.slider(
                "Recipes to retrieve",
                min_value=1,
                max_value=10,
                value=3,
                help="Number of recipes to retrieve for each query"
            )
            st.session_state.min_similarity = st.slider(
                "Minimum similarity",
                min_value=0.0,
                max_value=1.0,
                value=0.3,
                step=0.05,
                help="Minimum similarity threshold for recipe matching"
            )

def render_message(message):
    """Render a chat message"""
    with st.chat_message(message["role"]):
        st.write(message["content"])

        # Show sources for assistant messages
        if message["role"] == "assistant" and "sources" in message:
            with st.expander("📚 Source Recipes", expanded=False):
                for i, recipe in enumerate(message["sources"], 1):
                    st.markdown(f"""
                    <div class="recipe-card">
                        <div class="similarity-badge">Similarity: {recipe['similarity']:.2%}</div>
                        <h4>{recipe['title']}</h4>
                        <p><strong>Ingredients:</strong></p>
                        <div style="margin-left: 1rem;">
                            {format_ingredients(recipe['ingredients'])}
                        </div>
                        <p style="margin-top: 1rem;"><strong>Directions Preview:</strong></p>
                        <div style="margin-left: 1rem; font-size: 0.9rem; color: #c1baba;">
                            {format_directions(recipe['directions'])}
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

def handle_query(query):
    """Handle a user query"""
    if not st.session_state.models_loaded:
        st.error("Please load the models first")
        return

    # Add user message
    st.session_state.messages.append({"role": "user", "content": query})

    # Search for recipes
    with st.spinner("Searching for recipes..."):
        recipes = st.session_state.recipe_search.search(
            query,
            k=st.session_state.k_results,
            min_similarity=st.session_state.min_similarity
        )

    if recipes:
        # Generate response
        with st.spinner("Generating response..."):
            context = st.session_state.recipe_search.format_context(recipes)
            prompt = format_response_prompt(query, context)
            response = st.session_state.generation_model.generate(prompt)

        # Add assistant message with sources
        st.session_state.messages.append({
            "role": "assistant",
            "content": response,
            "sources": recipes
        })
    else:
        # No recipes found
        response = "I couldn't find any recipes matching your query. Please try different keywords or ingredients."
        st.session_state.messages.append({
            "role": "assistant",
            "content": response
        })

def clear_chat():
    """Clear chat history"""
    st.session_state.messages = []
    st.rerun()

def main():
    """Main application logic"""
    # Initialize session state
    init_session_state()

    # Render sidebar
    render_sidebar()

    # Main content area
    st.title("🍳 Master Chef Assistant")
    st.markdown("Ask me anything about cooking! I'll search through thousands of recipes to help you.")

    # Show warning if not ready
    if not st.session_state.db_connected:
        st.warning("⚠️ Please connect to the database first")
        return

    if not st.session_state.models_loaded:
        st.warning("⚠️ Please load the models first")
        return

    # Chat history
    st.write("---")
    for message in st.session_state.messages:
        render_message(message)

    # Chat input
    if query := st.chat_input("Ask about a recipe... (e.g., 'How do I make chocolate cake?')"):
        handle_query(query)
        st.rerun()

    # Clear chat button
    if st.session_state.messages:
        st.markdown("---")
        col1, col2, col3 = st.columns([1, 1, 5])
        with col1:
            if st.button("🗑️ Clear Chat"):
                clear_chat()

    # Example queries
    if not st.session_state.messages:
        st.markdown("---")
        st.subheader("💡 Try asking:")
        example_queries = [
            "How do I make a chocolate cake?",
            "What can I cook with chicken and rice?",
            "Give me a vegetarian pasta recipe",
            "How to make homemade pizza?",
            "Quick dinner ideas with ground beef"
        ]

        cols = st.columns(2)
        for i, query in enumerate(example_queries):
            with cols[i % 2]:
                if st.button(query, key=f"example_{i}"):
                    handle_query(query)
                    st.rerun()

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666; font-size: 0.875rem;'>
        Powered by PostgreSQL + pgvector | Sentence Transformers | DistilGPT2
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()