"""
PostgreSQL Database Setup for Master Chef System
Sets up the database schema with pgvector extension for storing embeddings
"""

import psycopg2
import sys

# Database configuration - update these with your credentials
DB_CONFIG = {
    'dbname': 'recipe_rag',
    'user': 'postgres',
    'password': 'root',  # Change this to your PostgreSQL password
    'host': 'localhost',
    'port': '5432'
}


def connect_to_postgres():
    """Connect to default postgres database"""
    try:
        conn = psycopg2.connect(
            database='postgres',
            user=DB_CONFIG['user'],
            password=DB_CONFIG['password'],
            host=DB_CONFIG['host'],
            port=DB_CONFIG['port']
        )
        return conn
    except Exception as e:
        print(f"Error connecting to PostgreSQL: {e}")
        return None


def connect_to_recipe_db():
    """Connect to recipe_rag database"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        return conn
    except Exception as e:
        print(f"Error connecting to recipe_rag database: {e}")
        return None


def create_database():
    """Create the recipe_rag database if it doesn't exist"""
    conn = connect_to_postgres()
    if not conn:
        return False

    try:
        conn.autocommit = True
        cursor = conn.cursor()

        # Check if database exists
        cursor.execute(f"SELECT 1 FROM pg_database WHERE datname='{DB_CONFIG['dbname']}'")
        if not cursor.fetchone():
            cursor.execute(f'CREATE DATABASE {DB_CONFIG["dbname"]}')
            print(f"Database '{DB_CONFIG['dbname']}' created successfully!")
        else:
            print(f"Database '{DB_CONFIG['dbname']}' already exists.")

        conn.close()
        return True

    except Exception as e:
        print(f"Error creating database: {e}")
        conn.close()
        return False


def setup_extensions_and_tables():
    """Setup pgvector extension and create tables"""
    conn = connect_to_recipe_db()
    if not conn:
        return False

    try:
        conn.autocommit = True
        cursor = conn.cursor()

        # Enable pgvector extension
        cursor.execute('CREATE EXTENSION IF NOT EXISTS vector')
        print("pgvector extension enabled!")

        # Create recipes table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS recipes (
                id SERIAL PRIMARY KEY,
                title TEXT NOT NULL,
                ingredients TEXT[],
                directions TEXT,
                source TEXT,
                chunk_text TEXT NOT NULL,
                embedding vector(384),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            );
        ''')

        # Create HNSW index for faster vector search (better for large datasets)
        cursor.execute('''
                       CREATE INDEX IF NOT EXISTS recipes_embedding_hnsw_idx
                           ON recipes
                           USING hnsw (embedding vector_cosine_ops)
                           WITH (m = 16, ef_construction = 64);
                       ''')

        # Create additional indexes
        # Check if pg_trgm extension is available
        cursor.execute("SELECT 1 FROM pg_available_extensions WHERE name = 'pg_trgm'")
        if cursor.fetchone():
            cursor.execute(
                "CREATE EXTENSION IF NOT EXISTS pg_trgm"
            )
            # Create GIN index for ingredients array (pg_trgm is for text similarity, not needed for tsvector)
            try:
                # Use to_tsvector for full-text search
                cursor.execute(
                    "CREATE INDEX IF NOT EXISTS recipes_title_idx ON recipes USING gin(to_tsvector('english', title))"
                )
                print("Created title text search index")
            except Exception as idx_error:
                print(f"Warning: Could not create text search index: {idx_error}")
                # Fallback: simple trgm index on title if pg_trgm is available
                try:
                    cursor.execute(
                        "CREATE INDEX IF NOT EXISTS recipes_title_trgm_idx ON recipes USING gin(title gin_trgm_ops)"
                    )
                    print("Created title trigram index instead")
                except:
                    print("Warning: Could not create any text index on title")

        # Create GIN index for ingredients array
        cursor.execute("CREATE INDEX IF NOT EXISTS recipes_ingredients_idx ON recipes USING gin(ingredients)")

        # Create metadata table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS embedding_metadata (
                id SERIAL PRIMARY KEY,
                model_name VARCHAR(255) NOT NULL,
                embedding_dimension INTEGER NOT NULL,
                total_recipes INTEGER DEFAULT 0,
                dataset_version VARCHAR(50),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                UNIQUE(model_name, embedding_dimension)
            );
        ''')

        # Create a table for tracking population progress
        cursor.execute('''
                       CREATE TABLE IF NOT EXISTS population_status
                       (
                           id
                           SERIAL
                           PRIMARY
                           KEY,
                           status
                           TEXT
                           NOT
                           NULL,
                           processed
                           INTEGER
                           DEFAULT
                           0,
                           total
                           INTEGER
                           DEFAULT
                           0,
                           error_message
                           TEXT,
                           updated_at
                           TIMESTAMP
                           DEFAULT
                           CURRENT_TIMESTAMP
                       );
                       ''')

        print("Database tables and indexes created successfully!")
        conn.close()
        return True

    except Exception as e:
        print(f"Error setting up database: {e}")
        conn.close()
        return False


def verify_setup():
    """Verify the database setup"""
    conn = connect_to_recipe_db()
    if not conn:
        return False

    try:
        cursor = conn.cursor()

        # Check pgvector extension
        cursor.execute("SELECT * FROM pg_extension WHERE extname = 'vector'")
        if cursor.fetchone():
            print("pgvector extension: [OK] Active")
        else:
            print("pgvector extension: [ERROR] Not found")
            return False

        # Check tables
        cursor.execute('''
                       SELECT table_name
                       FROM information_schema.tables
                       WHERE table_schema = 'public'
                       ''')
        tables = [row[0] for row in cursor.fetchall()]
        required_tables = ['recipes', 'embedding_metadata', 'population_status']

        for table in required_tables:
            if table in tables:
                print(f"Table '{table}': [OK] Created")
            else:
                print(f"Table '{table}': [ERROR] Not found")
                return False

        # Check indexes
        cursor.execute('''
                       SELECT indexname
                       FROM pg_indexes
                       WHERE schemaname = 'public'
                         AND tablename = 'recipes'
                       ''')
        indexes = [row[0] for row in cursor.fetchall()]
        if 'recipes_embedding_hnsw_idx' in indexes:
            print("Vector index: [OK] HNSW index created")
        else:
            print("Vector index: [WARNING] Creating HNSW index...")
            cursor.execute('''
                           CREATE INDEX CONCURRENTLY IF NOT EXISTS recipes_embedding_hnsw_idx
                               ON recipes
                               USING hnsw (embedding vector_cosine_ops)
                               WITH (m = 16, ef_construction = 64);
                           ''')

        conn.close()
        return True

    except Exception as e:
        print(f"Error verifying setup: {e}")
        conn.close()
        return False


def main():
    print("=== PostgreSQL Database Setup ===")
    print("\nPrerequisites:")
    print("1. PostgreSQL 14+ with pgvector extension installed")
    print("2. Update DB_CONFIG in this script with your credentials")
    print()

    # Create database
    if not create_database():
        print("\n[X] Failed to create database")
        sys.exit(1)

    # Setup extensions and tables
    if not setup_extensions_and_tables():
        print("\n[X] Failed to setup extensions and tables")
        sys.exit(1)

    # Verify setup
    if verify_setup():
        print("\n[SUCCESS] Database setup complete!")
        print("\nNext steps:")
        print("1. Run 'python populate_data_v2.py --csv full_dataset.csv' to populate with recipes")
        print("2. Run 'streamlit run app.py' to start the chatbot")
    else:
        print("\n[X] Database setup verification failed")
        sys.exit(1)


if __name__ == "__main__":
    main()
