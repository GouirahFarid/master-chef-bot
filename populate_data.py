"""
Master Chef Data Population Module
Enhanced version with robust progress saving and resumption
"""

import psycopg2
import pandas as pd
import numpy as np
from sentence_transformers import SentenceTransformer
import re
from ast import literal_eval
from tqdm import tqdm
import time
import sys
import json
from datetime import datetime
import os
import signal
import multiprocessing as mp

# Database configuration
DB_CONFIG = {
    'dbname': 'recipe_rag',
    'user': 'postgres',
    'password': 'root',
    'host': 'localhost',
    'port': '5432'
}

# Configuration
EMBEDDING_MODEL = 'paraphrase-multilingual-MiniLM-L12-v2'
EMBEDDING_DIM = 384
BATCH_SIZE = 5000  # Number of recipes to process in each batch
CHUNK_SIZE = 5000  # Number of embeddings to generate at once
SAVE_INTERVAL = 1000  # Save progress after every N recipes
AUTOSAVE_INTERVAL = 300  # Auto-save every N seconds (5 minutes)
N_WORKERS = mp.cpu_count() - 1  # Number of workers for parallel processing

# Global variables for progress tracking
current_progress = {
    "csv_file": "full_dataset.csv",
    "processed_count": 0,
    "current_row": 0,
    "total_rows": 0,
    "timestamp": None,
    "batches_processed": 0
}

# Progress file name
PROGRESS_FILE = "populate_progress.json"

# Graceful shutdown handler
def signal_handler(signum, frame):
    print(f"\n\n[WARNING] Interrupt received! Saving progress...")
    save_progress_file()
    sys.exit(0)

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)

def connect_to_db():
    """Connect to PostgreSQL database"""
    try:
        conn = psycopg2.connect(**DB_CONFIG)
        conn.autocommit = False
        return conn
    except Exception as e:
        print(f"Error connecting to database: {e}")
        return None

def clean_text(text: str) -> str:
    """Clean text data"""
    if pd.isna(text) or text is None:
        return ""
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    text = text.replace("\xa0", " ")
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")
    return text.strip()

def parse_ingredients(ingredients_str):
    """Parse ingredients from string format to list"""
    if pd.isna(ingredients_str) or ingredients_str is None:
        return []

    # Try to parse as literal
    try:
        ingredients = literal_eval(ingredients_str)
        if isinstance(ingredients, list):
            return [clean_text(ing) for ing in ingredients if ing and str(ing).strip()]
    except:
        pass

    # Fallback: split by common delimiters
    ingredients_str = str(ingredients_str)
    ingredients = re.split(r'[,;\n]', ingredients_str)
    return [clean_text(ing) for ing in ingredients if ing.strip()]

def save_progress_file():
    """Save detailed progress to file"""
    # Add timestamp
    current_progress["timestamp"] = datetime.now().isoformat()

    # Save to file
    try:
        with open(PROGRESS_FILE, "w") as f:
            json.dump(current_progress, f, indent=2)
        print(f"\n[OK] Progress saved: {current_progress['processed_count']:,} recipes processed")
        print(f"   Row: {current_progress['current_row']:,}/{current_progress['total_rows']:,}")
    except Exception as e:
        print(f"[WARNING] Failed to save progress: {e}")

def load_progress_file():
    """Load detailed progress from file"""
    try:
        with open(PROGRESS_FILE, "r") as f:
            progress = json.load(f)
        print(f"[OK] Progress file found")
        print(f"   Previously processed: {progress['processed_count']:,} recipes")
        print(f"   Last row: {progress['current_row']:,}")
        print(f"   Saved at: {progress['timestamp']}")
        return progress
    except FileNotFoundError:
        print("📄 No previous progress file found")
        return None
    except Exception as e:
        print(f"[WARNING] Failed to load progress: {e}")
        return None

def update_status(status, processed=0, total=0, error_message=None):
    """Update population status in database"""
    conn = connect_to_db()
    if not conn:
        return

    try:
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO population_status (status, processed, total, error_message)
            VALUES (%s, %s, %s, %s)
        ''', (status, processed, total, error_message))
        conn.commit()
    except Exception as e:
        print(f"Error updating status: {e}")
    finally:
        conn.close()

def get_total_recipes(csv_file):
    """Get total number of recipes in CSV"""
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            return sum(1 for _ in f) - 1  # Subtract 1 for header
    except:
        return 0

def check_existing_progress():
    """Check if there's existing data and ask if user wants to resume or start fresh"""
    conn = connect_to_db()
    if not conn:
        return False, 0

    try:
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM recipes")
        existing_count = cursor.fetchone()[0]

        if existing_count > 0:
            print(f"\n[WARNING] Found {existing_count:,} recipes already in database!")
            choice = input("Do you want to resume? (y/n): ").lower()
            return choice == 'y', existing_count

        return False, 0
    except:
        return False, 0
    finally:
        conn.close()

def insert_recipes_batch(conn, recipes):
    """Insert a batch of recipes into database"""
    if not recipes:
        return

    cursor = conn.cursor()

    # Prepare data for insertion
    insert_data = []
    for recipe in recipes:
        embedding_str = '[' + ','.join(map(str, recipe['embedding'])) + ']'
        insert_data.append((
            recipe['title'],
            recipe['ingredients'],
            recipe['directions'],
            'full_dataset',
            recipe['chunk_text'],
            embedding_str
        ))

    # Batch insert
    cursor.executemany('''
        INSERT INTO recipes (title, ingredients, directions, source, chunk_text, embedding)
        VALUES (%s, %s, %s, %s, %s, %s::vector)
    ''', insert_data)

    conn.commit()

def populate_database(csv_file="full_dataset.csv", resume=False):
    """Populate database with recipes from CSV with resumption support"""
    global current_progress

    print(f"\n=== Enhanced Database Population ===")
    print(f"CSV file: {csv_file}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"Batch size: {BATCH_SIZE}")
    print(f"Chunk size: {CHUNK_SIZE}")
    print(f"Auto-save interval: {AUTOSAVE_INTERVAL} seconds")

    # Check for existing data
    has_existing, existing_count = check_existing_progress()

    # Load progress if resuming
    start_row = 0
    processed_count = 0

    if resume or has_existing:
        progress = load_progress_file()
        if progress:
            start_row = progress.get("current_row", 0)
            processed_count = progress.get("processed_count", 0)
            print(f"\n📍 Resuming from row {start_row:,}")
            print(f"📍 Already processed: {processed_count:,} recipes")
        elif has_existing:
            # Database has data but no progress file
            print(f"\n[WARNING] Database has {existing_count:,} recipes but no progress file")
            print("   Continuing from where we left off...")
            start_row = existing_count
            processed_count = existing_count

    # Get total count
    total_lines = get_total_recipes(csv_file)
    print(f"\n📊 Total recipes in CSV: {total_lines:,}")

    # Initialize progress tracking
    current_progress.update({
        "csv_file": csv_file,
        "processed_count": processed_count,
        "current_row": start_row,
        "total_rows": total_lines,
        "timestamp": None,
        "batches_processed": 0
    })

    # Update initial status
    update_status("Processing", processed_count, total_lines)

    # Load embedding model
    print(f"\n[MODEL] Loading embedding model...")
    model = SentenceTransformer(EMBEDDING_MODEL)

    # Connect to database
    conn = connect_to_db()
    if not conn:
        print("[ERROR] Failed to connect to database")
        return

    # Clear existing data only if not resuming
    if not resume and not has_existing:
        print("\n[CLEAR] Removing existing data...")
        cursor = conn.cursor()
        cursor.execute("TRUNCATE TABLE recipes CASCADE")
        cursor.execute("DELETE FROM embedding_metadata")
        conn.commit()

    # Processing variables
    all_recipes = []
    start_time = time.time()
    last_autosave = time.time()

    try:
        # Open CSV file with proper encoding
        print(f"\n[CSV] Reading CSV file...")
        # Skip rows but keep the header row
        skip_range = list(range(1, start_row + 1)) if start_row > 0 else None
        csv_reader = pd.read_csv(csv_file, chunksize=BATCH_SIZE, dtype=str, skiprows=skip_range)

        print(f"[PROCESS] Processing recipes (starting from row {start_row:,})...")

        for chunk_idx, chunk_df in enumerate(csv_reader):
            current_row = start_row + chunk_idx * BATCH_SIZE

            # Process recipes in this chunk
            for idx, row in chunk_df.iterrows():
                try:
                    title = clean_text(row.get('title', ''))
                    ingredients = parse_ingredients(row.get('ingredients', ''))
                    directions = clean_text(row.get('directions', ''))

                    if title and len(ingredients) > 0 and directions and len(directions) > 50:
                        ingredients_str = ", ".join(ingredients[:20])
                        chunk_text = f"Title: {title}. Ingredients: {ingredients_str}. Instructions: {directions[:1000]}"

                        all_recipes.append({
                            'title': title,
                            'ingredients': ingredients[:20],
                            'directions': directions[:2000],
                            'chunk_text': chunk_text
                        })
                except Exception as e:
                    # Log the error but continue processing
                    print(f"Warning: Error processing row {current_row + idx}: {e}")
                    continue

            # Generate embeddings when we have enough recipes or at end of file
            if len(all_recipes) >= CHUNK_SIZE or current_row + len(chunk_df) >= total_lines:
                if all_recipes:
                    print(f"\n[EMBED] Generating embeddings for {len(all_recipes)} recipes...")
                    texts = [r['chunk_text'] for r in all_recipes]
                    embeddings = model.encode(texts, batch_size=32, show_progress_bar=True)

                    # Prepare for database insertion
                    for i, recipe in enumerate(all_recipes):
                        recipe['embedding'] = embeddings[i]

                    # Insert into database
                    print(f"[SAVE] Inserting {len(all_recipes)} recipes into database...")
                    insert_recipes_batch(conn, all_recipes)

                    # Update counters
                    processed_count += len(all_recipes)
                    current_progress["processed_count"] = processed_count
                    current_progress["current_row"] = current_row
                    current_progress["batches_processed"] += 1

                    all_recipes = []

                    # Periodic progress updates
                    if processed_count % SAVE_INTERVAL == 0:
                        # Update database status
                        update_status("Processing", processed_count, total_lines)

                        # Calculate stats
                        elapsed = time.time() - start_time
                        rate = processed_count / elapsed if elapsed > 0 else 0
                        eta = (total_lines - processed_count) / rate if rate > 0 else 0
                        percentage = (processed_count / total_lines * 100) if total_lines > 0 else 0

                        print(f"\n[PROGRESS] Progress: {processed_count:,}/{total_lines:,} ({percentage:.1f}%)")
                        print(f"[TIME]  Rate: {rate:.1f} recipes/sec | ETA: {eta/60:.1f} minutes")
                        print(f"[ELAPSED] Elapsed: {elapsed/60:.1f} minutes")

                    # Auto-save to file periodically
                    if time.time() - last_autosave > AUTOSAVE_INTERVAL:
                        save_progress_file()
                        last_autosave = time.time()

    except KeyboardInterrupt:
        print("\n\n[WARNING] Population interrupted by user!")
        if all_recipes:
            print(f"⏳ Saving {len(all_recipes)} pending recipes...")
            insert_recipes_batch(conn, all_recipes)
            current_progress["processed_count"] += len(all_recipes)
    except Exception as e:
        print(f"\n[ERROR] Error during population: {e}")
        update_status("Error", processed_count, total_lines, str(e))
        if all_recipes:
            print(f"⏳ Saving {len(all_recipes)} pending recipes...")
            insert_recipes_batch(conn, all_recipes)
            current_progress["processed_count"] += len(all_recipes)
    finally:
        # Save final progress
        save_progress_file()

        # Update metadata
        cursor = conn.cursor()
        cursor.execute('''
            INSERT INTO embedding_metadata (model_name, embedding_dimension, total_recipes, dataset_version)
            VALUES (%s, %s, %s, %s)
            ON CONFLICT (model_name, embedding_dimension)
            DO UPDATE SET
                total_recipes = EXCLUDED.total_recipes,
                dataset_version = EXCLUDED.dataset_version,
                created_at = CURRENT_TIMESTAMP
        ''', (EMBEDDING_MODEL, EMBEDDING_DIM, processed_count, "full_dataset_v2"))
        conn.commit()

        # Create vector index if not exists
        print(f"\n[CHECK] Checking vector index...")
        cursor.execute('''
            SELECT indexname FROM pg_indexes
            WHERE tablename = 'recipes' AND indexname = 'recipes_embedding_hnsw_idx'
        ''')
        if not cursor.fetchone():
            print(f"⚡ Creating HNSW vector index (this may take a while)...")
            cursor.execute('''
                CREATE INDEX CONCURRENTLY recipes_embedding_hnsw_idx
                ON recipes
                USING hnsw (embedding vector_cosine_ops)
                WITH (m = 16, ef_construction = 64)
            ''')
            conn.commit()
        else:
            print(f"[OK] Vector index already exists")

        conn.close()

    # Final statistics
    total_time = time.time() - start_time
    print(f"\n[OK] Population complete!")
    print(f"📊 Total recipes processed: {processed_count:,}")
    print(f"[TIME]  Total time: {total_time/60:.1f} minutes")
    print(f"🚀 Average rate: {processed_count/total_time:.1f} recipes/second")
    print(f"📁 Progress saved to: {PROGRESS_FILE}")

    # Cleanup progress file on successful completion
    if processed_count >= total_lines - 100:  # Allow for some variance
        try:
            os.remove(PROGRESS_FILE)
            print(f"[CLEAN]  Progress file cleaned up (population complete)")
        except:
            pass

def test_population():
    """Test the population with a small sample"""
    print("\n=== Testing Population ===")
    conn = connect_to_db()
    if not conn:
        print("[ERROR] Cannot connect to database")
        return

    cursor = conn.cursor()

    # Get recipe count
    cursor.execute("SELECT COUNT(*) FROM recipes")
    count = cursor.fetchone()[0]
    print(f"📊 Total recipes in database: {count:,}")

    if count > 0:
        # Test vector search
        model = SentenceTransformer(EMBEDDING_MODEL)
        query = "chocolate cake recipe"
        query_embedding = model.encode([query])[0]
        query_vector_str = '[' + ','.join(map(str, query_embedding)) + ']'

        cursor.execute('''
            SELECT title, 1 - (embedding <=> %s::vector) as similarity
            FROM recipes
            ORDER BY embedding <=> %s::vector
            LIMIT 5
        ''', (query_vector_str, query_vector_str))

        results = cursor.fetchall()
        print(f"\n[CHECK] Top 5 recipes for '{query}':")
        for i, (title, similarity) in enumerate(results, 1):
            print(f"{i}. {title} (similarity: {similarity:.3f})")

    conn.close()

def main():
    """Main function with command-line arguments"""
    import argparse

    parser = argparse.ArgumentParser(description='Populate database with recipes from CSV')
    parser.add_argument('--csv', default='full_dataset.csv', help='CSV file path (default: full_dataset.csv)')
    parser.add_argument('--resume', action='store_true', help='Resume from previous progress')
    parser.add_argument('--test', action='store_true', help='Test the population after completion')

    args = parser.parse_args()

    print(f"\n[START] Population with CSV: {args.csv}")
    print(f"[RESUME] Mode: {'Yes' if args.resume else 'No'}")

    # Populate database
    populate_database(args.csv, resume=args.resume)

    # Test if requested
    if args.test:
        print("\n" + "="*50)
        test_population()

if __name__ == "__main__":
    main()