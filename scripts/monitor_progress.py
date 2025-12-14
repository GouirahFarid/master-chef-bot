"""
Monitor the progress of data population
"""

import json
import time
import sys
from datetime import datetime

def monitor_progress(interval=10):
    """Monitor progress file and display updates"""
    progress_file = "populate_progress.json"
    last_update = None

    print("📊 Monitoring population progress...")
    print("Press Ctrl+C to stop monitoring\n")

    try:
        while True:
            try:
                with open(progress_file, "r") as f:
                    progress = json.load(f)

                # Only update if file changed
                if progress != last_update:
                    last_update = progress

                    # Calculate percentage
                    total = progress.get("total_rows", 0)
                    processed = progress.get("processed_count", 0)
                    percentage = (processed / total * 100) if total > 0 else 0

                    # Calculate time remaining
                    timestamp = progress.get("timestamp")
                    if timestamp:
                        saved_time = datetime.fromisoformat(timestamp)
                        time_since_save = (datetime.now() - saved_time).total_seconds()
                        status = f"Last save: {time_since_save:.0f}s ago"
                    else:
                        status = "No saves yet"

                    # Clear screen and print progress
                    if sys.platform == "win32":
                        os.system('cls')
                    else:
                        os.system('clear')

                    print("="*60)
                    print(f"🍳 RECIPE DATABASE POPULATION MONITOR")
                    print("="*60)
                    print(f"📄 File: {progress.get('csv_file', 'Unknown')}")
                    print(f"📊 Progress: {processed:,}/{total:,} ({percentage:.2f}%)")
                    print(f"📍 Current Row: {progress.get('current_row', 0):,}")
                    print(f"📦 Batches Processed: {progress.get('batches_processed', 0)}")
                    print(f"⏰ Status: {status}")

                    # Progress bar
                    bar_length = 50
                    filled = int(bar_length * percentage / 100)
                    bar = '█' * filled + '░' * (bar_length - filled)
                    print(f"\n[{bar}] {percentage:.2f}%")

                    # Estimated time remaining
                    if percentage > 0:
                        # Rough estimate based on progress rate
                        elapsed = time.time() - time.mktime(datetime.strptime(timestamp, "%Y-%m-%dT%H:%M:%S.%f").timetuple()) if timestamp else 0
                        if elapsed > 0:
                            rate = processed / elapsed
                            remaining = (total - processed) / rate
                            print(f"\n⏱️  Rate: {rate:.1f} recipes/sec")
                            print(f"⏳ ETA: {remaining/60:.1f} minutes")

            except FileNotFoundError:
                print("⏳ Waiting for progress file to be created...")
            except Exception as e:
                print(f"⚠️ Error reading progress: {e}")

            time.sleep(interval)

    except KeyboardInterrupt:
        print("\n\n✅ Monitoring stopped")

if __name__ == "__main__":
    import os
    monitor_progress()