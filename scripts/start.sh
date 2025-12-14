#!/bin/bash

echo "=== Recipe RAG Chatbot Launcher ==="

# Change to project directory
cd "$(dirname "$0")/.."

# Check if PostgreSQL is running
if ! pg_isready -q; then
    echo "PostgreSQL is not running. Please start PostgreSQL first."
    exit 1
fi

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    echo "Warning: Virtual environment not detected."
    echo "Consider activating a virtual environment first:"
    echo "  source .venv/bin/activate"
    echo ""
fi

# Menu
echo "What would you like to do?"
echo "1. Setup database (run once)"
echo "2. Populate with full dataset (2M+ recipes)"
echo "3. Monitor population progress"
echo "4. Start the chatbot"
echo "5. Exit"

read -p "Enter your choice (1-5): " choice

case $choice in
    1)
        echo "Setting up database..."
        python database_setup.py
        ;;
    2)
        echo "Starting data population..."
        echo "This may take several hours for the full dataset."
        echo "The script includes automatic progress saving and resumption."
        echo ""
        echo "Options:"
        echo "  a) Fresh start"
        echo "  b) Resume from previous progress"
        echo "  c) Run with monitoring"
        read -p "Choose option (a/b/c): " option

        case $option in
            a)
                echo "Starting fresh population..."
                python populate_data_v2.py --csv full_dataset.csv
                ;;
            b)
                echo "Resuming from previous progress..."
                python populate_data_v2.py --csv full_dataset.csv --resume
                ;;
            c)
                echo "Starting with progress monitoring..."
                # Start monitor in background
                python scripts/monitor_progress.py &
                MONITOR_PID=$!
                # Run population
                python populate_data_v2.py --csv full_dataset.csv --test
                # Kill monitor when done
                kill $MONITOR_PID 2>/dev/null
                ;;
            *)
                echo "Invalid option"
                ;;
        esac
        ;;
    3)
        echo "Starting progress monitor..."
        python scripts/monitor_progress.py
        ;;
    4)
        echo "Starting Streamlit chatbot..."
        streamlit run app.py --server.port=8501 --server.address=0.0.0.0
        ;;
    5)
        echo "Goodbye!"
        exit 0
        ;;
    *)
        echo "Invalid choice"
        exit 1
        ;;
esac