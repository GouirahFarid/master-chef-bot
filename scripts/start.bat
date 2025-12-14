@echo off
echo === Master Chef Launcher ===

REM Change to project directory
cd /d "%~dp0.."

REM Check if virtual environment exists
if not exist ".venv\Scripts\activate.bat" (
    echo Warning: Virtual environment not found.
    echo Consider creating one first: python -m venv .venv
    echo.
)

REM Menu
echo What would you like to do?
echo 1. Setup database (run once)
echo 2. Populate with full dataset (2M+ recipes)
echo 3. Start Master Chef
echo 4. Exit

set /p choice="Enter your choice (1-4): "

if "%choice%"=="1" (
    echo Setting up database...
    python database_setup.py
) else if "%choice%"=="2" (
    echo Starting data population...
    echo This may take several hours for the full dataset.
    echo The script includes automatic progress saving and resumption.
    echo.
    echo Options:
    echo   a) Fresh start
    echo   b) Resume from previous progress
    echo   c) Fresh start with testing
    set /p option="Choose option (a/b/c): "

    if /i "%option%"=="a" (
        echo Starting fresh population...
        python populate_data.py --csv full_dataset.csv
    ) else if /i "%option%"=="b" (
        echo Resuming from previous progress...
        python populate_data.py --csv full_dataset.csv --resume
    ) else if /i "%option%"=="c" (
        echo Starting fresh population with testing...
        python populate_data.py --csv full_dataset.csv --test
    ) else (
        echo Invalid option
    )
) else if "%choice%"=="3" (
    echo Starting Master Chef...
    streamlit run app.py --server.port=8501
) else if "%choice%"=="4" (
    echo Goodbye!
    exit /b 0
) else (
    echo Invalid choice
    exit /b 1
)