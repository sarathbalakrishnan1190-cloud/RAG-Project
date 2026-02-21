@echo off
echo Setting up DocuMind Enterprise...
echo.

REM Activate virtual environment
call .\venv\Scripts\activate

REM Install dependencies
echo Installing Python dependencies...
pip install -r requirements.txt

echo.
echo Setup complete!
echo.
echo To run the application:
echo 1. Start Redis: docker run -p 6379:6379 redis
echo 2. Start Celery Worker: celery -A celery_worker worker --loglevel=info --pool=solo
echo 3. Start API Server: uvicorn server:app --reload
pause
