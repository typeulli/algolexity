@echo off
cd /d "%~dp0"

docker build -t algolexity-python python
echo.
echo Docker image 'algolexity-python' has been built successfully.
echo.
pause > nul