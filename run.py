import os
import sys
import subprocess

VENV_DIR = "venv"

# Create virtual environment if it doesn't exist
if not os.path.exists(VENV_DIR):
    print("Creating virtual environment...")
    subprocess.check_call([sys.executable, "-m", "venv", VENV_DIR])

# Locate the Python executable inside the venv
if os.name == "nt":  # Windows
    venv_python = os.path.join(VENV_DIR, "Scripts", "python.exe")
else:  # Linux/macOS
    venv_python = os.path.join(VENV_DIR, "bin", "python")

# Upgrade pip (optional but recommended)
print("Upgrading pip...")
subprocess.check_call([
    venv_python,
    "-m",
    "pip",
    "install",
    "--upgrade",
    "pip"
])

# Install dependencies
print("Installing requirements...")
subprocess.check_call([
    venv_python,
    "-m",
    "pip",
    "install",
    "-r",
    "requirements.txt"
])

# Run the main application
print("Starting application...")
subprocess.check_call([
    venv_python,
    "main.py"
])