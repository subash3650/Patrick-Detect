Complete Installation Guide for Backend:
1. Ubuntu/System Dependencies

sudo apt-get update
sudo apt-get install -y python3-dev python3-venv python3-pip
sudo apt-get install -y ffmpeg
sudo apt-get install -y libasound2-dev libsndfile1-dev
sudo apt-get install -y build-essential cmake

2. Create Virtual Environment
cd /mnt/a/PD/Backend
python3 -m venv venv
source venv/bin/activate

3. Install Python Dependencies# Make sure venv is activated
source venv/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

4. Verify Installation
python -c "
import flask
import flask_cors
import joblib
import pandas
import numpy
import scipy
import sklearn
import parselmouth
import pydub
import soundfile
print('✓ All packages installed successfully!')
"

To Start Your Application Backend:
cd /mnt/a/PD/Backend
source venv/bin/activate
python app.py



Complete Installation Guide for Frontend:
cd /mnt/a/PD/frontend
npm install axios
npm start
npm build