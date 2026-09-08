# Local Flask prototype

Academic prototype only; not for clinical diagnosis or treatment.

## Windows PowerShell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
python app.py

## macOS/Linux
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python app.py

Open http://127.0.0.1:5000
