# Credit Risk Analyser

An AI-powered web application to assess credit risk from scanned or digital bank statement PDFs.

---

## Tech Stack Used
Python 3.9+ (Base Language)<br>
Pandas, NumPy (Data Handling and Preprocessing)<br>
XGBoost, joblib (ML Model Training and Serialization)<br>
Gemini API (PDF Content Extraction and Aggregation)<br>

---

## Setup Instruction

### Create Virtual Environment
python -m venv venv<br>
venv\Scripts\activate.bat

### Install Dependencies 
pip install -r requirements.txt

### Add your gemini api
create a .env file in root folder:<br>
GEMINI_API_KEY=your_gemini_api_key_here

---

## How to Run

1. Place your pdfs in data/raw_pdfs
2. Run the extractor: python extract.py
3. The summary .csv file will be saved in data/extracted_csv
4. Run the predictor: python predict.py
5. The prediction will be shown in terminal.

---
