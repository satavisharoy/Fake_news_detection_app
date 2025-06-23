# Fake News Detection App

A simple web application built using Streamlit that detects whether a news article is real or fake using a machine learning model.

## Features

- Input a news article and get a prediction: Real or Fake
- Built with a trained Logistic Regression model using TF-IDF
- Clean and minimal UI for easy use

## Technologies Used

- Python
- Streamlit
- Scikit-learn
- Joblib
- Pandas

## File Structure

- `app.py` – Main Streamlit app
- `train.py` – Script to train the model
- `model.pkl` – Trained ML model
- `vectorizer.pkl` – TF-IDF vectorizer
- `requirements.txt` – Python dependencies
- `dataset/` – Contains `train.csv` and `test.csv`

## How to Run

1. Clone the repository:

   ```bash
   git clone https://github.com/your_username/Fake_news_detection_app.git
   cd Fake_news_detection_app

2. Install dependencies:

    pip install -r requirements.txt

3. Run the app:

    streamlit run app.py
