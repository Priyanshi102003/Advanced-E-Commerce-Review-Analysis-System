# Amazon Reviews NLP Intelligence App

Streamlit app for Amazon review sentiment analysis, weak fake-review detection, and weak issue classification using NLTK preprocessing, TF-IDF features, and classic machine-learning models.

This application analyzes customer reviews using Natural Language Processing and Machine Learning techniques to identify sentiment, detect potentially fake reviews, classify common product issues, and visualize insights through interactive dashboards.

Dataset source: [Amazon Reviews for Sentiment Analysis on Kaggle](https://www.kaggle.com/datasets/bittlingmayer/amazonreviews). The source files are in fastText format, where `__label__1` is negative sentiment from 1–2 star reviews and `__label__2` is positive sentiment from 4–5 star reviews.

## What It Includes

- Sentiment model comparison using Logistic Regression, Multinomial Naive Bayes, Random Forest, and Linear SVM
- Real-time review predictions with sentiment, issue category, and fake-review risk
- Interactive dashboards for sentiment mix, issue trends, fake-risk distribution, confusion matrices, and model performance
- Top predictive keyword visualization using TF-IDF features
- Modular project architecture under the `src/` directory
- Safe sampling controls for handling large Kaggle datasets
- Upload support for `.bz2`, `.txt`, `.zip`, and `.csv` review datasets
- Modern Streamlit dashboard with analytics and NLP visualizations

## Important Label Note

The Kaggle dataset contains sentiment labels only. It does not provide verified fake-review labels or issue-category labels. This project handles those tasks using transparent weak supervision techniques.

- Issue classification begins with a keyword taxonomy covering delivery, quality, packaging, returns, pricing, usability, authenticity, and related categories
- Fake-review detection uses behavioral text signals such as repetition, excessive punctuation, promotional language, URLs, generic praise, and unusually short reviews
- Weak classifiers can be trained automatically from heuristic labels when sufficient data is available

## Features

- Sentiment Analysis
- Fake Review Detection
- Weak Issue Classification
- NLP Text Preprocessing
- TF-IDF Feature Engineering
- Interactive Analytics Dashboard
- Real-time Predictions
- Model Comparison
- Confusion Matrix Visualization
- Review Upload Support
- Streamlit-based UI
- CPU Compatible

## Setup

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

## Add The Kaggle Data

Option 1: Download the dataset manually from Kaggle and place these files inside `data/raw/`:

- `train.ft.txt.bz2`
- `test.ft.txt.bz2`

Option 2: Use the helper script after configuring Kaggle API credentials:

```powershell
python scripts/download_data.py --target data/raw
```

The application also supports uploading local review datasets directly through the dashboard.

Supported file formats:

- `.bz2`
- `.txt`
- `.zip`
- `.csv`

## Run

```powershell
streamlit run app.py
```

Open the local URL Streamlit prints, usually:

```text
http://localhost:8501
```

## Project Structure

```text
Amazon-Reviews-NLP-App/
│
├── app.py
├── requirements.txt
├── README.md
│
├── src/
│   ├── config.py
│   ├── data_loader.py
│   ├── evaluation.py
│   ├── fake_detection.py
│   ├── modeling.py
│   ├── preprocessing.py
│   └── weak_labeling.py
│
├── scripts/
│   └── download_data.py
│
├── data/
│   └── raw/
│
├── assets/
│   └── screenshots/
│
└── outputs/
```

## Tech Stack

- Python
- Streamlit
- Scikit-learn
- NLTK
- Pandas
- NumPy
- Matplotlib
- Plotly

## Models Used

This project supports multiple Machine Learning models for sentiment classification:

- Logistic Regression
- Multinomial Naive Bayes
- Random Forest
- Linear SVM

## Deployment

You can deploy this project on:

- Streamlit Cloud
- Hugging Face Spaces
- Render
- Railway

## Future Improvements

- Deep Learning integration
- Transformer-based NLP models
- Advanced fake-review detection
- Multi-language review analysis
- Real-time API integration
- User authentication system
- Cloud database support
- GPU acceleration

## Git Commands

### Push Updates

```powershell
git add .
git commit -m "Updated project"
git push
```

### Pull Latest Changes

```powershell
git pull
```

## Contributing

Contributions are welcome.

To contribute:

1. Fork the repository
2. Create a new branch
3. Make improvements
4. Submit a Pull Request

## License

This project is licensed under the MIT License.

## Author

Priyanshi

GitHub:
https://github.com/Priyanshi102003

LinkedIn: 
[https://www.linkedin.com/in/priyanshi-530b4a350)

## Acknowledgements

- Kaggle
- Streamlit
- Scikit-learn
- NLTK
- Open Source NLP Community
