import pandas as pd
import numpy as np
import re
import string
import nltk
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from imblearn.over_sampling import SMOTE
from imblearn.combine import SMOTETomek
from sklearn.svm import SVC

# Download NLTK resources (required only for first run)
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer


def preprocess_text(text):
    # Convert to lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove code blocks wrapped in backticks
    text = re.sub(r'`.*?`', '', text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove numbers
    text = re.sub(r'\d+', '', text)
    # Remove extra whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    # Tokenize, remove stopwords, lemmatize, and filter short words
    lemmatizer = WordNetLemmatizer()
    tokens = text.split()
    filtered_tokens = [lemmatizer.lemmatize(token) for token in tokens
                       if token not in stopwords.words('english') and len(token) > 2]
    return " ".join(filtered_tokens)


def load_data():
    df = pd.read_csv("datasets/pytorch.csv", encoding="utf-8")
    df.dropna(inplace=True)
    df["cleaned_text"] = (df["Title"].astype(str) + " " + df["Body"].astype(str)).apply(preprocess_text)

    # Determine label column
    if "class" in df.columns:
        df["label"] = df["class"]
    elif "related" in df.columns:
        df["label"] = df["related"]
    else:
        raise ValueError("No category label column found, please check the dataset!")
    return df


def train_and_evaluate():
    df = load_data()

    accuracy_scores, precision_scores, recall_scores, f1_scores, auc_scores, auc_cv_list = [], [], [], [], [], []

    for i in range(10):  # 10 experiments
        X_train, X_test, y_train, y_test = train_test_split(
            df["cleaned_text"], df["label"], test_size=0.3, random_state=i, stratify=df["label"])

        # TF-IDF feature extraction
        tfidf = TfidfVectorizer(ngram_range=(1, 1), max_df=0.9, min_df=3, sublinear_tf=True)
        X_train_tfidf = tfidf.fit_transform(X_train)
        X_test_tfidf = tfidf.transform(X_test)

        # Handle class imbalance (using SMOTE)
        smote_tomek = SMOTETomek(sampling_strategy=0.7, random_state=i)  # Oversample then clean
        X_train_tfidf, y_train = smote_tomek.fit_resample(X_train_tfidf, y_train)

        # Train SVM model (with class balancing)
        model = SVC(kernel='linear', probability=True, class_weight='balanced')
        model.fit(X_train_tfidf, y_train)

        # Predictions
        y_pred = model.predict(X_test_tfidf)
        y_pred_proba = model.predict_proba(X_test_tfidf)[:, 1]

        # Calculate metrics
        accuracy_scores.append(accuracy_score(y_test, y_pred))
        precision_scores.append(precision_score(y_test, y_pred, zero_division=0))
        recall_scores.append(recall_score(y_test, y_pred, zero_division=0))
        f1_scores.append(f1_score(y_test, y_pred, zero_division=0))
        auc_scores.append(roc_auc_score(y_test, y_pred_proba))

        # Cross-validated AUC
        auc_cv = cross_val_score(model, X_train_tfidf, y_train, cv=5, scoring="roc_auc")
        auc_cv_list.extend(auc_cv)

    # Output final averaged results
    print("Number of repeats: 10")
    print(f"Average Accuracy: {np.mean(accuracy_scores):.4f}")
    print(f"Average Precision: {np.mean(precision_scores):.4f}")
    print(f"Average Recall: {np.mean(recall_scores):.4f}")
    print(f"Average F1 score: {np.mean(f1_scores):.4f}")
    print(f"Average AUC: {np.mean(auc_scores):.4f}")
    print(f"CV_list(AUC): {auc_cv_list}")


if __name__ == "__main__":
    train_and_evaluate()