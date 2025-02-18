# sentiment_analysis.py
import pandas as pd
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier  # Izmantojam priekšpadeves neironu tīklu
from sklearn.metrics import classification_report

# Funkcija datu ielādei
def load_data(file_path):
    data = pd.read_csv(file_path, sep='\t', header=None, names=['sentiment', 'tweet'])
    return data

# Teksta priekšapstrāde
def clean_text(text):
    text = text.lower()  # Pārvēršam tekstu mazajiem burtiem
    text = re.sub(r'http\S+', '', text)  # Dzēšam saites
    text = re.sub(r'@\S+', '', text)  # Dzēšam lietotāju minējumus
    text = re.sub(r'[^a-zāēīļņōūž0-9\s]', '', text)  # Dzēšam nevēlamus simbolus
    return text

# Ielādējam apmācības datus
train_data = load_data('tweets.train')
train_data['cleaned_tweet'] = train_data['tweet'].apply(clean_text)

# Ielādējam pozitīvos un negatīvos vārdus
def load_word_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return set(f.read().splitlines())

positive_words = load_word_list('lv_positive_words')
negative_words = load_word_list('lv_negative_words')

# Funkcija funkciju izveidošanai
def feature_engineering(text, positive_words, negative_words):
    pos_count = sum(1 for word in text.split() if word in positive_words)
    neg_count = sum(1 for word in text.split() if word in negative_words)
    return [pos_count, neg_count]

# Izveidojam funkcijas
train_data['features'] = train_data['cleaned_tweet'].apply(lambda x: feature_engineering(x, positive_words, negative_words))

# Tfidf vektorizācija
vectorizer = TfidfVectorizer(max_features=5000)
X = vectorizer.fit_transform(train_data['cleaned_tweet'])
y = train_data['sentiment']

# Datu sadalīšana treniņa un testēšanas kopās
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Izveidojam un apmācām priekšpadeves neironu tīklu (MLPClassifier)
model = MLPClassifier(hidden_layer_sizes=(50,), max_iter=1000, random_state=42)
model.fit(X_train, y_train)

# Novērtējam rezultātus
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))

# Prognozes uz jauniem tvītiem
def predict_sentiment(model, vectorizer, input_texts):
    cleaned_texts = [clean_text(text) for text in input_texts]
    X_input = vectorizer.transform(cleaned_texts)
    predictions = model.predict(X_input)
    return predictions

input_tweets = [
    "@ivca79 @nilsusakovs Paldies par info! Reaģēsim!",
    "@AnitraTooma izcili, un nespēju novaldīt smaidu :)",
    "@mansLMT kurš izstrādāja šo divplākšņu pakalpojumu?"
]

predictions = predict_sentiment(model, vectorizer, input_tweets)

# Saglabājam rezultātus
def save_predictions(predictions, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for pred in predictions:
            f.write(f"{pred}\n")

save_predictions(predictions, 'output.txt')
