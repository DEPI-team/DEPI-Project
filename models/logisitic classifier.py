import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load your dataset
data = pd.read_csv(r'movies_with_tags_ratings')

df = pd.DataFrame(data)

# Text Vectorization (TF-IDF)
tfidf = TfidfVectorizer()
X = tfidf.fit_transform(df['description'])  # Transform descriptions into TF-IDF matrix
y = df['title']  # Labels (movie titles)

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# Example of making a prediction
new_description = ['A computer hacker learns about the true nature of his reality.']
new_description_tfidf = tfidf.transform(new_description)
predicted_movie = model.predict(new_description_tfidf)
print(f'Predicted Movie: {predicted_movie[0]}')
