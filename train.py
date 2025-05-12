import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

def scale_tfidf(x):
    return x * 2.0

def train(data_path="train.csv"):
    df = pd.read_csv(data_path)
    df = df[['Age', 'BookAge', 'Title', 'Author', 'Publisher', 'Rating']].dropna()
    X = df[['Age', 'BookAge', 'Title', 'Author', 'Publisher']]
    y = df['Rating']

    top_authors = X['Author'].value_counts().nlargest(30).index
    top_publishers = X['Publisher'].value_counts().nlargest(30).index
    X.loc[:, 'Author'] = X['Author'].apply(lambda a: a if a in top_authors else 'Other')
    X.loc[:, 'Publisher'] = X['Publisher'].apply(lambda p: p if p in top_publishers else 'Other')

    column_transformer = ColumnTransformer(transformers=[
        ('tfidf_title', Pipeline([
            ('tfidf', TfidfVectorizer(max_features=500, stop_words='english')),
            ('scale', FunctionTransformer(scale_tfidf, validate=False))
        ]), 'Title'),
        ('author', OneHotEncoder(handle_unknown='ignore'), ['Author']),
        ('publisher', OneHotEncoder(handle_unknown='ignore'), ['Publisher']),
        ('num', StandardScaler(), ['Age', 'BookAge'])
    ])

    pipeline = Pipeline([
        ('transform', column_transformer),
        ('model', LinearRegression())
    ])

    pipeline.fit(X, y)
    joblib.dump(pipeline, "linear_model.pkl")
    print("Model saved as linear_model.pkl")

if __name__ == "__main__":
    train()
