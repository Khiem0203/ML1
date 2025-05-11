import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_squared_error, mean_absolute_error,
    r2_score, mean_absolute_percentage_error
)
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder, FunctionTransformer
from sklearn.feature_extraction.text import TfidfVectorizer

def scale_tfidf(x):
    return x * 2.0

def train_and_evaluate(data_path="data_merged.csv"):
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

    x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(x_train, y_train)
    y_pred = pipeline.predict(x_test)

    print("RMSE:", mean_squared_error(y_test, y_pred) ** 0.5)
    print("MAE:", mean_absolute_error(y_test, y_pred))
    print("R²:", r2_score(y_test, y_pred))
    print("MAPE:", mean_absolute_percentage_error(y_test, y_pred))
    print("CV RMSE:", -cross_val_score(pipeline, X, y, cv=5, scoring='neg_root_mean_squared_error').mean())

    plt.figure()
    sns.regplot(x=y_test, y=y_pred, line_kws={"color": "red"})
    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")
    plt.title("Actual vs Predicted Ratings")
    plt.tight_layout()
    plt.show()

    joblib.dump(pipeline, "linear_model.pkl")
    print("Model saved as linear_model.pkl")

if __name__ == "__main__":
    train_and_evaluate()
