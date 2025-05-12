import pandas as pd
from datetime import datetime
import os
from sklearn.model_selection import train_test_split

def clean_column_names(df):
    df.columns = df.columns.str.strip().str.replace('\ufeff', '')
    return df

def run_preprocessing(books_path, users_path, ratings_path, save_path="data_merged.csv"):
    books = pd.read_csv(books_path, delimiter=';', encoding='latin-1', on_bad_lines='skip')
    users = pd.read_csv(users_path, delimiter=';', encoding='latin-1', on_bad_lines='skip', dtype=str, low_memory=False)
    ratings = pd.read_csv(ratings_path, delimiter=';', encoding='latin-1', on_bad_lines='skip')

    books = clean_column_names(books)
    users = clean_column_names(users)
    ratings = clean_column_names(ratings)

    ratings['User-ID'] = ratings['User-ID'].astype(str)
    users['User-ID'] = users['User-ID'].astype(str)
    ratings['ISBN'] = ratings['ISBN'].astype(str)
    books['ISBN'] = books['ISBN'].astype(str)

    books = books.rename(columns={'Year-Of-Publication': 'Year', 'Book-Title': 'Title'})
    ratings = ratings.rename(columns={'Book-Rating': 'Rating'})

    books['Year'] = pd.to_numeric(books['Year'], errors='coerce')
    users['Age'] = pd.to_numeric(users['Age'], errors='coerce')
    ratings['Rating'] = pd.to_numeric(ratings['Rating'], errors='coerce')

    data = ratings.merge(books, on='ISBN').merge(users, on='User-ID')
    data = data[data['Rating'] > 0]
    data = data.dropna(subset=['Year', 'Age', 'Rating', 'Title'])
    data = data[(data['Age'] >= 5) & (data['Age'] <= 110)]
    data['Liked'] = (data['Rating'] >= 7).astype(int)
    data['BookAge'] = datetime.now().year - data['Year']

    full_path = save_path
    data.to_csv(full_path, index=False)
    print(f"Data saved to {full_path}")

    train_df, test_df = train_test_split(data, test_size=0.2, random_state=42)

    train_df.to_csv("train.csv", index=False)
    test_df.to_csv("test.csv", index=False)

    print(f"Train set: {len(train_df)} books")
    print(f"Test set: {len(test_df)} books")

if __name__ == "__main__":
    run_preprocessing("Books.csv", "Users.csv", "Ratings.csv")
