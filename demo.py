import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import NearestNeighbors

def scale_tfidf(x):
    return x * 2.0

def hybrid_recommend(user_id, books_path, ratings_path, users_path, model_path="linear_model.pkl"):
    model = joblib.load(model_path)

    books = pd.read_csv(books_path, delimiter=';', encoding='latin-1', on_bad_lines='skip', dtype=str, low_memory=False)
    ratings = pd.read_csv(ratings_path, delimiter=';', encoding='latin-1', on_bad_lines='skip', dtype=str, low_memory=False)
    users = pd.read_csv(users_path, delimiter=';', encoding='latin-1', on_bad_lines='skip', dtype=str, low_memory=False)

    books.columns = books.columns.str.strip().str.replace('\ufeff', '')
    ratings.columns = ratings.columns.str.strip().str.replace('\ufeff', '')
    users.columns = users.columns.str.strip().str.replace('\ufeff', '')

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
    data = data[(data['Rating'] > 0) & data['Year'].notna() & data['Age'].between(5, 110)]

    user_data = data[data['User-ID'] == str(user_id)]
    high_rated = user_data[user_data['Rating'] >= 7]

    if not high_rated.empty:
        seed_titles = high_rated['Title'].dropna().unique()
        age = high_rated['Age'].mean()
        seed_year = high_rated['Year'].dropna().mean()
    else:
        if user_data.empty:
            print("No user data available.")
            return
        age = user_data['Age'].mean()
        seed_year = user_data['Year'].dropna().mean()
        community = data[
            (data['Age'].between(age - 5, age + 5)) &
            (data['Year'].between(seed_year - 2, seed_year + 2)) &
            (data['Rating'] >= 7)
        ]
        if community.empty:
            print("No fallback books found.")
            return
        top_books = community.groupby('Title').agg({'Rating': 'mean'}).sort_values(by='Rating', ascending=False)
        print("Suggested books from similar-age readers:")
        print(top_books.head(5))
        return

    tfidf = TfidfVectorizer(max_features=500, stop_words='english')
    tfidf_matrix = tfidf.fit_transform(data['Title'])
    knn = NearestNeighbors(n_neighbors=6, metric='cosine')
    knn.fit(tfidf_matrix)

    similar_books = pd.DataFrame()
    for title in seed_titles:
        matches = data[data['Title'] == title]
        if matches.empty:
            continue
        first_idx = matches.index[0]
        row_pos = data.index.get_loc(first_idx)
        _, indices = knn.kneighbors(tfidf_matrix[row_pos])
        similar = data.iloc[indices[0]]
        similar_books = pd.concat([similar_books, similar], ignore_index=True)

    similar_books = similar_books.drop_duplicates(subset=['Title', 'Author'])
    current_year = 2025
    similar_books['BookAge'] = current_year - similar_books['Year']
    similar_books['Age'] = age

    input_df = similar_books[['Age', 'BookAge', 'Title', 'Author', 'Publisher']].dropna()
    similar_books = similar_books.loc[input_df.index]
    similar_books['Predicted Rating'] = model.predict(input_df)
    similar_books = similar_books.sort_values(by='Predicted Rating', ascending=False)

    print(f"\nBooks rated ≥7 by user {user_id}:\n")
    high_rated = high_rated.reset_index(drop=True)
    for i, row in high_rated.iterrows():
        print(f"{i+1}. {row['Title']} (Rating: {row['Rating']})")

    print(f"\nRecommended books similar to those rated ≥7 by user {user_id}:\n")
    for i, row in similar_books.head(5).iterrows():
        print(f"{i+1}. {row['Title']} (Year: {int(row['Year'])}) → Predicted: {row['Predicted Rating']:.2f}")

if __name__ == "__main__":
    user_id = input("Enter user ID: ").strip()
    hybrid_recommend(
        user_id=user_id,
        books_path="Books.csv",
        ratings_path="Ratings.csv",
        users_path="Users.csv",
        model_path="linear_model.pkl"
    )
