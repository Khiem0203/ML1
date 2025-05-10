import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

books_path = "Books.csv"
ratings_path = "Ratings.csv"

books = pd.read_csv(books_path, delimiter=';', encoding='latin-1', on_bad_lines='skip')
ratings = pd.read_csv(ratings_path, delimiter=';', encoding='latin-1', on_bad_lines='skip')

ratings['ISBN'] = ratings['ISBN'].astype(str)
books['ISBN'] = books['ISBN'].astype(str)

books = books.rename(columns={'Year-Of-Publication': 'Year', 'Book-Title': 'Title'})
books['Year'] = pd.to_numeric(books['Year'], errors='coerce')

model = joblib.load("trained_model.pkl")

print("Enter user age and book year to get a predicted rating (type 'exit' to quit):")
while True:
    age_input = input("User Age: ")
    if age_input.lower() == 'exit':
        break
    year_input = input("Book Year: ")
    if year_input.lower() == 'exit':
        break

    try:
        age_val = float(age_input)
        year_val = float(year_input)
        sample = pd.DataFrame({'Year': [year_val], 'Age': [age_val]})
        prediction = model.predict(sample)[0]
        print(f"Predicted Rating: {prediction:.2f}")

        matched_books = books[
            books['Year'].between(year_val - 1, year_val + 1)
        ]['Title'].dropna().unique()

        print("Books published around that year:")
        for title in matched_books[:5]:
            print(title)

    except ValueError:
        print("Invalid input, please enter numeric values.")
