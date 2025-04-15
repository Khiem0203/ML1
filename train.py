import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

if __name__ == "__main__":
    books_path = "Books.csv"
    users_path = "Users.csv"
    ratings_path = "Ratings.csv"

    books = pd.read_csv(books_path, delimiter=';', encoding='latin-1', on_bad_lines='skip')
    users = pd.read_csv(users_path, delimiter=';', encoding='latin-1', on_bad_lines='skip', low_memory=False)
    ratings = pd.read_csv(ratings_path, delimiter=';', encoding='latin-1', on_bad_lines='skip')

    # === Convert merge keys to same type ===
    ratings['User-ID'] = ratings['User-ID'].astype(str)
    users['User-ID'] = users['User-ID'].astype(str)
    ratings['ISBN'] = ratings['ISBN'].astype(str)
    books['ISBN'] = books['ISBN'].astype(str)

    # === Merge datasets ===
    data = ratings.merge(books, on='ISBN').merge(users, on='User-ID')

    # === 🧹 Clean and prepare ===
    data = data.rename(columns={
        'Book-Rating': 'Rating',
        'Year-Of-Publication': 'Year',
        'Age': 'Age',
        'Book-Title': 'Title'
    })
    data = data[data['Rating'] > 0]
    data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
    data['Age'] = pd.to_numeric(data['Age'], errors='coerce')
    data = data.dropna(subset=['Year', 'Age', 'Rating'])

    # === Feature and target ===
    X = data[['Year', 'Age']]
    y = data['Rating']

    # === Train/test split ===
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # === Train model ===
    model = LinearRegression()
    model.fit(X_train, y_train)

    # === Predict & evaluate ===
    y_pred = model.predict(X_test)
    print(f"\n Model Evaluation:")
    print(f"RMSE: {mean_squared_error(y_test, y_pred) ** 0.5:.2f}")
    print(f"R² Score: {r2_score(y_test, y_pred):.2f}")

    # === Plot Actual vs Predicted ===
    plt.figure(figsize=(6, 6))
    sns.regplot(x=y_test, y=y_pred, line_kws={"color": "red"})
    plt.xlabel("Actual Rating")
    plt.ylabel("Predicted Rating")
    plt.title("Actual vs Predicted Book Ratings")
    plt.tight_layout()
    plt.show()

    # === Interactive Prediction ===
    print("\n Enter user age and book year to get a predicted rating (type 'exit' to quit):")
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
            print(f"➜ Predicted Rating: {prediction:.2f}\n")

            # Recommend books with matching year (±1) and print some titles
            matched_books = books[
                pd.to_numeric(books['Year'], errors='coerce').between(year_val - 1, year_val + 1)
            ]['Title'].dropna().unique()

            print("Books published around that year:")
            for title in matched_books[:5]:
                print(f"• {title}")

            print()

        except ValueError:
            print("Invalid input, please enter numeric values.")
