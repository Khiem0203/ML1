import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib

books_path = "Books.csv"
users_path = "Users.csv"
ratings_path = "Ratings.csv"

books = pd.read_csv(books_path, delimiter=';', encoding='latin-1', on_bad_lines='skip')
users = pd.read_csv(users_path, delimiter=';', encoding='latin-1', on_bad_lines='skip', low_memory=False)
ratings = pd.read_csv(ratings_path, delimiter=';', encoding='latin-1', on_bad_lines='skip')

ratings['User-ID'] = ratings['User-ID'].astype(str)
users['User-ID'] = users['User-ID'].astype(str)
ratings['ISBN'] = ratings['ISBN'].astype(str)
books['ISBN'] = books['ISBN'].astype(str)

data = ratings.merge(books, on='ISBN').merge(users, on='User-ID')
data = data.rename(columns={'Book-Rating': 'Rating', 'Year-Of-Publication': 'Year', 'Age': 'Age', 'Book-Title': 'Title'})
data = data[data['Rating'] > 0]
data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
data['Age'] = pd.to_numeric(data['Age'], errors='coerce')
data = data.dropna(subset=['Year', 'Age', 'Rating'])

X = data[['Year', 'Age']]
y = data['Rating']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
print(f"RMSE: {mean_squared_error(y_test, y_pred) ** 0.5:.2f}")
print(f"R² Score: {r2_score(y_test, y_pred):.2f}")

a = model.coef_[0]
b = model.coef_[1]
c = model.intercept_
print(f"Predicted Rating = {a:.4f} * Year + {b:.4f} * Age + {c:.4f}")

joblib.dump(model, "trained_model.pkl")
print("Model saved")
