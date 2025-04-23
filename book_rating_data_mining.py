
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression, RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.cluster import KMeans

# === Load & preprocess ===
books = pd.read_csv("Books.csv", delimiter=';', encoding='latin-1', on_bad_lines='skip')
users = pd.read_csv("Users.csv", delimiter=';', encoding='latin-1', on_bad_lines='skip', low_memory=False)
ratings = pd.read_csv("Ratings.csv", delimiter=';', encoding='latin-1', on_bad_lines='skip')

# Convert merge keys
ratings['User-ID'] = ratings['User-ID'].astype(str)
users['User-ID'] = users['User-ID'].astype(str)
ratings['ISBN'] = ratings['ISBN'].astype(str)
books['ISBN'] = books['ISBN'].astype(str)

# Rename columns for consistency
books = books.rename(columns={'Year-Of-Publication': 'Year'})
data = ratings.merge(books, on='ISBN').merge(users, on='User-ID')
data = data.rename(columns={'Book-Rating': 'Rating'})

# Clean
data = data[data['Rating'] > 0]
data['Year'] = pd.to_numeric(data['Year'], errors='coerce')
data['Age'] = pd.to_numeric(data['Age'], errors='coerce')
data = data.dropna(subset=['Year', 'Age', 'Rating'])

# === 📊 EDA ===
plt.figure(figsize=(8, 4))
sns.histplot(data['Age'], bins=30, kde=True)
plt.title("Distribution of User Age")
plt.xlabel("Age")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 4))
sns.boxplot(data['Rating'])
plt.title("Boxplot of Book Ratings")
plt.tight_layout()
plt.show()

# === 🔍 Clustering ===
kmeans = KMeans(n_clusters=3, random_state=42)
data['Cluster'] = kmeans.fit_predict(data[['Age', 'Rating']])
sns.scatterplot(data=data, x='Age', y='Rating', hue='Cluster', palette='tab10')
plt.title("User Clustering by Age & Rating")
plt.tight_layout()
plt.show()

# === 📈 Modeling: Linear vs Ridge ===
X = data[['Year', 'Age']]
y = data['Rating']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Linear Regression
lr = LinearRegression()
lr.fit(X_train, y_train)
lr_pred = lr.predict(X_test)
lr_rmse = mean_squared_error(y_test, lr_pred) ** 0.5
lr_r2 = r2_score(y_test, lr_pred)

# Ridge Regression
ridge = RidgeCV(alphas=[0.1, 1.0, 10.0])
ridge.fit(X_train, y_train)
ridge_pred = ridge.predict(X_test)
ridge_rmse = mean_squared_error(y_test, ridge_pred) ** 0.5
ridge_r2 = r2_score(y_test, ridge_pred)

# Print evaluation
print("📊 Model Comparison:")
print(f"Linear Regression -> RMSE: {lr_rmse:.2f}, R²: {lr_r2:.2f}")
print(f"Ridge Regression  -> RMSE: {ridge_rmse:.2f}, R²: {ridge_r2:.2f}")

# Plot actual vs predicted for both
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
sns.regplot(x=y_test, y=lr_pred, line_kws={"color": "red"})
plt.title("Linear Regression")
plt.xlabel("Actual"); plt.ylabel("Predicted")

plt.subplot(1, 2, 2)
sns.regplot(x=y_test, y=ridge_pred, line_kws={"color": "green"})
plt.title("Ridge Regression")
plt.xlabel("Actual"); plt.ylabel("Predicted")

plt.tight_layout()
plt.show()
