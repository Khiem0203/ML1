import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    mean_absolute_percentage_error
)
from sklearn.model_selection import cross_val_score

def scale_tfidf(x):
    return x * 2.0

test_df = pd.read_csv("test.csv")
X_test = test_df[['Age', 'BookAge', 'Title', 'Author', 'Publisher']].dropna()
y_test = test_df.loc[X_test.index, 'Rating']

model = joblib.load("linear_model.pkl")

y_pred = model.predict(X_test)

mse = mean_squared_error(y_test, y_pred)
rmse = mse ** 0.5
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
mape = mean_absolute_percentage_error(y_test, y_pred)

cv_rmse = -cross_val_score(model, X_test, y_test, cv=5, scoring='neg_root_mean_squared_error').mean()

print("Evaluation on test.csv using saved model:")
print("RMSE:", round(rmse, 4))
print("MAE:", round(mae, 4))
print("R²:", round(r2, 4))
print("MAPE:", round(mape * 100, 2), "%")
print("CV RMSE:", round(cv_rmse, 4))

plt.figure()
sns.regplot(x=y_test, y=y_pred, line_kws={"color": "red"})
plt.xlabel("Actual Rating")
plt.ylabel("Predicted Rating")
plt.title("Actual vs Predicted Ratings (Test Set)")
plt.tight_layout()
plt.show()
