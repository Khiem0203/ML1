import joblib
import pandas as pd

def scale_tfidf(x):
    return x * 2.0

pipeline = joblib.load("linear_model.pkl")

model = pipeline.named_steps["model"]
preprocessor = pipeline.named_steps["transform"]

tfidf_names = preprocessor.named_transformers_["tfidf_title"].named_steps["tfidf"].get_feature_names_out()
author_names = preprocessor.named_transformers_["author"].get_feature_names_out(["Author"])
publisher_names = preprocessor.named_transformers_["publisher"].get_feature_names_out(["Publisher"])
numerical_names = ["Age", "BookAge"]

feature_names = list(tfidf_names * 2) + list(author_names) + list(publisher_names) + numerical_names

coef_df = pd.DataFrame({
    "Feature": feature_names,
    "Weight": model.coef_
})

print(coef_df.reindex(coef_df.Weight.abs().sort_values(ascending=False).index).head(20))
