import pickle
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression

client = {
  "lead_source": "paid_ads",
  "industry": "technology",
  "employment_status": "employed",
  "location": "north_america",
  "number_of_courses_viewed": 2,
  "annual_income": 79276.0,
  "interaction_count": 4,
  "lead_score": 0.41
}

model_file = "pipeline.bin"

with open(model_file, 'rb') as f_in:
    dv, model = pickle.load(f_in)
  
X = dv.transform([client])
y_pred = model.predict_proba(X)[0, 1]

print(y_pred)