import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier

# Load Titanic dataset
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
    return pd.read_csv(url)

df = load_data()

# Sidebar for user input
st.sidebar.title("Titanic Survival Prediction")
pclass = st.sidebar.radio("Passenger Class", [1, 2, 3])
sex = st.sidebar.radio("Sex", ["male", "female"])
age = st.sidebar.slider("Age", 0, 80, 30)
sibsp = st.sidebar.slider("Siblings/Spouses Aboard", 0, 8, 0)
parch = st.sidebar.slider("Parents/Children Aboard", 0, 6, 0)
fare = st.sidebar.slider("Fare", 0, 500, 50)

# Model training
df["Sex"] = df["Sex"].map({"male": 0, "female": 1})
df.dropna(subset=["Age", "Fare"], inplace=True)
X = df[["Pclass", "Sex", "Age", "SibSp", "Parch", "Fare"]]
y = df["Survived"]

model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X, y)

# Prediction
user_input = pd.DataFrame([[pclass, 1 if sex == "female" else 0, age, sibsp, parch, fare]], 
                          columns=X.columns)
prediction = model.predict(user_input)[0]

st.sidebar.subheader("Prediction:")
st.sidebar.success("✅ Survived" if prediction == 1 else "❌ Did Not Survive")

# Visualization
st.title("Titanic Data Visualization")

fig, axes = plt.subplots(1, 2, figsize=(12, 5))
sns.countplot(x="Pclass", hue="Survived", data=df, ax=axes[0], palette="coolwarm")
axes[0].set_title("Survival by Class")

sns.histplot(df["Age"], bins=30, kde=True, color="blue", ax=axes[1])
axes[1].set_title("Age Distribution")

st.pyplot(fig)
