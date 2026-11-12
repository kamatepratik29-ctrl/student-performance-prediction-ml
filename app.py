from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

st.set_page_config(page_title="Student Performance Predictor", page_icon="🎓", layout="wide")
st.title("🎓 Student Performance Prediction")
st.caption("Explore a sample dataset, train a machine learning model, and estimate a sample final exam score.")

DATA_PATH = Path(__file__).parent / "data" / "student_performance.csv"
FEATURES = ["study_hours_per_day", "attendance_percentage", "previous_exam_score", "assignment_average", "sleep_hours_per_day"]
TARGET = "final_exam_score"

@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH).drop_duplicates()
    cols = FEATURES + [TARGET]
    df[cols] = df[cols].apply(pd.to_numeric, errors="coerce")
    return df.dropna(subset=cols)

@st.cache_resource
def train_model():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=0.20, random_state=42
    )
    model = RandomForestRegressor(
        n_estimators=200, max_depth=8, min_samples_leaf=2, random_state=42
    )
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    return model, y_test.reset_index(drop=True), pd.Series(predictions), len(X_train), len(X_test)

df = load_data()
model, actual, predicted, train_n, test_n = train_model()

with st.sidebar:
    st.header("About this project")
    st.write("**Algorithm:** Random Forest Regressor")
    st.write("**Dataset:** 500 illustrative synthetic records")
    st.warning("Educational demonstration only. Do not use predictions for real academic decisions.")

overview, analysis, predict, evaluation = st.tabs(
    ["Overview", "Data Analysis", "Predict Score", "Model Evaluation"]
)

with overview:
    a, b, c = st.columns(3)
    a.metric("Student records", len(df))
    b.metric("Input features", len(FEATURES))
    c.metric("Missing values after cleaning", int(df.isna().sum().sum()))
    st.subheader("Sample dataset")
    st.dataframe(df.head(10), use_container_width=True)
    st.write("Inputs: study hours, attendance, previous exam score, assignment average, and sleep hours.")

with analysis:
    st.subheader("Explore the sample data")
    left, right = st.columns(2)
    with left:
        fig, ax = plt.subplots()
        ax.hist(df[TARGET], bins=18, edgecolor="black")
        ax.set(title="Distribution of Final Exam Scores", xlabel="Final exam score", ylabel="Number of records")
        st.pyplot(fig)
        plt.close(fig)
    with right:
        fig, ax = plt.subplots()
        ax.scatter(df["study_hours_per_day"], df[TARGET], alpha=.6)
        ax.set(title="Study Hours vs Final Exam Score", xlabel="Study hours per day", ylabel="Final exam score")
        st.pyplot(fig)
        plt.close(fig)
    st.subheader("Feature correlation")
    corr = df[FEATURES + [TARGET]].corr()
    fig, ax = plt.subplots(figsize=(9, 5))
    im = ax.imshow(corr, vmin=-1, vmax=1, aspect="auto")
    ax.set_xticks(range(len(corr.columns)), labels=corr.columns, rotation=45, ha="right")
    ax.set_yticks(range(len(corr.columns)), labels=corr.columns)
    ax.set_title("Feature Correlation Matrix")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)

with predict:
    st.subheader("Estimate a sample final exam score")
    st.caption("Enter values, then select Predict score.")
    with st.form("prediction_form"):
        c1, c2 = st.columns(2)
        with c1:
            study = st.slider("Study hours per day", .5, 8.0, 3.0, .5)
            attendance = st.slider("Attendance (%)", 55, 100, 85)
            previous = st.slider("Previous exam score", 35, 100, 70)
        with c2:
            assignment = st.slider("Assignment average", 40, 100, 75)
            sleep = st.slider("Sleep hours per day", 4.0, 9.0, 7.0, .5)
        submitted = st.form_submit_button("Predict score", type="primary")
    if submitted:
        sample = pd.DataFrame([{
            "study_hours_per_day": study,
            "attendance_percentage": attendance,
            "previous_exam_score": previous,
            "assignment_average": assignment,
            "sleep_hours_per_day": sleep,
        }])
        score = float(model.predict(sample)[0])
        st.success(f"Estimated final exam score: {max(0, min(100, score)):.1f} / 100")
        st.caption("This estimate comes from synthetic training data. It is not a guarantee or real academic assessment.")

with evaluation:
    st.subheader("Model evaluation")
    mae = mean_absolute_error(actual, predicted)
    r2 = r2_score(actual, predicted)
    a, b, c = st.columns(3)
    a.metric("Training records", train_n)
    b.metric("Testing records", test_n)
    c.metric("Mean Absolute Error", f"{mae:.2f}")
    st.metric("R² score", f"{r2:.3f}")
    st.caption("MAE is the average absolute prediction error (lower is better). R² measures explained variation.")
    comparison = pd.DataFrame({"Actual score": actual, "Predicted score": predicted})
    st.dataframe(comparison.head(15), use_container_width=True)
    fig, ax = plt.subplots()
    ax.scatter(actual, predicted, alpha=.7)
    ax.plot([0, 100], [0, 100], linestyle="--")
    ax.set(xlim=(0, 100), ylim=(0, 100), xlabel="Actual score", ylabel="Predicted score", title="Actual vs Predicted")
    st.pyplot(fig)
    plt.close(fig)

st.divider()
st.caption("Final AI & ML Project | Python · Pandas · Scikit-learn · Streamlit · Matplotlib")
