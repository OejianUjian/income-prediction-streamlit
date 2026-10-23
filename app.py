import streamlit as st
import pandas as pd
import joblib

st.set_page_config(
    page_title="Income Prediction",
    page_icon="📊",
    layout="centered"
)

@st.cache_resource
def load_files():
    model = joblib.load("model_income_xgboost.pkl")
    prep = joblib.load("preprocessing_income.pkl")
    return model, prep

model, prep = load_files()

st.title("Income Prediction using XGBoost")
st.write(
    "Predict whether annual income is <=50K or >50K "
    "using a machine learning model trained with SMOTE."
)

# Kolom numerik
numeric_values = {}

st.subheader("Personal Information")

for col in prep["numeric_cols"]:
    default = prep["numeric_medians"][col]

    numeric_values[col] = st.number_input(
        col,
        value=float(default),
        step=1.0,
    )

# Gender
gender = st.selectbox(
    "Gender",
    ["Female", "Male"]
)

# Kolom kategorikal
st.subheader("Other Information")

categorical_values = {}

for col, categories in zip(
    prep["nominal_cols"],
    prep["encoder"].categories_
):
    categorical_values[col] = st.selectbox(
        col,
        options=list(categories)
    )

if st.button("Predict Income", type="primary"):

    # Input numerik
    input_data = numeric_values.copy()

    # Encoding gender
    input_data["Gender"] = {
        "Female": 0,
        "Male": 1
    }[gender]

    # DataFrame kategori
    cat_df = pd.DataFrame([categorical_values])

    # Gunakan encoder saat training
    encoded = prep["encoder"].transform(
        cat_df[prep["nominal_cols"]]
    )

    encoded_df = pd.DataFrame(
        encoded,
        columns=prep["encoder"].get_feature_names_out(
            prep["nominal_cols"]
        )
    )

    # Gabungkan fitur
    input_df = pd.concat(
        [pd.DataFrame([input_data]), encoded_df],
        axis=1
    )

    # Samakan urutan fitur dengan training
    input_df = input_df.reindex(
        columns=prep["feature_columns"],
        fill_value=0
    )

    # Prediksi
    prediction = int(model.predict(input_df)[0])
    probability = model.predict_proba(input_df)[0, 1]

    st.subheader("Prediction Result")

    if prediction == 1:
        st.success("Predicted Income: >50K")
    else:
        st.info("Predicted Income: <=50K")

    st.metric(
        "Estimated Probability of Income >50K",
        f"{probability * 100:.2f}%"
    )

    st.caption(
        "This is a statistical prediction, "
        "not a verified income or a decision recommendation."
    )
