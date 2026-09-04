import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow import keras

st.set_page_config(page_title="AI Jobs Salary Predictor", layout="wide", page_icon="💼")

# ---------------------------------------------------------------
# Load artifacts (cached so they only load once per session)
# ---------------------------------------------------------------
@st.cache_resource
def load_artifacts():
    reg_ann = keras.models.load_model('models/salary_regression_ann.keras')
    clf_ann = keras.models.load_model('models/salary_classification_ann.keras')
    rf_reg = joblib.load('models/rf_regressor.joblib')
    rf_clf = joblib.load('models/rf_classifier.joblib')
    feature_scaler = joblib.load('models/feature_scaler.joblib')
    target_scaler = joblib.load('models/target_scaler.joblib')
    label_encoders = joblib.load('models/label_encoders.joblib')
    band_encoder = joblib.load('models/band_encoder.joblib')
    metadata = joblib.load('models/metadata.joblib')
    return reg_ann, clf_ann, rf_reg, rf_clf, feature_scaler, target_scaler, label_encoders, band_encoder, metadata

@st.cache_data
def load_data():
    return pd.read_csv('models/cleaned_data.csv')

reg_ann, clf_ann, rf_reg, rf_clf, feature_scaler, target_scaler, label_encoders, band_encoder, metadata = load_artifacts()
df = load_data()

FEATURE_COLS = metadata['feature_columns']
METRICS = metadata['model_metrics']

# ---------------------------------------------------------------
# Sidebar navigation
# ---------------------------------------------------------------
st.sidebar.title("💼 AI Jobs Salary Predictor")
page = st.sidebar.radio("Navigate", ["🔮 Predict Salary", "📊 Dataset Explorer", "🧠 Model Performance"])

# =================================================================
# PAGE 1: PREDICTION
# =================================================================
if page == "🔮 Predict Salary":
    st.title("🔮 Predict a Job's Salary")
    st.write("Fill in the job details below to get an ANN-powered salary prediction and salary band classification.")

    col1, col2, col3 = st.columns(3)

    with col1:
        work_year = st.selectbox("Work Year", metadata['work_year_options'], index=len(metadata['work_year_options']) - 1)
        experience_level = st.selectbox(
            "Experience Level", metadata['experience_level_options'],
            format_func=lambda x: {'EN': 'Entry-level', 'MI': 'Mid-level', 'SE': 'Senior-level', 'EX': 'Executive-level'}.get(x, x)
        )
        employment_type = st.selectbox(
            "Employment Type", metadata['employment_type_options'],
            format_func=lambda x: {'FT': 'Full-time', 'PT': 'Part-time', 'CT': 'Contract', 'FL': 'Freelance'}.get(x, x)
        )

    with col2:
        remote_ratio = st.select_slider("Remote Ratio (%)", options=[0, 50, 100], value=100)
        company_size = st.selectbox(
            "Company Size", metadata['company_size_options'],
            format_func=lambda x: {'S': 'Small', 'M': 'Medium', 'L': 'Large'}.get(x, x)
        )
        work_mode = st.selectbox("Work Mode", metadata['work_mode_options'])

    with col3:
        role_family = st.selectbox("Role Family", metadata['role_family_options'])
        job_title_grouped = st.selectbox("Job Title", metadata['job_title_options'])
        company_location_grouped = st.selectbox("Company Location", metadata['company_location_options'])
        employee_residence_grouped = st.selectbox("Employee Residence", metadata['employee_residence_options'])

    st.divider()

    if st.button("Predict Salary", type="primary", use_container_width=True):
        raw_input = {
            'work_year': work_year,
            'experience_level': experience_level,
            'employment_type': employment_type,
            'remote_ratio': remote_ratio,
            'company_size': company_size,
            'work_mode': work_mode,
            'role_family': role_family,
            'job_title_grouped': job_title_grouped,
            'company_location_grouped': company_location_grouped,
            'employee_residence_grouped': employee_residence_grouped,
        }

        # Encode categoricals using the saved label encoders
        encoded = {}
        for col in FEATURE_COLS:
            if col in label_encoders:
                encoded[col] = label_encoders[col].transform([raw_input[col]])[0]
            else:
                encoded[col] = raw_input[col]

        input_df = pd.DataFrame([encoded])[FEATURE_COLS]
        input_scaled = feature_scaler.transform(input_df)

        # --- Regression prediction ---
        ann_pred_scaled = reg_ann.predict(input_scaled, verbose=0).flatten()
        ann_salary = target_scaler.inverse_transform(ann_pred_scaled.reshape(-1, 1)).flatten()[0]
        rf_salary = rf_reg.predict(input_df)[0]

        # --- Classification prediction ---
        ann_band_probs = clf_ann.predict(input_scaled, verbose=0)[0]
        ann_band = band_encoder.inverse_transform([ann_band_probs.argmax()])[0]
        rf_band = rf_clf.predict(input_df)[0]
        rf_band_label = band_encoder.inverse_transform([rf_band])[0]

        st.subheader("Results")
        r1, r2 = st.columns(2)

        with r1:
            st.metric("ANN Predicted Salary", f"${ann_salary:,.0f}")
            st.metric("Random Forest Predicted Salary", f"${rf_salary:,.0f}")

        with r2:
            st.metric("ANN Predicted Salary Band", ann_band)
            st.metric("Random Forest Predicted Salary Band", rf_band_label)

        st.write("**Salary band probabilities (ANN):**")
        prob_df = pd.DataFrame({
            'Band': band_encoder.classes_,
            'Probability': ann_band_probs
        }).sort_values('Probability', ascending=False)

        fig, ax = plt.subplots(figsize=(8, 3))
        sns.barplot(x='Probability', y='Band', data=prob_df, palette='viridis', ax=ax)
        ax.set_xlim(0, 1)
        st.pyplot(fig)

        st.caption(
            "Note: predictions reflect patterns in historical job-posting data and are estimates, "
            "not guarantees — real salaries depend on many factors not captured here (specific employer, "
            "negotiation, cost of living, etc.)."
        )

# =================================================================
# PAGE 2: DATASET EXPLORER
# =================================================================
elif page == "📊 Dataset Explorer":
    st.title("📊 Dataset Explorer")
    st.write(f"Cleaned dataset: **{df.shape[0]:,} rows** × **{df.shape[1]} columns**")

    tab1, tab2, tab3, tab4 = st.tabs(["Overview", "By Experience & Role", "By Location & Remote", "Trends Over Time"])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.histplot(df['salary_in_usd'], bins=50, kde=True, ax=ax, color='steelblue')
            ax.set_title('Salary Distribution (USD)')
            st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(x='salary_band', y='salary_in_usd', data=df,
                        order=['Low', 'Mid', 'High', 'Executive'], ax=ax, palette='viridis')
            ax.set_title('Salary by Band')
            st.pyplot(fig)

        st.dataframe(df.head(50), use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(8, 5))
            order = df.groupby('experience_level_label')['salary_in_usd'].median().sort_values().index
            sns.boxplot(x='experience_level_label', y='salary_in_usd', data=df, order=order, palette='coolwarm', ax=ax)
            ax.set_title('Salary by Experience Level')
            plt.setp(ax.get_xticklabels(), rotation=20)
            st.pyplot(fig)
        with c2:
            fig, ax = plt.subplots(figsize=(8, 5))
            role_order = df.groupby('role_family')['salary_in_usd'].median().sort_values(ascending=False).index
            sns.barplot(x='salary_in_usd', y='role_family', data=df, order=role_order, estimator=np.median, palette='mako', ax=ax)
            ax.set_title('Median Salary by Role Family')
            st.pyplot(fig)

    with tab3:
        c1, c2 = st.columns(2)
        with c1:
            fig, ax = plt.subplots(figsize=(8, 5))
            sns.boxplot(x='work_mode', y='salary_in_usd', data=df, palette='Set2', ax=ax)
            ax.set_title('Salary by Work Mode')
            st.pyplot(fig)
        with c2:
            top_countries = df['company_location'].value_counts().nlargest(10).index
            country_data = df[df['company_location'].isin(top_countries)]
            fig, ax = plt.subplots(figsize=(8, 5))
            order = country_data.groupby('company_location')['salary_in_usd'].median().sort_values(ascending=False).index
            sns.boxplot(x='company_location', y='salary_in_usd', data=country_data, order=order, palette='Spectral', ax=ax)
            ax.set_title('Salary — Top 10 Company Locations')
            st.pyplot(fig)

    with tab4:
        fig, ax = plt.subplots(figsize=(10, 5))
        yearly = df.groupby('work_year')['salary_in_usd'].median()
        sns.lineplot(x=yearly.index, y=yearly.values, marker='o', linewidth=2.5, color='darkorange', ax=ax)
        ax.set_title('Median Salary Trend by Year')
        st.pyplot(fig)

# =================================================================
# PAGE 3: MODEL PERFORMANCE
# =================================================================
elif page == "🧠 Model Performance":
    st.title("🧠 Model Performance Comparison")

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Regression (Salary Prediction)")
        st.metric("Random Forest R²", f"{METRICS['rf_r2']:.3f}")
        st.metric("ANN R²", f"{METRICS['ann_r2']:.3f}")
        st.metric("Random Forest MAE", f"${METRICS['rf_mae']:,.0f}")
        st.metric("ANN MAE", f"${METRICS['ann_mae']:,.0f}")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(['Random Forest', 'ANN'], [METRICS['rf_r2'], METRICS['ann_r2']], color=['#4C72B0', '#DD8452'])
        ax.set_title('R² Comparison')
        ax.set_ylabel('R² Score')
        st.pyplot(fig)

    with c2:
        st.subheader("Classification (Salary Band)")
        st.metric("Random Forest Accuracy", f"{METRICS['rf_acc']:.3f}")
        st.metric("ANN Accuracy", f"{METRICS['ann_acc']:.3f}")
        st.caption("Random baseline for 4 classes ≈ 25%")

        fig, ax = plt.subplots(figsize=(6, 4))
        ax.bar(['Random Forest', 'ANN'], [METRICS['rf_acc'], METRICS['ann_acc']], color=['#4C72B0', '#DD8452'])
        ax.set_title('Accuracy Comparison')
        ax.set_ylabel('Accuracy')
        st.pyplot(fig)

    st.info(
        "Both models perform similarly, which is expected: salary is only partially explained by "
        "the categorical job-posting features available (experience, role, location, remote status). "
        "The remaining variance comes from factors not present in this dataset, such as the specific "
        "employer, individual negotiation, and cost-of-living adjustments."
    )
