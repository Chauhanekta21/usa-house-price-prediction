from pathlib import Path
import math

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "usa_house_price_prediction_model.pkl"
TRAINING_DATA_PATH = PROJECT_ROOT / "data" / "usa_housing.csv"
FEATURE_NAMES = [
    "Avg. Area Income",
    "Avg. Area House Age",
    "Avg. Area Number of Rooms",
    "Avg. Area Number of Bedrooms",
    "Area Population",
]


@st.cache_data
def get_feature_outlier_bounds() -> dict[str, tuple[float, float]]:
    training_data = pd.read_csv(TRAINING_DATA_PATH)
    quartiles = training_data[FEATURE_NAMES].quantile([0.25, 0.75])
    bounds = {}
    for feature in FEATURE_NAMES:
        lower_quartile = quartiles.loc[0.25, feature]
        upper_quartile = quartiles.loc[0.75, feature]
        interquartile_range = upper_quartile - lower_quartile
        bounds[feature] = (
            float(lower_quartile - 1.5 * interquartile_range),
            float(upper_quartile + 1.5 * interquartile_range),
        )
    return bounds


@st.cache_data
def load_training_data() -> pd.DataFrame:
    return pd.read_csv(TRAINING_DATA_PATH)


st.set_page_config(
    page_title="USA Housing Price Prediction",
    page_icon="⌂",
    layout="wide",
    initial_sidebar_state="expanded",
)


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
            --ivory: #f8f6f0;
            --result-background: #e7eee6;
            --sage: #718675;
            --sage-dark: #526957;
            --charcoal: #28332e;
            --muted: #68736c;
        }
        .stApp {
            background: var(--ivory);
            color: var(--charcoal);
            font-family: "Segoe UI", "Helvetica Neue", sans-serif;
        }
        [data-testid="stSidebar"] {
            background: var(--result-background);
            border-right: 1px solid #e1dacd;
            color: var(--charcoal);
        }
        [data-testid="stSidebar"] [data-testid="stSidebarUserContent"] {
            padding: 1.5rem 1rem;
        }
        [data-testid="stSidebar"] h1,
        [data-testid="stSidebar"] h2,
        [data-testid="stSidebar"] h3,
        [data-testid="stSidebar"] p,
        [data-testid="stSidebar"] label,
        [data-testid="stSidebar"] [data-testid="stCaptionContainer"] {
            color: var(--charcoal) !important;
            font-family: "Segoe UI", "Helvetica Neue", sans-serif;
        }
        .sidebar-brand {
            margin: 0.25rem 0 2rem;
        }
        .sidebar-brand-title {
            color: var(--sage-dark);
            font-size: 1.25rem;
            font-weight: 700;
            letter-spacing: -0.035em;
            line-height: 1.25;
        }
        .sidebar-brand-subtitle {
            color: var(--muted);
            font-size: 0.86rem;
            line-height: 1.45;
            margin-top: 0.35rem;
            max-width: 15rem;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] {
            background: rgba(255, 255, 255, 0.28) !important;
            border: 1px solid color-mix(in srgb, var(--sage-dark) 28%, var(--result-background)) !important;
            border-radius: 9px;
            box-sizing: border-box;
            color: var(--charcoal) !important;
            display: flex;
            align-items: center;
            width: 100% !important;
            min-width: 100% !important;
            max-width: 100% !important;
            height: 2.9rem;
            min-height: 2.9rem;
            margin: 0.18rem 0;
            padding: 0.55rem 0.8rem;
            transition: background-color 140ms ease, border-color 140ms ease;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
            display: none !important;
        }
        #root [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]
        > div:last-child [data-testid="stMarkdownContainer"] p {
            color: var(--charcoal) !important;
            font-size: 0.94rem;
            font-weight: 550;
            line-height: 1.4;
            margin: 0;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:hover {
            background: #e2e8df !important;
            border-color: color-mix(in srgb, var(--sage-dark) 28%, var(--result-background)) !important;
            color: var(--charcoal) !important;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input[type="radio"]:checked) {
            background: rgba(255, 255, 255, 0.28) !important;
            border-color: color-mix(in srgb, var(--sage-dark) 28%, var(--result-background)) !important;
            color: var(--charcoal) !important;
        }
        #root [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input[type="radio"]:checked)
        > div:last-child [data-testid="stMarkdownContainer"] p {
            color: var(--charcoal) !important;
            font-weight: 550;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:focus-within {
            outline: 2px solid var(--sage);
            outline-offset: 2px;
        }
        h1, h2, h3 {
            color: var(--charcoal);
            letter-spacing: -0.025em;
        }
        .stApp [data-testid="stMain"] h1 {
            color: var(--sage-dark);
        }
        .st-key-dataset_overview_heading h3,
        .st-key-dataset_preview_heading h3 {
            color: var(--sage-dark);
        }
        .st-key-dataset_preview_heading {
            margin-top: 1.5rem;
        }
        .st-key-model_performance_heading h3,
        .st-key-model_limitations_heading h3 {
            color: var(--sage-dark);
        }
        .st-key-model_performance_heading,
        .st-key-model_limitations_heading {
            margin-top: 0.5rem;
            padding-top: 0;
        }
        [data-testid="stHorizontalBlock"]:has(.model-metric-card) {
            margin-bottom: 1.5rem;
        }
        .stApp p,
        .stApp label,
        .stApp [data-testid="stCaptionContainer"] {
            color: var(--charcoal);
            font-family: "Segoe UI", "Helvetica Neue", sans-serif;
        }
        .eyebrow {
            color: var(--sage-dark);
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.13em;
            text-transform: uppercase;
            margin-bottom: 0.5rem;
        }
        .intro-card {
            background: var(--result-background);
            border: 1px solid #e4ddcf;
            border-radius: 22px;
            padding: clamp(1.5rem, 5vw, 3.5rem);
            color: var(--charcoal);
        }
        .hero {
            margin: 0.5rem 0 1.5rem;
            padding-block: clamp(1rem, 2.5vw, 1.75rem);
        }
        .hero p, .muted {
            color: var(--muted);
            line-height: 1.7;
        }
        .disclaimer-card {
            margin-top: 1rem;
            padding-block: 1rem;
            line-height: 1.7;
        }
        .feature-card {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid #e7e2d8;
            border-radius: 16px;
            padding: 1.1rem 1.2rem;
            min-height: 112px;
        }
        .feature-card strong {
            color: var(--charcoal);
            display: block;
            margin-bottom: 0.4rem;
        }
        .feature-card span {
            color: var(--muted);
            font-size: 0.92rem;
            line-height: 1.5;
        }
        .dataset-metric {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid #e4e9e0;
            border-radius: 14px;
            min-height: 88px;
            padding: 1rem 1.1rem;
        }
        .dataset-metric-label {
            color: var(--muted);
            font-size: 0.82rem;
            line-height: 1.4;
        }
        .dataset-metric-value {
            color: var(--charcoal);
            font-size: 1.25rem;
            font-weight: 700;
            line-height: 1.35;
            margin-top: 0.35rem;
        }
        .model-metric-card,
        .limitation-card {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid #e4e9e0;
            border-radius: 14px;
            box-sizing: border-box;
            height: 11rem;
            padding: 1rem 1.1rem;
        }
        .model-metric-card {
            padding-bottom: 0.8rem;
        }
        .limitation-card {
            height: 8rem;
            padding-bottom: 0.5rem;
        }
        .model-metric-label {
            color: var(--muted);
            font-size: 0.84rem;
            font-weight: 600;
            line-height: 1.4;
            min-height: 2.4em;
        }
        .model-metric-value {
            color: var(--charcoal);
            font-size: 1.3rem;
            font-weight: 700;
            line-height: 1.35;
            margin: 0.35rem 0 0.5rem;
        }
        .model-metric-description,
        .limitation-card p {
            color: var(--muted);
            font-size: 0.9rem;
            line-height: 1.55;
            margin: 0;
        }
        .model-metric-card .model-metric-description {
            color: var(--muted);
            font-size: 0.9rem;
        }
        .limitation-card strong {
            color: var(--charcoal);
            display: block;
            font-size: 0.95rem;
            margin-bottom: 0.35rem;
        }
        [data-testid="stForm"] {
            background: rgba(255, 255, 255, 0.72);
            border: 1px solid #e7e2d8;
            border-radius: 18px;
            padding: 1.4rem;
        }
        div.stButton > button, div.stFormSubmitButton > button {
            background: var(--sage-dark);
            color: #fff;
            border: 1px solid var(--sage-dark);
            border-radius: 10px;
            min-height: 2.8rem;
            font-weight: 650;
            transition: background-color 120ms ease;
        }
        div.stButton > button p, div.stFormSubmitButton > button p {
            color: #fff !important;
            font-weight: 650;
        }
        div.stButton > button:hover, div.stFormSubmitButton > button:hover {
            background: var(--charcoal);
            border-color: var(--charcoal);
            color: #fff;
        }
        div.stButton > button:hover p, div.stFormSubmitButton > button:hover p {
            color: #fff !important;
        }
        .result-card {
            background: var(--result-background);
            border: 1px solid #d5e0d4;
            border-radius: 18px;
            padding: 1.5rem 1.7rem;
            margin-top: 1.2rem;
        }
        .result-label {
            color: var(--sage-dark);
            font-size: 0.82rem;
            font-weight: 700;
            letter-spacing: 0.1em;
            text-transform: uppercase;
        }
        .result-price {
            color: var(--charcoal);
            font-size: clamp(2rem, 5vw, 3rem);
            font-weight: 750;
            letter-spacing: -0.04em;
            margin: 0.35rem 0;
        }
        .st-key-negative_prediction_warning [data-testid="stAlert"] {
            background: #fff1cc !important;
            border: 1px solid #ead9ad !important;
            border-radius: 12px !important;
            color: #1f1708 !important;
            color-scheme: light;
        }
        .st-key-negative_prediction_warning [data-testid="stAlert"] * {
            background-color: transparent !important;
            color: #1f1708 !important;
            -webkit-text-fill-color: #1f1708 !important;
            opacity: 1 !important;
        }
        .st-key-negative_prediction_warning [data-testid="stAlert"] svg {
            color: #704600 !important;
            fill: #704600 !important;
            -webkit-text-fill-color: #704600 !important;
        }
        @media (max-width: 640px) {
            [data-testid="stForm"] { padding: 1rem; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)


def show_home() -> None:
    st.markdown(
        """
        <section class="intro-card hero">
            <h1>USA House Price Predictor</h1>
            <p>
                This project uses Linear Regression to predict house prices in the USA.
                Predictions are based on five features: Average Area Income, Average
                Area House Age, Average Number of Rooms, Average Number of Bedrooms,
                and Area Population. Users can enter these details on the Price
                Predictor page to receive a predicted house price.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    with st.container(key="dataset_overview_heading"):
        st.subheader("Dataset Overview")
    metrics = [
        ("Total Records", "5,000"),
        ("Input Features", "5"),
        ("Prediction Target", "House Price"),
        ("Dataset Location", "USA"),
    ]
    for column, (label, value) in zip(st.columns(4), metrics):
        with column:
            st.markdown(
                f'<div class="dataset-metric"><div class="dataset-metric-label">{label}</div>'
                f'<div class="dataset-metric-value">{value}</div></div>',
                unsafe_allow_html=True,
            )
    with st.container(key="dataset_preview_heading"):
        st.subheader("Dataset Preview")
    st.dataframe(
        load_training_data().head(25),
        height=520,
        width="stretch",
        hide_index=True,
    )


def show_prediction(model) -> None:
    st.markdown('<div class="eyebrow">Property details</div>', unsafe_allow_html=True)
    st.title("House Price Prediction")
    st.markdown(
        '<p class="muted">Enter the average values for the area. Bedrooms can be fractional '
        "because the model uses area-level averages.</p>",
        unsafe_allow_html=True,
    )

    with st.form("prediction_form"):
        first, second = st.columns(2)
        with first:
            income = st.number_input(
                "Average Area Income (USD)",
                min_value=0.0,
                value=68_804.0,
                step=1_000.0,
                help="Average income for residents in the area.",
            )
            house_age = st.number_input(
                "Average Area House Age (years)",
                min_value=0.0,
                value=5.97,
                step=0.1,
                help="Average age of houses in the area.",
            )
            rooms = st.number_input(
                "Average Number of Rooms",
                min_value=0.0,
                value=7.0,
                step=0.1,
                help="Average number of rooms per house.",
            )
        with second:
            bedrooms = st.number_input(
                "Average Number of Bedrooms",
                min_value=0.0,
                value=4.05,
                step=0.1,
                help="Average number of bedrooms per house; fractional averages are accepted.",
            )
            population = st.number_input(
                "Area Population",
                min_value=0.0,
                value=36_199.0,
                step=500.0,
                help="Population of the surrounding area.",
            )

        submitted = st.form_submit_button("Predict Price", use_container_width=True)

    if not submitted:
        return

    values = [income, house_age, rooms, bedrooms, population]
    if not all(math.isfinite(value) and value > 0 for value in values):
        st.error("Enter a finite value greater than zero for each field.")
        return

    input_data = pd.DataFrame(
        [[income, house_age, rooms, bedrooms, population]],
        columns=FEATURE_NAMES,
    )
    predicted_price = float(model.predict(input_data)[0])
    if not math.isfinite(predicted_price):
        st.error("The model could not produce a valid estimate for these inputs.")
        return
    if predicted_price < 0:
        outlier_bounds = get_feature_outlier_bounds()
        unusual_values = []
        for feature, value in zip(FEATURE_NAMES, values):
            lower_bound, upper_bound = outlier_bounds[feature]
            if value < lower_bound:
                unusual_values.append(
                    f"- **{feature}**: {value:,.2f} (unusually low; below "
                    f"{lower_bound:,.2f})"
                )
            elif value > upper_bound:
                unusual_values.append(
                    f"- **{feature}**: {value:,.2f} (unusually high; above "
                    f"{upper_bound:,.2f})"
                )

        warning = (
            "This input combination is outside the model's reliable prediction "
            "range. Please try different input values."
        )
        if unusual_values:
            warning += (
                "\n\nUnusual values based on the training dataset's 1.5×IQR "
                "bounds:\n\n"
                + "\n".join(unusual_values)
            )
        else:
            warning += (
                "\n\nNo individual input is unusually low or high based on the "
                "training dataset's 1.5×IQR bounds; the combination itself may "
                "still be outside the model's reliable prediction range."
            )
        with st.container(key="negative_prediction_warning"):
            st.warning(warning)
        return

    st.markdown(
        f"""
        <div class="result-card" role="status">
            <div class="result-label">Estimated house price</div>
            <div class="result-price">${predicted_price:,.2f}</div>
            <div class="muted">Estimate shown in US dollars. This is not a property appraisal.</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def show_about() -> None:
    st.markdown(
        """
        <section class="intro-card hero">
            <h1>About the Model</h1>
            <p>
                This project uses a previously trained Linear Regression model to estimate
                house prices in the USA based on five property and area-related details.
                The model was trained and evaluated separately in the project's Jupyter
                Notebook. The trained model is saved in the `models` folder and loaded by
                the application whenever a user requests a prediction.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    with st.container(key="model_performance_heading"):
        st.subheader("Model Performance")
    metrics = [
        (
            "Prediction Accuracy (R² Score)",
            "91.80%",
            "Explains approximately 91.8% of price variation in the test data.",
        ),
        (
            "Average Prediction Error (MAE)",
            "$80,879.10",
            "Predictions differ from actual prices by about $80,879 on average.",
        ),
        (
            "Typical Prediction Error (RMSE)",
            "$100,444.06",
            "Gives more weight to larger prediction errors.",
        ),
        (
            "Training Score",
            "91.80%",
            "Performance on the data used to train the model.",
        ),
    ]
    for column, (label, value, description) in zip(st.columns(4), metrics):
        with column:
            st.markdown(
                f'<div class="model-metric-card">'
                f'<div class="model-metric-label">{label}</div>'
                f'<div class="model-metric-value">{value}</div>'
                f'<p class="model-metric-description">{description}</p>'
                f"</div>",
                unsafe_allow_html=True,
            )

    with st.container(key="model_limitations_heading"):
        st.subheader("Model Limitations")
    limitation_columns = st.columns(2)
    limitations = [
        (
            "Unusual Inputs",
            "Unusual combinations of otherwise valid inputs can sometimes produce unrealistic predictions.",
        ),
        (
            "Negative Predictions",
            "The model may predict a negative house price, which is not realistic and should be treated as an unreliable estimate.",
        ),
    ]
    for column, (title, description) in zip(limitation_columns, limitations):
        with column:
            st.markdown(
                f'<div class="limitation-card"><strong>{title}</strong>'
                f"<p>{description}</p></div>",
                unsafe_allow_html=True,
            )

    st.markdown(
        """
        <div class="intro-card disclaimer-card">
            This project is for educational exploration. Its estimates should not be used as financial advice or a substitute for a professional appraisal.
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    apply_styles()

    with st.sidebar:
        st.markdown(
            """
            <div class="sidebar-brand">
                <div class="sidebar-brand-title">House Price Prediction</div>
                <div class="sidebar-brand-subtitle">
                    Predicting house prices using a Linear Regression model.
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )
        page = st.radio(
            "Navigate",
            [
                ":material/dashboard: Overview",
                ":material/query_stats: Price Predictor",
                ":material/info: About the Model",
            ],
            label_visibility="collapsed",
        )

    if page.endswith("Overview"):
        show_home()
        return
    if page.endswith("About the Model"):
        show_about()
        return

    if not MODEL_PATH.is_file():
        st.error(
            "The trained model file was not found. Expected it at "
            f"`{MODEL_PATH}`. Make sure the project’s `models` folder contains "
            "`usa_house_price_prediction_model.pkl`."
        )
        st.stop()

    model = load_model()
    model_features = list(getattr(model, "feature_names_in_", []))
    if model_features != FEATURE_NAMES:
        st.error(
            "The loaded model's input feature names or order do not match this "
            "app. Check that the expected trained model is in the `models` folder."
        )
        st.stop()
    show_prediction(model)


if __name__ == "__main__":
    main()
