from pathlib import Path
import math

import joblib
import pandas as pd
import streamlit as st


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = PROJECT_ROOT / "models" / "usa_house_price_prediction_model.pkl"
FEATURE_NAMES = [
    "Avg. Area Income",
    "Avg. Area House Age",
    "Avg. Area Number of Rooms",
    "Avg. Area Number of Bedrooms",
    "Area Population",
]


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
            --beige: #eee8dc;
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
            background: var(--beige);
            border-right: 1px solid #e1dacd;
            color: var(--charcoal);
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
        [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] {
            background: transparent;
            border: 1px solid transparent;
            border-radius: 10px;
            color: var(--charcoal) !important;
            padding: 0.45rem 0.65rem;
            transition: background-color 120ms ease, border-color 120ms ease;
        }
        #root [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]
        > div:last-child [data-testid="stMarkdownContainer"] p {
            color: #28332e !important;
            font-size: 0.96rem;
            font-weight: 550;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child {
            background: transparent !important;
            border: 2px solid var(--sage) !important;
            border-radius: 50%;
            box-sizing: border-box;
            width: 0.55rem !important;
            height: 0.55rem !important;
            min-width: 0.55rem !important;
            min-height: 0.55rem !important;
            flex: 0 0 0.55rem;
            margin-right: 0.2rem;
            align-items: center;
            justify-content: center;
            transition: none !important;
            transform: none !important;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"] > div:first-child > div:first-child {
            width: 0.18rem !important;
            height: 0.18rem !important;
            min-width: 0.18rem !important;
            min-height: 0.18rem !important;
            border-radius: 50%;
            background: #28332e !important;
            border-color: #28332e !important;
            transition: none !important;
            transform: none !important;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input[type="radio"]:checked)
        > div:first-child > div:first-child {
            background: #fff !important;
            border-color: #fff !important;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:hover > div:first-child,
        [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:hover > div:first-child > div:first-child {
            transition: none !important;
            transform: none !important;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:hover {
            background: #e2e8df;
            border-color: #d3ddcf;
            color: var(--charcoal) !important;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input[type="radio"]:checked) {
            background: var(--sage-dark);
            border-color: var(--sage-dark);
            color: #fff !important;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input[type="radio"]:checked)
        > div:first-child {
            background: var(--sage-dark) !important;
            border-color: var(--sage-dark) !important;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input[type="radio"]:checked)
        > div:first-child > div:first-child {
            background: #fff !important;
            border-color: #fff !important;
        }
        #root [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:has(input[type="radio"]:checked)
        > div:last-child [data-testid="stMarkdownContainer"] p {
            color: #fff !important;
            font-weight: 650;
        }
        [data-testid="stSidebar"] [data-testid="stRadio"] label[data-baseweb="radio"]:focus-within {
            outline: 3px solid #879b88;
            outline-offset: 2px;
        }
        h1, h2, h3 {
            color: var(--charcoal);
            letter-spacing: -0.025em;
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
            background: var(--beige);
            border: 1px solid #e4ddcf;
            border-radius: 22px;
            padding: clamp(1.5rem, 5vw, 3.5rem);
            color: var(--charcoal);
        }
        .hero {
            margin: 0.5rem 0 1.5rem;
        }
        .hero p, .muted {
            color: var(--muted);
            line-height: 1.7;
        }
        .disclaimer-card {
            margin-top: 1.5rem;
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
            background: #e7eee6;
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
            <div class="eyebrow">A data-informed home estimate</div>
            <h1>USA Housing Price Prediction</h1>
            <p>
                Explore an estimated house price from a few neighborhood and
                property characteristics. Enter the five model inputs in the
                prediction section to get started.
            </p>
        </section>
        """,
        unsafe_allow_html=True,
    )
    st.subheader("A simple way to explore an estimate")
    st.markdown(
        '<p class="muted">Use the navigation on the left to enter property details or learn about the model. '
        "The result is a model estimate for educational use, not a professional appraisal.</p>",
        unsafe_allow_html=True,
    )
    columns = st.columns(3)
    cards = [
        ("Five inputs", "Provide neighborhood income, house age, rooms, bedrooms, and population."),
        ("Instant estimate", "Get a price estimate in US dollars from the saved model."),
        ("Transparent approach", "See how the trained linear regression model is used."),
    ]
    for column, (title, description) in zip(columns, cards):
        with column:
            st.markdown(
                f'<div class="feature-card"><strong>{title}</strong><span>{description}</span></div>',
                unsafe_allow_html=True,
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
    st.markdown('<div class="eyebrow">How it works</div>', unsafe_allow_html=True)
    st.title("About the Model")
    st.markdown(
        """
        This application uses a previously trained **Linear Regression** model.
        The model is loaded from the project’s `models` folder; it is not trained
        again when the app runs. Model training and evaluation were carried out
        separately in the project’s Jupyter Notebook.
        """
    )
    st.subheader("Prediction inputs")
    for feature in [
        "Average Area Income",
        "Average Area House Age",
        "Average Number of Rooms",
        "Average Number of Bedrooms",
        "Area Population",
    ]:
        st.markdown(f"- {feature}")
    st.markdown(
        """
        <div class="intro-card disclaimer-card">
            This project is for educational exploration. Its estimates should not
            be used as financial advice or a substitute for a professional appraisal.
        </div>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    apply_styles()

    with st.sidebar:
        st.markdown("## ⌂ Homewise")
        st.caption("USA Housing Price Prediction")
        page = st.radio(
            "Navigate",
            ["Home", "Predict", "About the Model"],
            label_visibility="collapsed",
        )
        st.divider()

    if page == "Home":
        show_home()
        return
    if page == "About the Model":
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
