
"""
Smart Disease Prediction System — patient dashboard.
Run: python -m streamlit run app.py
"""

import json
import os
import sys
from datetime import date

import streamlit as st

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(ROOT, "src"))

import auth
import predict as predict_module

MODEL_PATH = os.path.join(ROOT, "models", "ensemble_model.pkl")

st.set_page_config(
    page_title="CarePoint | Smart Disease Prediction",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@400;500;600;700;800&display=swap');
:root { --ink:#162b3d; --muted:#718096; --teal:#087f8c; --pale:#eef8f8; }
html, body, [class*="css"] { font-family:'DM Sans',sans-serif; }
.stApp { background:linear-gradient(180deg,#f5fbfc 0%,#f7f9fc 42%,#f7f9fc 100%); color:var(--ink); }
.block-container { padding-top:1.8rem; padding-bottom:2.5rem; max-width:1180px; }
h1,h2,h3 { font-family:'Manrope',sans-serif !important; color:#173449 !important; }
.hero { padding:1.55rem 1.7rem; border-radius:22px; background:linear-gradient(120deg,#073b4c,#087f8c 65%,#2ca6a4); color:white; margin-bottom:1.2rem; box-shadow:0 12px 32px #0b52631c; }
.hero h1 { color:white !important; font-size:2rem; margin:0 0 .45rem 0; }
.hero p { color:#e1f5f5; margin:0; font-size:1rem; }
.eyebrow { font-size:.76rem; font-weight:700; text-transform:uppercase; letter-spacing:.12em; color:#a9e7e5; }
.panel { background:white; border:1px solid #e4edf1; border-radius:18px; padding:1.2rem 1.35rem; box-shadow:0 5px 20px #183b4a08; }
.stat { background:white; border:1px solid #e4edf1; border-radius:16px; padding:1rem 1.1rem; }
.stat-label { color:#708392; font-size:.83rem; margin-bottom:.35rem; }
.stat-value { color:#12394a; font-size:1.45rem; font-weight:800; font-family:'Manrope',sans-serif; }
div[data-testid="stSidebar"] { background:#edf6f7; border-right:1px solid #dcebed; }
div.stButton > button[kind="primary"] { background:#087f8c; border:0; border-radius:10px; font-weight:700; padding:.65rem 1rem; }
div.stButton > button { border-radius:10px; }
div[data-baseweb="input"] input, div[data-baseweb="select"] { border-radius:10px; }
.small-muted { color:#6b7f8b; font-size:.86rem; }
.auth-wrap { max-width:760px; margin:1rem auto; }
footer {visibility:hidden;}
</style>
""", unsafe_allow_html=True)

auth.init_db()


@st.cache_resource
def get_bundle():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Model file not found at {MODEL_PATH}. "
            "Run the model training pipeline first."
        )
    return predict_module.load_model_bundle(MODEL_PATH)


def hero(title, subtitle, eyebrow="PATIENT PORTAL"):
    st.markdown(
        f'<div class="hero"><div class="eyebrow">{eyebrow}</div>'
        f'<h1>{title}</h1><p>{subtitle}</p></div>',
        unsafe_allow_html=True,
    )


def logout():
    for key in ("patient", "last_prediction", "page", "nav_page"):
        st.session_state.pop(key, None)
    st.rerun()


def open_symptom_checker():
    # Callback runs before Streamlit recreates the radio widget.
    st.session_state.nav_page = "Symptom checker"


def auth_screen():
    st.markdown('<div class="auth-wrap">', unsafe_allow_html=True)

    st.markdown("""
    <div class="hero">
      <div class="eyebrow">SMART HEALTH • PRIVATE PATIENT ACCESS</div>
      <h1>CarePoint</h1>
      <p>A clearer place to record symptoms and review your prediction history.</p>
    </div>
    """, unsafe_allow_html=True)

    left, right = st.columns([1.08, .92], gap="large")

    with left:
        st.markdown("### Your health dashboard")
        st.write(
            "Sign in to access symptom prediction, your personal history, "
            "and profile details."
        )
        st.markdown("""
        <div class="panel">
          <h3>Designed around your privacy</h3>
          <p>🔐 Individual patient accounts</p>
          <p>🗂️ Personal prediction history</p>
          <p>🩺 Symptom-based prediction tool</p>
          <p class="small-muted">Your history is filtered by your signed-in account.</p>
        </div>
        """, unsafe_allow_html=True)
        st.caption(
            "Educational decision-support demo — not a medical diagnosis."
        )

    with right:
        login_tab, register_tab = st.tabs(["Sign in", "Create account"])

        with login_tab:
            with st.form("login_form"):
                email = st.text_input(
                    "Email address", placeholder="you@example.com"
                )
                password = st.text_input("Password", type="password")
                submit = st.form_submit_button(
                    "Sign in securely",
                    type="primary",
                    use_container_width=True,
                )

            if submit:
                patient = auth.authenticate_patient(email, password)
                if patient:
                    st.session_state.patient = patient
                    st.session_state["nav_page"] = "Dashboard"
                    st.rerun()
                else:
                    st.error("Email or password is incorrect.")

        with register_tab:
            with st.form("register_form", clear_on_submit=False):
                full_name = st.text_input("Full name")
                new_email = st.text_input(
                    "Email address", key="register_email"
                )
                dob = st.date_input(
                    "Date of birth",
                    value=date(2000, 1, 1),
                    min_value=date(1900, 1, 1),
                    max_value=date.today(),
                )
                new_password = st.text_input(
                    "Create password (10+ characters)",
                    type="password",
                    key="register_password",
                )
                confirm = st.text_input(
                    "Confirm password", type="password"
                )
                accepted = st.checkbox(
                    "I understand this is an educational prediction tool."
                )
                create = st.form_submit_button(
                    "Create patient account",
                    type="primary",
                    use_container_width=True,
                )

            if create:
                if new_password != confirm:
                    st.error("The passwords do not match.")
                elif not accepted:
                    st.error("Please confirm the educational-use notice.")
                else:
                    try:
                        auth.register_patient(
                            full_name,
                            new_email,
                            new_password,
                            dob.isoformat(),
                        )
                        st.success(
                            "Account created. Please sign in using your new credentials."
                        )
                    except ValueError as exc:
                        st.error(str(exc))

    st.markdown("</div>", unsafe_allow_html=True)


# Show login and registration until a patient signs in.
if "patient" not in st.session_state:
    auth_screen()
    st.stop()


patient = st.session_state.patient

# Sidebar navigation
with st.sidebar:
    st.markdown("## 🩺 CarePoint")
    st.caption("SMART DISEASE PREDICTION")
    st.divider()
    st.markdown(f"**{patient['full_name']}**")
    st.caption(patient["email"])

    # This is the only widget using the nav_page key.
    page = st.radio(
        "NAVIGATION",
        [
            "Dashboard",
            "Symptom checker",
            "Prediction history",
            "My profile",
        ],
        key="nav_page",
        label_visibility="visible",
    )

    st.divider()
    st.markdown(
        '<p class="small-muted">🔒 Signed-in patient session</p>',
        unsafe_allow_html=True,
    )

    if st.button("Sign out", use_container_width=True):
        logout()


# Dashboard page
if page == "Dashboard":
    hero(
        f"Welcome back, {patient['full_name'].split()[0]}",
        "Your personal symptom prediction space, all in one place.",
    )

    history = auth.get_patient_history(patient["id"])

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(
            f'<div class="stat"><div class="stat-label">Saved predictions</div>'
            f'<div class="stat-value">{len(history)}</div></div>',
            unsafe_allow_html=True,
        )

    with c2:
        st.markdown(
            '<div class="stat"><div class="stat-label">Access</div>'
            '<div class="stat-value">Protected</div></div>',
            unsafe_allow_html=True,
        )

    with c3:
        st.markdown(
            '<div class="stat"><div class="stat-label">Prediction engine</div>'
            '<div class="stat-value">Ensemble ML</div></div>',
            unsafe_allow_html=True,
        )

    st.write("")
    col1, col2 = st.columns([1.2, .8], gap="large")

    with col1:
        st.markdown(
            '<div class="panel"><h3>Start with your symptoms</h3>'
            '<p>Choose symptoms to receive a ranked set of model predictions. '
            'Your results can be saved to your private account history.</p></div>',
            unsafe_allow_html=True,
        )

        # FIX: use a callback instead of changing nav_page after the radio exists.
        st.button(
            "Open symptom checker →",
            type="primary",
            on_click=open_symptom_checker,
        )

    with col2:
        st.markdown(
            '<div class="panel"><h3>Quick privacy note</h3>'
            '<p>Your saved history is associated with your patient account. '
            'Sign out when using a shared computer.</p></div>',
            unsafe_allow_html=True,
        )

    st.subheader("Recent predictions")

    if history:
        for item in history[:5]:
            with st.container(border=True):
                col_a, col_b = st.columns([3, 1])

                col_a.markdown(
                    f"**{(item['primary_condition'] or 'Prediction').replace('_',' ').title()}**"
                )
                col_a.caption(f"Symptoms: {item['symptoms']}")
                col_b.caption(str(item["created_at"]))

                if item["confidence"] is not None:
                    col_b.metric(
                        "Model score",
                        f"{float(item['confidence']) * 100:.1f}%",
                    )
    else:
        st.info(
            "No predictions saved yet. Visit Symptom checker to create your first record."
        )


# Symptom checker page
elif page == "Symptom checker":
    hero(
        "Symptom checker",
        "Select the symptoms you are experiencing to see the model's ranked predictions.",
        "PREDICTION WORKSPACE",
    )

    try:
        bundle = get_bundle()
        feature_names = bundle["feature_names"]
        model_name = bundle.get("final_model_name", "Ensemble model")

        st.caption(f"Prediction engine: {model_name}")

        pretty_to_feature = {
            f.replace("_", " "): f for f in feature_names
        }
        pretty_names = sorted(pretty_to_feature.keys())

        with st.form("prediction_form"):
            selected = st.multiselect(
                "Select symptoms",
                options=pretty_names,
                help="Search by typing part of a symptom name.",
            )
            free_text = st.text_input(
                "Other symptoms (optional)",
                placeholder="e.g. headache, nausea",
            )
            run_prediction = st.form_submit_button(
                "Predict disease", type="primary"
            )

        if run_prediction:
            symptoms = list(selected)

            if free_text.strip():
                symptoms.extend(
                    s.strip()
                    for s in free_text.split(",")
                    if s.strip()
                )

            if not symptoms:
                st.warning("Select or enter at least one symptom first.")
            else:
                result = predict_module.predict_disease(
                    symptoms, bundle, top_k=3
                )

                if result.get("message"):
                    st.error(result["message"])
                else:
                    st.markdown("### Prediction results")
                    col1, col2 = st.columns([1.4, 1])

                    with col1:
                        st.markdown(
                            '<div class="panel">', unsafe_allow_html=True
                        )
                        st.caption("TOP MODEL PREDICTION")
                        st.subheader(
                            result["predicted_disease"].replace("_", " ").title()
                        )
                        st.write(
                            "This is a model-generated estimate, not a confirmed diagnosis."
                        )
                        st.markdown(
                            "</div>", unsafe_allow_html=True
                        )

                    with col2:
                        if result["confidence"] is not None:
                            st.metric(
                                "Model score",
                                f"{result['confidence'] * 100:.1f}%",
                            )

                    st.subheader("Other ranked predictions")

                    for rank, item in enumerate(
                        result["top_k_predictions"], 1
                    ):
                        label = item["disease"].replace("_", " ").title()
                        prob = item.get("probability")
                        score = (
                            f" — {prob * 100:.1f}%"
                            if prob is not None
                            else ""
                        )
                        st.write(f"**{rank}. {label}**{score}")

                    st.caption(
                        "Recognized symptoms: "
                        + ", ".join(result["recognized_symptoms"])
                    )

                    if result["unrecognized_symptoms"]:
                        st.warning(
                            "Not recognized: "
                            + ", ".join(result["unrecognized_symptoms"])
                        )

                    auth.save_prediction(
                        patient["id"],
                        result["recognized_symptoms"],
                        str(result["predicted_disease"]),
                        (
                            float(result["confidence"])
                            if result["confidence"] is not None
                            else None
                        ),
                        json.dumps(result["top_k_predictions"]),
                    )

                    st.success(
                        "This prediction was saved to your account history."
                    )

        st.warning(
            "This tool is for educational/research use only. Do not use it "
            "to diagnose, treat, or rule out a medical condition. Seek a "
            "qualified healthcare professional for medical concerns."
        )

    except Exception as exc:
        st.error(f"Could not load or run the prediction model: {exc}")
        st.info(
            "Confirm that models/ensemble_model.pkl exists and that the project's requirements are installed."
        )


# Prediction history page
elif page == "Prediction history":
    hero(
        "Prediction history",
        "Only records belonging to your signed-in account are shown.",
        "YOUR RECORDS",
    )

    history = auth.get_patient_history(patient["id"])

    if not history:
        st.info(
            "Your history is empty. Make a prediction in Symptom checker to add a record."
        )
    else:
        for item in history:
            with st.container(border=True):
                left, right = st.columns([3, 1])

                left.markdown(
                    f"### {(item['primary_condition'] or 'Prediction').replace('_',' ').title()}"
                )
                left.caption(f"Recorded: {item['created_at']}")
                left.write(f"**Symptoms:** {item['symptoms']}")

                try:
                    top3 = json.loads(item.get("top_3_json") or "[]")

                    if top3:
                        top_results = " · ".join(
                            f"{x['disease'].replace('_',' ').title()} "
                            f"({x['probability'] * 100:.1f}%)"
                            for x in top3
                            if x.get("probability") is not None
                        )
                        left.write("**Top results:** " + top_results)

                except (ValueError, TypeError, KeyError):
                    pass

                if item["confidence"] is not None:
                    right.metric(
                        "Model score",
                        f"{float(item['confidence']) * 100:.1f}%",
                    )


# Patient profile page
elif page == "My profile":
    hero(
        "My profile",
        "Review your account information and optionally record health context.",
        "PERSONAL DETAILS",
    )

    profile = auth.get_patient_profile(patient["id"])

    st.markdown('<div class="panel">', unsafe_allow_html=True)
    st.markdown("### Account")

    c1, c2 = st.columns(2)
    c1.text_input(
        "Full name", value=patient["full_name"], disabled=True
    )
    c2.text_input(
        "Email", value=patient["email"], disabled=True
    )
    st.text_input(
        "Date of birth", value=patient["date_of_birth"], disabled=True
    )
    st.markdown("</div>", unsafe_allow_html=True)

    st.write("")

    gender_options = [
        "Prefer not to say",
        "Female",
        "Male",
        "Non-binary",
        "Other",
    ]
    saved_gender = profile.get("gender") or "Prefer not to say"
    gender_index = (
        gender_options.index(saved_gender)
        if saved_gender in gender_options
        else 0
    )

    with st.form("profile_form"):
        gender = st.selectbox(
            "Gender (optional)",
            gender_options,
            index=gender_index,
        )
        conditions = st.text_area(
            "Existing conditions (optional)",
            value=profile.get("existing_conditions", ""),
        )
        medications = st.text_area(
            "Current medications (optional)",
            value=profile.get("current_medications", ""),
        )
        allergies = st.text_area(
            "Allergies (optional)",
            value=profile.get("allergies", ""),
        )
        save_profile = st.form_submit_button(
            "Save profile", type="primary"
        )

    if save_profile:
        auth.update_patient_profile(
            patient["id"],
            "" if gender == "Prefer not to say" else gender,
            conditions,
            medications,
            allergies,
        )
        st.success("Your profile has been saved.")

    st.caption(
        "Only add information you are comfortable storing in this local demo database."
    )


st.divider()
st.caption(
    "CarePoint · Smart Disease Prediction System · Educational use only, not medical advice."
)
