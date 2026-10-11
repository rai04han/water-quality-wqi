"""
Water Quality (CCME WQI) classifier — demo app.
Run:   streamlit run app.py
Needs in the same folder: China_dataset.csv + the .pkl files (see make_demo_models.py)
"""
import joblib
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import confusion_matrix, f1_score, classification_report

FEATURES = [
    'Ammonia (mg/l)', 'Biochemical Oxygen Demand (mg/l)', 'Dissolved Oxygen (mg/l)',
    'Orthophosphate (mg/l)', 'pH (ph units)', 'Temperature (cel)',
    'Nitrogen (mg/l)', 'Nitrate (mg/l)',
]

st.set_page_config(page_title="Water Quality Classifier", layout="wide")


@st.cache_resource
def load_artifacts():
    return {
        'model': joblib.load('best_wqi_model.pkl'),
        'scaler': joblib.load('scaler.pkl'),
        'le': joblib.load('label_encoder.pkl'),
        'ranges': joblib.load('feature_ranges.pkl'),
    }


@st.cache_data
def load_data():
    d = pd.read_csv('China_dataset.csv')
    d['year'] = pd.to_datetime(d['Date'], format='%d-%m-%Y').dt.year
    return d


@st.cache_data
def holdout_evaluation():
    """Evaluate the 2001-2012-trained model on every 2013-2017 record (never seen in training)."""
    d, le = load_data(), load_artifacts()['le']
    model, scaler = joblib.load('holdout_model.pkl'), joblib.load('holdout_scaler.pkl')
    te = d[d['year'] >= 2013].reset_index(drop=True)
    proba = model.predict_proba(scaler.transform(te[FEATURES].values))
    pred = model.classes_[proba.argmax(axis=1)]
    true = le.transform(te['CCME_WQI'])
    return te, proba, pred, true


try:
    A = load_artifacts()
    df = load_data()
except FileNotFoundError as e:
    st.error(f"Missing file: {e.filename}. Put China_dataset.csv and the .pkl files next to app.py "
             "(run make_demo_models.py if the .pkl files are missing).")
    st.stop()

le, model, scaler, ranges = A['le'], A['model'], A['scaler'], A['ranges']
CLASSES = list(le.classes_)

st.title("Water Quality Classification (CCME WQI)")
st.caption("Random Forest · Hou Bay, China dataset (2001–2017) · enter 8 measurements → predicted WQI class")

tab1, tab2 = st.tabs(["1 · Predict from readings", "2 · Test on unseen data (accuracy evidence)"])

# ───────────────────────────── TAB 1: live prediction ─────────────────────────────
with tab1:
    # Initial values = median of the 'Good' class (so the form starts with realistic numbers)
    start = df[df['CCME_WQI'] == 'Good'][FEATURES].median()
    for f in FEATURES:
        st.session_state.setdefault(f, float(start[f]))

    def apply_preset():
        p = st.session_state['preset']
        if p != 'Custom':
            med = df[df['CCME_WQI'] == p][FEATURES].median()      # typical reading profile of that class
            for f in FEATURES:
                st.session_state[f] = float(med[f])

    st.selectbox("Load a typical profile (median of each class), or choose Custom",
                 ['Custom'] + CLASSES, key='preset', on_change=apply_preset)

    cols = st.columns(4)
    values = []
    for i, f in enumerate(FEATURES):
        values.append(cols[i % 4].number_input(f, key=f, step=0.01, format="%.3f"))

    if st.button("Predict water quality", type="primary", key="predict_btn"):
        sample = np.array([values])
        for f, v, lo, hi in zip(FEATURES, values, ranges.loc['min'], ranges.loc['max']):
            if v < lo or v > hi:
                st.warning(f"{f} = {v} is outside the training range [{lo}, {hi}] — "
                           "the model is extrapolating, treat this prediction with caution.")
        proba = model.predict_proba(scaler.transform(sample))[0]
        order = le.inverse_transform(model.classes_)
        top = int(proba.argmax())
        c1, c2 = st.columns([1, 2])
        c1.metric("Predicted class", order[top])
        c1.metric("Vote share of the winning class", f"{proba[top]:.0%}")
        c2.bar_chart(pd.DataFrame({'probability': proba}, index=order))
        st.info("The percentage is the share of the 100 trees that voted for this class. "
                "It is NOT a guarantee that the prediction is correct — it is an uncalibrated score. "
                "Tab 2 shows how often high-vote predictions are actually right on data the model never saw.")

    st.caption("This model was trained on all 45,997 rows, so typing in a real row from the CSV is a working "
               "check, not an accuracy test. Use tab 2 for that.")

# ───────────────────────────── TAB 2: unseen-data evidence ─────────────────────────────
with tab2:
    try:
        te, proba, pred, true = holdout_evaluation()
    except FileNotFoundError as e:
        st.error(f"Missing {e.filename}. Run:  python make_demo_models.py")
        st.stop()

    st.markdown(
        "**Setup:** a separate Random Forest was trained only on **2001–2012** records. "
        f"Everything below uses the **{len(te):,} records from 2013–2017**, which it has never seen."
    )

    m1, m2, m3 = st.columns(3)
    m1.metric("Macro F1 on unseen years", f"{f1_score(true, pred, average='macro'):.3f}")
    m2.metric("Overall accuracy", f"{(pred == true).mean():.2%}")
    m3.metric("Unseen records", f"{len(te):,}")

    st.subheader("Confusion matrix (rows = actual, columns = predicted)")
    st.dataframe(pd.DataFrame(confusion_matrix(true, pred, labels=range(len(CLASSES))),
                              index=[f"actual {c}" for c in CLASSES],
                              columns=[f"pred {c}" for c in CLASSES]))

    st.subheader("Per-class scores")
    rep = classification_report(true, pred, target_names=CLASSES, output_dict=True, zero_division=0)
    st.dataframe(pd.DataFrame(rep).T.loc[CLASSES + ['macro avg']].round(3))
    st.caption("Marginal has very few unseen records, so its scores are noisy — read the 'support' column.")

    st.subheader("Is a high vote share trustworthy?")
    conf = proba.max(axis=1)
    bins = pd.cut(conf, [0, 0.6, 0.8, 0.9, 0.99, 1.0], include_lowest=True)
    calib = (pd.DataFrame({'bin': bins, 'correct': pred == true})
             .groupby('bin', observed=True)['correct'].agg(['count', 'mean']))
    calib.columns = ['records', 'actually correct']
    calib['actually correct'] = (calib['actually correct'] * 100).round(1).astype(str) + " %"
    st.dataframe(calib)
    st.caption("If predictions with vote share ≈100% are right far more often than low-vote ones, the score is "
               "informative. It still does not mean 'certain' — check the errors in the confusion matrix.")

    st.subheader("Try a random unseen record")
    pick = st.selectbox("Sample from class", ['Any'] + CLASSES, key='pick_class')
    st.session_state.setdefault('tally', [0, 0])
    if st.button("Pick a random unseen record", key="random_btn"):
        pool = np.arange(len(te)) if pick == 'Any' else np.where(te['CCME_WQI'].values == pick)[0]
        i = int(np.random.choice(pool))
        actual, predicted = CLASSES[true[i]], CLASSES[pred[i]]
        st.dataframe(te.loc[[i], ['Date'] + FEATURES])
        ok = actual == predicted
        (st.success if ok else st.error)(
            f"Actual: **{actual}** | Predicted: **{predicted}** | vote share {conf[i]:.0%}")
        st.session_state['tally'][0] += int(ok)
        st.session_state['tally'][1] += 1
    ok_n, tot = st.session_state['tally']
    if tot:
        st.caption(f"Live tally this session: {ok_n} correct out of {tot} random picks "
                   "(small samples — rely on the full-table numbers above).")
