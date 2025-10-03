# --- Knikkerbaan Experiment Demo (NL) - wide layout + cached tree + live path + rerun after retrain ---
import streamlit as st
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor, plot_tree

st.set_page_config(page_title="Knikkerbaan Demo")

# ---------------------------
# Const & helpers
# ---------------------------
KENMERKEN = ['recht','gebogen','pijpje','knikker_encoded']

def encode_knikker(lbl: str) -> int:
    return 1 if lbl == "A" else 0

def lineaire_formule_tex(model_lm, feature_names):
    feature_names = ["type\ knikker" if i == "knikker_encoded" else i for i in feature_names]

    coef = getattr(model_lm, "coef_", None)
    intercept = getattr(model_lm, "intercept_", None)
    if coef is None or intercept is None:
        return r"\text{Lineair model nog niet getraind.}"
    termen = [f"{coef[i]:.3f}\\cdot\\mathrm{{{feature_names[i]}}}" for i in range(len(feature_names))]
    rhs = " + ".join(termen) if termen else "0"
    return r"tijd\ (s) = " + f"{intercept:.3f} + " + rhs

def decision_path_text(model_dt, x_row, feature_names):
    tree = model_dt.tree_
    node_indicator = model_dt.decision_path(x_row)
    leaf_id = model_dt.apply(x_row)[0]
    feature = tree.feature
    threshold = tree.threshold
    children_left = tree.children_left
    children_right = tree.children_right

    start = node_indicator.indptr[0]
    end = node_indicator.indptr[1]
    node_index = node_indicator.indices[start:end]

    regels = []
    for node_id in node_index:
        if node_id == leaf_id:
            leaf_pred = tree.value[node_id][0][0]
            regels.append(f"⮑ Blad {node_id}: voorspelling = {leaf_pred:.3f} s")
            break
        feat_id = feature[node_id]
        fname = feature_names[feat_id]
        thresh = threshold[node_id]
        val = x_row[0, feat_id]
        teken = "<=" if val <= thresh else ">"
        next_id = children_left[node_id] if val <= thresh else children_right[node_id]
        regels.append(f"Node {node_id}: {fname} ({val:.3f}) {teken} {thresh:.3f} → naar node {next_id}")
    return "\n".join(regels)

def werk_modellen_bij(model_lm, model_rf, model_dt, data, p1, p2, p3, p4, gemeten_tijd):
    X_nieuw = pd.DataFrame([[p1, p2, p3, encode_knikker(p4)]], columns=KENMERKEN)
    y_nieuw = pd.Series([gemeten_tijd], name="tijd")
    data = pd.concat([data, pd.concat([X_nieuw, y_nieuw], axis=1)], ignore_index=True)

    X = data[KENMERKEN]; y = data['tijd']
    model_lm.fit(X, y); model_rf.fit(X, y); model_dt.fit(X, y)

    joblib.dump(model_lm, 'marble_run_model_lm.pkl')
    joblib.dump(model_rf, 'marble_run_model_rf.pkl')
    joblib.dump(model_dt, 'marble_run_model_dt.pkl')
    data[KENMERKEN + ['tijd']].to_csv('marble_run_data.csv', index=False)
    return model_lm, model_rf, model_dt, data

def laad_of_init_state():
    if "model_version" not in st.session_state:
        st.session_state.model_version = 0

    if "data" not in st.session_state:
        pad_data = Path('marble_run_data.csv')
        st.session_state.data = pd.read_csv(pad_data) if pad_data.exists() else pd.DataFrame(columns=KENMERKEN + ['tijd'])

    if "model_lm" not in st.session_state or "model_rf" not in st.session_state or "model_dt" not in st.session_state:
        # maak basis
        lm = LinearRegression()
        rf = RandomForestRegressor(n_estimators=200, random_state=42)
        dt = DecisionTreeRegressor(max_depth=4, random_state=42)

        # initial fit als er wel data is
        if len(st.session_state.data) >= 2 and not Path('marble_run_model_lm.pkl').exists():
            X = st.session_state.data[KENMERKEN]; y = st.session_state.data['tijd']
            lm.fit(X, y); rf.fit(X, y); dt.fit(X, y)

        st.session_state.model_lm = lm
        st.session_state.model_rf = rf
        st.session_state.model_dt = dt

# Cache ONLY the rendered figure; invalidate when model_version changes
@st.cache_data(show_spinner=False)
def maak_beslisboom_fig(version: int, feature_names):
    model = st.session_state.model_dt
    fig, ax = plt.subplots(figsize=(16, 10))
    plot_tree(model, feature_names=feature_names, filled=True, rounded=True, fontsize=10)
    return fig

# ---------------------------
# App
# ---------------------------
st.title("Knikkerbaan Experiment Demo")
laad_of_init_state()

# Invoer
p1 = st.slider("Aantal **rechte** bruggen", min_value=0, max_value=10, step=1)
p2 = st.slider("Aantal **gebogen** bruggen", min_value=0, max_value=10, step=1)
p3 = st.slider("Aantal **kleine pijpjes**", min_value=0, max_value=10, step=1)
knikker_label = st.selectbox("Type knikker", ('Groot', 'Klein'))
p4 = 'A' if knikker_label == 'Groot' else 'B'

# Voorspellingen (gebaseerd op huidige state)
if not st.session_state.data.empty:
    x_row = np.array([[p1, p2, p3, encode_knikker(p4)]], dtype=float)
    y_lm = st.session_state.model_lm.predict(x_row)[0]
    y_rf = st.session_state.model_rf.predict(x_row)[0]
    y_dt = st.session_state.model_dt.predict(x_row)[0]
    st.write(f"**Voorspelde tijd (lineair model):** {y_lm:.2f} s")
    st.write(f"**Voorspelde tijd (beslisboom):** {y_dt:.2f} s")
    st.write(f"**Voorspelde tijd (random forest):** {y_rf:.2f} s")
else:
    st.warning("Nog geen data. Voeg een meting toe en klik **Model bijwerken**.")
    x_row = np.array([[p1, p2, p3, encode_knikker(p4)]], dtype=float)

# Nieuwe meting
st.subheader("Nieuwe meting toevoegen")
gemeten_tijd = st.number_input("Werkelijke tijd (seconden)", min_value=0.0, step=0.01, format="%.2f")

# Retrain + immediate refresh of predictions and cached tree
if st.button("Model bijwerken"):
    if not gemeten_tijd or gemeten_tijd == 0.0:
        st.warning("Vul een geldige gemeten tijd in.")
    else:
        lm, rf, dt, data = werk_modellen_bij(
            st.session_state.model_lm, st.session_state.model_rf, st.session_state.model_dt,
            st.session_state.data, p1, p2, p3, p4, gemeten_tijd
        )
        st.session_state.model_lm = lm
        st.session_state.model_rf = rf
        st.session_state.model_dt = dt
        st.session_state.data = data
        st.session_state.model_version += 1  # invalideer cached figuur
        st.success("Modellen bijgewerkt en opgeslagen!")
        st.rerun()  # <-- direct opnieuw draaien zodat voorspellingen & pad syncen

# Formule + boom
st.header("Formule lineaire regressie")
st.latex(lineaire_formule_tex(st.session_state.model_lm, KENMERKEN))

st.header("Beslisboom (modelstructuur)")
if hasattr(st.session_state.model_dt, "tree_") and st.session_state.model_dt.tree_.node_count > 0:
    fig_tree = maak_beslisboom_fig(st.session_state.model_version, KENMERKEN)
    st.pyplot(fig_tree, use_container_width=True)

    st.subheader("Pad voor huidige invoer")
    uitleg = decision_path_text(st.session_state.model_dt, x_row, KENMERKEN)
    st.code(uitleg, language="text")

    # Check: leaf-voorspelling moet gelijk zijn aan y_dt
    leaf_id = st.session_state.model_dt.apply(x_row)[0]
    leaf_pred = st.session_state.model_dt.tree_.value[leaf_id][0][0]
    st.caption(f"Controle: leaf {leaf_id} voorspelling = {leaf_pred:.3f} s.")
else:
    st.info("De beslisboom is nog niet getraind.")
