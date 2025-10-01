import streamlit as st
from stmol import showmol
import py3Dmol
import requests
import numpy as np
from biotite.structure.io import pdb as bsio

st.sidebar.title('🎈 ESMFold')
st.sidebar.write(
    '[*ESMFold*](https://esmatlas.com/about) predicts single-sequence protein structures using the ESM-2 language model.'
)

def render_mol(pdb_text: str):
    view = py3Dmol.view()
    view.addModel(pdb_text, 'pdb')
    view.setStyle({'cartoon': {'color': 'spectrum'}})
    view.setBackgroundColor('white')
    view.zoomTo()
    view.zoom(2, 800)
    view.spin(True)
    showmol(view, height=500, width=800)

# --- robust loader for ESMFold PDBs ---
def load_pdb_safe(path: str):
    pdb_file = bsio.PDBFile.read(path)
    try:
        # Try default (model=1)
        arr = pdb_file.get_structure(extra_fields=["b_factor"])
    except ValueError:
        # If MODEL records are absent/odd, read all models
        arr = pdb_file.get_structure(model=None, extra_fields=["b_factor"])

    # Ensure numeric b_factor exists (ESMFold uses B-factor for pLDDT)
    cats = arr.get_annotation_categories()
    if "b_factor" not in cats:
        arr.add_annotation("b_factor", dtype=float)
        arr.b_factor[:] = 0.0
    else:
        bf = np.asarray(arr.b_factor)
        arr.b_factor[:] = np.nan_to_num(bf, nan=0.0)

    return arr

DEFAULT_SEQ = (
    "MGSSHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSS"
    "IKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNP"
    "SLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVA"
    "WMKRFMDNDTRYSTFACENPNSTRVSDFRTANCSLEDPAANKARKEAELAAATAEQ"
)

txt = st.sidebar.text_area('Input sequence', DEFAULT_SEQ, height=275)

def update(sequence: str = txt):
    try:
        # ESMFold API call
        resp = requests.post(
            'https://api.esmatlas.com/foldSequence/v1/pdb/',
            headers={'Content-Type': 'application/x-www-form-urlencoded'},
            data=sequence,
            timeout=60,
        )
        resp.raise_for_status()
    except Exception as e:
        st.error(f"Prediction failed: {e}")
        return

    pdb_string = resp.content.decode('utf-8')

    # Persist to file (so Biotite can parse)
    with open('predicted.pdb', 'w') as f:
        f.write(pdb_string)

    # Load PDB safely and compute mean pLDDT (B-factor)
    try:
        struct = load_pdb_safe('predicted.pdb')
        b_value = round(float(np.mean(struct.b_factor)), 4)
    except Exception as e:
        # Fall back if Biotite ever chokes; still show the structure
        st.warning(f"Could not parse pLDDT from PDB (showing structure anyway). Details: {e}")
        b_value = None

    # Visualization
    st.subheader('Visualization of predicted protein structure')
    render_mol(pdb_string)

    # pLDDT panel
    st.subheader('plDDT')
    st.write('plDDT is a per-residue confidence (0–100). In ESMFold PDBs it is stored in the B-factor column.')
    if b_value is not None:
        st.info(f'plDDT (mean): {b_value}')
    else:
        st.info('plDDT (mean): unavailable')

    # Download
    st.download_button(
        label="Download PDB",
        data=pdb_string,
        file_name='predicted.pdb',
        mime='text/plain',
    )

# Trigger
predict = st.sidebar.button('Predict', on_click=update)

if not predict:
    st.warning('👈 Enter protein sequence data and click **Predict**!')
