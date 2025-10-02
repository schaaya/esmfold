# streamlit_app.py
import re
import time
import json
import numpy as np
import requests
import streamlit as st
import py3Dmol
from stmol import showmol
from biotite.structure.io import pdb as bsio

# ---------------------- Config ----------------------
AF_BASE = "https://alphafold.ebi.ac.uk/files"
AF_PATTERNS = [
    "AF-{uid}-F1-model_v4.pdb",
    "AF-{uid}-F1-model_v3.pdb",
]
ESMFOLD_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"

st.set_page_config(page_title="Protein Structure Viewer", layout="wide")
st.title("🧬 Protein Structure Viewer (AlphaFold DB + ESMFold fallback)")

# ---------------------- Utilities ----------------------
VALID_AA = set("ARNDCQEGHILKMFPSTWYVBZJX")  # allowed tokens

def clean_and_validate_sequence(seq: str) -> str:
    seq = seq.upper()
    # remove whitespace and common invisible unicode chars
    seq = re.sub(r'[\s\u00a0\u200b\u200c\u200d\r\n]+', '', seq)
    if not seq:
        raise ValueError("Sequence is empty.")
    bad = [(i, ch) for i, ch in enumerate(seq) if ch not in VALID_AA]
    if bad:
        i, ch = bad[0]
        allowed = ''.join(sorted(VALID_AA))
        raise ValueError(
            f"Invalid token '{ch}' at position {i+1}. Allowed: {allowed}"
        )
    # optional length guard
    if len(seq) > 4000:
        raise ValueError("Sequence too long (>4000).")
    return seq

def render_mol(pdb_text: str, height: int = 560, width: int = 900):
    view = py3Dmol.view(width=width, height=height)
    view.addModel(pdb_text, "pdb")
    view.setStyle({"cartoon": {"color": "spectrum"}})
    view.setBackgroundColor("white")
    view.zoomTo()
    view.spin(True)
    showmol(view, height=height, width=width)

def load_pdb_safe_from_text(pdb_text: str):
    # Write to temp and parse with Biotite for pLDDT (B-factor)
    with open("tmp_af_or_esm.pdb", "w") as f:
        f.write(pdb_text)
    pdb_file = bsio.PDBFile.read("tmp_af_or_esm.pdb")
    try:
        arr = pdb_file.get_structure(extra_fields=["b_factor"])
    except ValueError:
        # handle files without MODEL numbering
        arr = pdb_file.get_structure(model=None, extra_fields=["b_factor"])

    cats = arr.get_annotation_categories()
    if "b_factor" not in cats:
        arr.add_annotation("b_factor", dtype=float)
        arr.b_factor[:] = 0.0
    else:
        bf = np.asarray(arr.b_factor)
        arr.b_factor[:] = np.nan_to_num(bf, nan=0.0)
        arr.b_factor[:] = bf
    return arr

def get_mean_plddt_from_pdb_text(pdb_text: str):
    try:
        struct = load_pdb_safe_from_text(pdb_text)
        return round(float(np.mean(struct.b_factor)), 2)
    except Exception:
        return None

def fetch_alphafold_pdb(uniprot_id: str) -> str:
    """
    Try v4 then v3 file names from AlphaFold DB.
    Returns PDB text or raises HTTPError if not found.
    """
    sess = requests.Session()
    for pattern in AF_PATTERNS:
        url = f"{AF_BASE}/" + pattern.format(uid=uniprot_id)
        r = sess.get(url, timeout=30)
        if r.status_code == 200 and r.text.strip().startswith("HEADER"):
            return r.text
    # If neither returned a PDB, raise the last attempt
    r.raise_for_status()
    return ""  # unreachable

def esmfold_with_retries(sequence: str, max_tries: int = 5, base_delay: float = 1.5) -> str:
    """
    Call public ESMFold endpoint with JSON payload.
    Retries on 429/5xx with exponential backoff + jitter.
    """
    headers = {"Accept": "text/plain", "Content-Type": "application/json"}
    payload = {"sequence": sequence}
    with requests.Session() as s:
        for attempt in range(1, max_tries + 1):
            try:
                resp = s.post(ESMFOLD_URL, headers=headers, data=json.dumps(payload), timeout=90)
                if resp.status_code == 200 and resp.text.strip().startswith("HEADER"):
                    return resp.text
                if resp.status_code in (429, 500, 502, 503, 504):
                    delay = base_delay * (2 ** (attempt - 1))
                    delay += 0.25 * delay * np.random.rand()
                    st.info(f"ESMFold busy (HTTP {resp.status_code}). Retrying in {delay:.1f}s…")
                    time.sleep(delay)
                    continue
                resp.raise_for_status()
            except requests.RequestException as e:
                if attempt < max_tries:
                    delay = base_delay * (2 ** (attempt - 1))
                    delay += 0.25 * delay * np.random.rand()
                    st.info(f"Network error: {e}. Retrying in {delay:.1f}s…")
                    time.sleep(delay)
                else:
                    raise
    raise RuntimeError("ESMFold: exhausted retries.")

# ---------------------- UI ----------------------
with st.sidebar:
    mode = st.radio("Choose mode", ["AlphaFold DB (by UniProt ID)", "ESMFold (by Sequence)"])
    st.markdown("---")
    st.caption("Tip: AlphaFold DB is fetch-only (no live prediction). Use ESMFold for on-the-fly folding.")

col1, col2 = st.columns([1.2, 1])

if mode == "AlphaFold DB (by UniProt ID)":
    with col1:
        uniprot_id = st.text_input("UniProt ID (e.g., P69905 for human hemoglobin subunit alpha)", value="P69905")
        go = st.button("Fetch from AlphaFold DB")
        if go:
            with st.status("Fetching AlphaFold model…", expanded=False) as status:
                try:
                    pdb_text = fetch_alphafold_pdb(uniprot_id.strip())
                    status.update(label="Downloaded PDB from AlphaFold DB ✅", state="complete")
                    st.subheader(f"AlphaFold structure for {uniprot_id}")
                    render_mol(pdb_text)
                    mean_plddt = get_mean_plddt_from_pdb_text(pdb_text)
                    st.subheader("pLDDT")
                    st.write("AlphaFold stores pLDDT in the PDB **B-factor** column (0–100).")
                    st.info(f"pLDDT (mean): {mean_plddt if mean_plddt is not None else 'unavailable'}")
                    st.download_button("Download PDB", data=pdb_text, file_name=f"{uniprot_id}.pdb", mime="text/plain")
                except requests.HTTPError as e:
                    st.error(f"AlphaFold DB: could not find a PDB for '{uniprot_id}'. ({e})")
                except Exception as e:
                    st.error(f"Error fetching AlphaFold PDB: {e}")

    with col2:
        st.markdown("**Notes**")
        st.markdown(
            "- Use a valid UniProt accession (e.g., `P69905`, `Q5VSL9`, or 10-char modern accessions).\n"
            "- The app tries `model_v4` first, then `model_v3` as a fallback.\n"
            "- Use the ESMFold tab to fold arbitrary sequences."
        )

else:  # ESMFold mode
    with col1:
        default_seq = (
            "MGSSHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSS"
            "IKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNP"
            "SLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVA"
            "WMKRFMDNDTRYSTFACENPNSTRVSDFRTANCSLEDPAANKARKEAELAAATAEQ"
        )
        seq_input = st.text_area("Protein sequence (one-letter codes, no spaces/newlines)", value=default_seq, height=180)
        fold = st.button("Predict with ESMFold")
        if fold:
            try:
                sequence = clean_and_validate_sequence(seq_input)
            except ValueError as e:
                st.error(str(e))
                st.stop()

            with st.status("Predicting with ESMFold…", expanded=False) as status:
                try:
                    pdb_text = esmfold_with_retries(sequence)
                    status.update(label="Prediction complete ✅", state="complete")
                    st.subheader("ESMFold prediction")
                    render_mol(pdb_text)
                    mean_plddt = get_mean_plddt_from_pdb_text(pdb_text)
                    st.subheader("pLDDT")
                    st.write("ESMFold also writes pLDDT to the PDB **B-factor** column (0–100).")
                    st.info(f"pLDDT (mean): {mean_plddt if mean_plddt is not None else 'unavailable'}")
                    st.download_button("Download PDB", data=pdb_text, file_name="ESMFold_predicted.pdb", mime="text/plain")
                except Exception as e:
                    st.error(f"ESMFold error: {e}")
                    st.info("You can still visualize a PDB by uploading it below.")

    with col2:
        st.markdown("**Fallbacks**")
        uploaded = st.file_uploader("Upload a PDB to visualize", type=["pdb"])
        if uploaded is not None:
            pdb_text = uploaded.read().decode("utf-8", errors="ignore")
            st.subheader("Uploaded structure")
            render_mol(pdb_text)
            mean_plddt = get_mean_plddt_from_pdb_text(pdb_text)
            st.subheader("pLDDT")
            st.info(f"pLDDT (mean): {mean_plddt if mean_plddt is not None else 'unavailable'}")
            st.download_button("Download PDB", data=pdb_text, file_name="uploaded.pdb", mime="text/plain")
