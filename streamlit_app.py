# streamlit_app.py

import json
import re
import time

import numpy as np
import requests
import streamlit as st
import py3Dmol
from stmol import showmol
from biotite.structure.io import pdb as bsio

AF_BASE = "https://alphafold.ebi.ac.uk/files"
AF_PATTERNS = ["AF-{uid}-F1-model_v4.pdb", "AF-{uid}-F1-model_v3.pdb"]
ESMFOLD_URL = "https://api.esmatlas.com/foldSequence/v1/pdb/"
VALID_AA = set("ARNDCQEGHILKMFPSTWYVBZJX")

st.set_page_config(page_title="Protein Structure Viewer", layout="wide")
st.title("Protein Structure Viewer")

def clean_seq(x: str) -> str:
    x = re.sub(r'[\s\u00a0\u200b\u200c\u200d\r\n]+', '', x.upper())
    if not x:
        raise ValueError("Empty sequence.")
    bad = [(i, ch) for i, ch in enumerate(x) if ch not in VALID_AA]
    if bad:
        i, ch = bad[0]
        raise ValueError(f"Invalid token '{ch}' at {i+1}.")
    if len(x) > 4000:
        raise ValueError("Sequence too long (>4000).")
    return x

def render(pdb_text: str, h: int = 560, w: int = 900):
    v = py3Dmol.view(width=w, height=h)
    v.addModel(pdb_text, "pdb")
    v.setStyle({"cartoon": {"color": "spectrum"}})
    v.setBackgroundColor("white")
    v.zoomTo()
    v.spin(True)
    showmol(v, height=h, width=w)

def parse_bfactor_mean(pdb_text: str):
    with open("tmp.pdb", "w") as f:
        f.write(pdb_text)
    pf = bsio.PDBFile.read("tmp.pdb")
    try:
        arr = pf.get_structure(extra_fields=["b_factor"])
    except ValueError:
        arr = pf.get_structure(model=None, extra_fields=["b_factor"])
    if "b_factor" not in arr.get_annotation_categories():
        arr.add_annotation("b_factor", dtype=float)
        arr.b_factor[:] = 0.0
    else:
        bf = np.asarray(arr.b_factor)
        arr.b_factor[:] = np.nan_to_num(bf, nan=0.0)
    return round(float(np.mean(arr.b_factor)), 2)

def fetch_af_pdb(uid: str) -> str:
    s = requests.Session()
    for p in AF_PATTERNS:
        url = f"{AF_BASE}/" + p.format(uid=uid)
        r = s.get(url, timeout=30)
        if r.status_code == 200 and r.text.strip().startswith("HEADER"):
            return r.text
    r.raise_for_status()

def esmfold(sequence: str, tries: int = 5, base: float = 1.5) -> str:
    headers = {"Accept": "text/plain", "Content-Type": "application/json"}
    payload = {"sequence": sequence}
    with requests.Session() as s:
        for k in range(1, tries + 1):
            try:
                r = s.post(ESMFOLD_URL, headers=headers, data=json.dumps(payload), timeout=90)
                if r.status_code == 200 and r.text.strip().startswith("HEADER"):
                    return r.text
                if r.status_code in (429, 500, 502, 503, 504):
                    delay = base * (2 ** (k - 1))
                    delay += 0.25 * delay * np.random.rand()
                    time.sleep(delay)
                    continue
                r.raise_for_status()
            except requests.RequestException:
                if k == tries:
                    raise
                delay = base * (2 ** (k - 1))
                delay += 0.25 * delay * np.random.rand()
                time.sleep(delay)
    raise RuntimeError("ESMFold retries exhausted.")

with st.sidebar:
    mode = st.radio("Mode", ["AlphaFold DB (UniProt ID)", "ESMFold (Sequence)"])
    st.markdown("---")

c1, c2 = st.columns([1.2, 1])

if mode == "AlphaFold DB (UniProt ID)":
    with c1:
        uid = st.text_input("UniProt ID", value="P69905")
        if st.button("Fetch"):
            with st.status("Fetching…"):
                try:
                    pdb = fetch_af_pdb(uid.strip())
                    st.success("Done")
                    st.subheader(f"AlphaFold: {uid}")
                    render(pdb)
                    try:
                        mean_plddt = parse_bfactor_mean(pdb)
                        st.subheader("pLDDT")
                        st.info(f"Mean: {mean_plddt}")
                    except Exception:
                        st.subheader("pLDDT")
                        st.info("Unavailable")
                    st.download_button("Download PDB", data=pdb, file_name=f"{uid}.pdb", mime="text/plain")
                except Exception as e:
                    st.error(str(e))
    with c2:
        st.markdown("- Use a valid UniProt accession (e.g., `P69905`).")
        st.markdown("- Tries `model_v4`, then `model_v3`.")

else:
    with c1:
        default_seq = (
            "MGSSHHHHHHSSGLVPRGSHMRGPNPTAASLEASAGPFTVRSFTVSRPSGYGAGTVYYPTNAGGTVGAIAIVPGYTARQSS"
            "IKWWGPRLASHGFVVITIDTNSTLDQPSSRSSQQMAALRQVASLNGTSSSPIYGKVDTARMGVMGWSMGGGGSLISAANNP"
            "SLKAAAPQAPWDSSTNFSSVTVPTLIFACENDSIAPVNSSALPIYDSMSRNAKQFLEINGGSHSCANSGNSNQALIGKKGVA"
            "WMKRFMDNDTRYSTFACENPNSTRVSDFRTANCSLEDPAANKARKEAELAAATAEQ"
        )
        seq = st.text_area("Sequence (one-letter codes)", value=default_seq, height=180)
        if st.button("Predict"):
            try:
                seq = clean_seq(seq)
            except Exception as e:
                st.error(str(e))
                st.stop()
            with st.status("Predicting…"):
                try:
                    pdb = esmfold(seq)
                    st.success("Done")
                    st.subheader("ESMFold")
                    render(pdb)
                    try:
                        mean_plddt = parse_bfactor_mean(pdb)
                        st.subheader("pLDDT")
                        st.info(f"Mean: {mean_plddt}")
                    except Exception:
                        st.subheader("pLDDT")
                        st.info("Unavailable")
                    st.download_button("Download PDB", data=pdb, file_name="ESMFold_predicted.pdb", mime="text/plain")
                except Exception as e:
                    st.error(str(e))
                    st.info("You can upload a PDB below.")
    with c2:
        up = st.file_uploader("Upload PDB", type=["pdb"])
        if up:
            pdb = up.read().decode("utf-8", errors="ignore")
            st.subheader("Uploaded")
            render(pdb)
            try:
                mean_plddt = parse_bfactor_mean(pdb)
                st.subheader("pLDDT")
                st.info(f"Mean: {mean_plddt}")
            except Exception:
                st.subheader("pLDDT")
                st.info("Unavailable")
            st.download_button("Download PDB", data=pdb, file_name="uploaded.pdb", mime="text/plain")
