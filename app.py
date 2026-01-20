import streamlit as st
import pandas as pd
import pickle
import numpy as np
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem import AllChem
from rdkit.Chem import Descriptors
from fpdf import FPDF
import tempfile
import os

# --- 1. SETUP THE PAGE ---
st.set_page_config(page_title="Drug Discovery AI", page_icon="💊", layout="wide")

st.title("💊 AI Drug Researcher")
st.markdown("""
This app uses **Artificial Intelligence** (Random Forest) to predict if a molecule 
could be an effective drug against **EGFR (Lung Cancer)**.
""")

# --- 2. LOAD THE SAVED MODEL ---
@st.cache_resource
def load_model():
    with open('drug_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model

try:
    model = load_model()
except FileNotFoundError:
    st.error("⚠️ Model file not found! Please make sure 'drug_model.pkl' is in the same folder.")
    st.stop()

# --- 3. SIDEBAR INPUT ---
st.sidebar.header("Input Molecule")
smiles_input = st.sidebar.text_area("Paste SMILES string here:", 
                                    value="COc1cc2ncnc(Nc3ccc(F)c(Cl)c3)c2cc1OCCCN4CCOCC4",
                                    height=150)

# --- 4. HELPER FUNCTIONS ---
def get_fingerprint(smiles):
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol:
            return AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        return None
    except:
        return None

def get_properties(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol:
        return {
            "Molecular Weight": Descriptors.MolWt(mol),
            "LogP (Solubility)": Descriptors.MolLogP(mol),
            "H-Bond Donors": Descriptors.NumHDonors(mol),
            "H-Bond Acceptors": Descriptors.NumHAcceptors(mol),
            "Rotatable Bonds": Descriptors.NumRotatableBonds(mol)
        }
    return None

def create_pdf(smiles, result_text, confidence, props):
    """Generates a PDF Report with the molecule image"""
    
    # 1. Setup PDF
    pdf = FPDF()
    pdf.add_page()
    pdf.set_font("Arial", size=12)
    
    # 2. Header
    pdf.set_font("Arial", 'B', 16)
    pdf.cell(200, 10, txt="AI Drug Discovery Report", ln=True, align='C')
    pdf.set_font("Arial", size=10)
    pdf.cell(200, 10, txt=f"Generated on: {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M')}", ln=True, align='C')
    pdf.ln(10)
    
    # 3. Prediction Section
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="1. Prediction Results", ln=True)
    pdf.set_font("Arial", size=12)
    
    # Color code the result text
    result_color = "Promising" if "ACTIVE" in result_text else "Not Promising"
    pdf.cell(200, 10, txt=f"Prediction: {result_text} ({result_color})", ln=True)
    pdf.cell(200, 10, txt=f"AI Confidence: {confidence*100:.1f}%", ln=True)
    pdf.ln(5)

    # 4. Chemical Properties Section
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="2. Chemical Properties", ln=True)
    pdf.set_font("Arial", size=12)
    for key, value in props.items():
        pdf.cell(200, 10, txt=f"{key}: {value:.2f}", ln=True)
    pdf.ln(5)
    
    # 5. Molecule Image
    pdf.set_font("Arial", 'B', 14)
    pdf.cell(200, 10, txt="3. Molecular Structure", ln=True)
    
    # Generate temporary image file
    mol = Chem.MolFromSmiles(smiles)
    img = Draw.MolToImage(mol, size=(400, 400))
    
    # Save to a temporary path so PDF can read it
    with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmpfile:
        img.save(tmpfile.name)
        pdf.image(tmpfile.name, x=10, y=None, w=100)
    
    # 6. SMILES String (Footer)
    pdf.ln(10)
    pdf.set_font("Arial", 'I', 8)
    pdf.multi_cell(0, 5, txt=f"SMILES Code: {smiles}")

    # Return PDF as bytes
    return pdf.output(dest='S').encode('latin-1')

# --- 5. PREDICTION LOGIC ---
if st.sidebar.button("Analyze Molecule", type="primary"):
    if not smiles_input:
        st.warning("Please enter a SMILES string.")
    else:
        fp = get_fingerprint(smiles_input)
        
        if fp is None:
            st.error("Invalid SMILES string.")
        else:
            # Predict
            fp_numpy = np.array(fp).reshape(1, -1)
            prediction = model.predict(fp_numpy)[0]
            probability = model.predict_proba(fp_numpy)[0]
            props = get_properties(smiles_input)
            
            # --- DISPLAY RESULTS ---
            st.divider()
            col1, col2, col3 = st.columns([1, 1, 1])
            
            with col1:
                st.subheader("Structure")
                mol = Chem.MolFromSmiles(smiles_input)
                st.image(Draw.MolToImage(mol), caption="Chemical Structure")

            with col2:
                st.subheader("Properties")
                st.dataframe(pd.DataFrame(props, index=["Value"]).T, use_container_width=True)

            with col3:
                st.subheader("AI Prediction")
                if prediction == 1:
                    result_text = "ACTIVE"
                    st.success(f"**{result_text}**")
                    confidence = probability[1]
                else:
                    result_text = "INACTIVE"
                    st.error(f"**{result_text}**")
                    confidence = probability[0]
                st.metric("Confidence", f"{confidence*100:.1f}%")
            
            # --- PDF GENERATION ---
            st.divider()
            
            # Generate the PDF bytes
            pdf_bytes = create_pdf(smiles_input, result_text, confidence, props)
            
            # Download Button
            st.download_button(
                label="📄 Download Report (PDF)",
                data=pdf_bytes,
                file_name="drug_analysis_report.pdf",
                mime="application/pdf"
            )