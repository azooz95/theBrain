import torch
import streamlit as st
import pandas as pd
from PIL import Image
import io

from utils.paper_detection import PaperDetector, PaperDetectionMethodType
from main import ExtractorPipline

torch.classes.__path__ = [] 
# Streamlit App Title
st.title("Invoice Data Extractor")

# File Upload
uploaded_file = st.file_uploader("Upload Invoice (Image/PDF)", type=["png", "jpg", "jpeg", "pdf"])

# Text Input for Extraction Query
extraction_query = st.text_input("Enter the field to extract (e.g., 'Total Amount')")

if uploaded_file:
    # Read Image
    if uploaded_file.type in ["image/png", "image/jpeg", "image/jpg"]:
        image = Image.open(uploaded_file)
        page = ExtractorPipline.DetectPage.method_class(PaperDetectionMethodType(2))(img=image)
        conetents,_,_ = ExtractorPipline.OCR.method_class().extract(page, score_threshold=0.5)
        enhanced_img = ExtractorPipline.RemoveBackground.method_class().run(page,0.7)
        conetents = ' '.join(conetents)
        extract_json = ExtractorPipline.GEMINIApi.method_class(contents=conetents, wanted_information=extraction_query)
        st.image(enhanced_img, caption="Uploaded Image", use_column_width=True)


    # Read PDF (basic method - for advanced, use PyMuPDF or pdfplumber)
    elif uploaded_file.type == "application/pdf":
        pass 
    
    print(extract_json,extraction_query)
    extaracted_text = extract_json
    # Display Extracted Text
    st.subheader("Extracted Text")
    st.text_area("", extract_json, height=200)
    
    # Simulating Extraction Logic
    if extraction_query:
        result = []
        for line in extaracted_text.split("\n"):
            if extraction_query.lower() in line.lower():
                result.append([extraction_query, line])
        
        if result:
            df = pd.DataFrame(result, columns=["Field", "Extracted Value"])
            st.subheader("Extracted Information")
            st.table(df)
        else:
            st.warning("No matching information found.")
