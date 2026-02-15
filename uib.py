import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from ocr import preprocess_image, extract_text_from_image, parse_receipt_text, categorize_items, analyze_spending, generate_budget_advice
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="AI Receipt Analyzer", page_icon="💳", layout="wide")

st.title("💳 AI-Powered Receipt Analyzer")
st.markdown("Upload a receipt image and get spending analysis with personalized budget advice.")

# Upload receipt image
uploaded_file = st.file_uploader("Choose a receipt image", type=["jpg", "jpeg", "png"])

if uploaded_file:
    # Save the uploaded image temporarily
    temp_path = Path("temp_receipt.jpg")
    with open(temp_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    # ---------------------------
    # Step 0: Preprocessing visualization
    # ---------------------------
    st.subheader("🖼️ Preprocessing Steps Visualization")

    # Load image as OpenCV
    image = np.array(Image.open(temp_path).convert("RGB"))

    # Step 0a: Original
    st.markdown("**Original Image**")
    st.image(image, width=700)

    # Step 0b: Grayscale
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    st.markdown("**Grayscale**")
    st.image(gray, width=700, clamp=True)

    # Step 0c: Denoising (Gaussian Blur)
    denoised = cv2.GaussianBlur(gray, (5, 5), 0)
    st.markdown("**Denoised**")
    st.image(denoised, width=700, clamp=True)

    # Step 0d: Thresholding
    _, thresh = cv2.threshold(denoised, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    st.markdown("**Thresholded**")
    st.image(thresh, width=700, clamp=True)

    # ---------------------------
    # Step 1: Preprocess using your existing function (for OCR consistency)
    # ---------------------------
    processed_path = preprocess_image(str(temp_path))

    # ---------------------------
    # Step 2: OCR
    # ---------------------------
    with st.spinner("Extracting text from receipt..."):
        ocr_text = extract_text_from_image(processed_path)
    st.subheader("📄 OCR Text")
    st.text_area("Detected Text", ocr_text, height=200)

    # ---------------------------
    # Step 3: Parse
    # ---------------------------
    structured_items = parse_receipt_text(ocr_text)
    st.subheader("📝 Structured Items")
    df_items = pd.DataFrame(structured_items)
    st.dataframe(df_items)

    # ---------------------------
    # Step 4: Categorize
    # ---------------------------
    categorized_data = categorize_items(structured_items)
    categorized_data["overall_total"] = sum(categorized_data["category_totals"].values())

    st.subheader("📊 Spending by Category")
    df_categories = pd.DataFrame(categorized_data["categorized_items"])
    st.dataframe(df_categories)

    # Pie chart
    # Pie chart (smaller size, no overlapping)
    st.subheader("📈 Spending Distribution")

    # Create smaller figure
    fig, ax = plt.subplots(figsize=(4, 4))  # reduced size

    labels = [cat.capitalize() for cat in categorized_data["category_totals"].keys()]
    values = [val for val in categorized_data["category_totals"].values()]

    def autopct_func(pct):
        return ('%1.1f%%' % pct) if pct > 5 else ''  # hide <5%

    # Draw pie chart
    wedges, texts, autotexts = ax.pie(
    values,
    labels=None,          # hide labels inside
    autopct=autopct_func, # custom autopct function
    startangle=90,
    shadow=True
)

    # Add legend outside
    ax.legend(
      wedges, labels, title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1)
    )
    # Add a legend outside the pie
    #ax.legend(wedges, labels, title="Categories", loc="center left", bbox_to_anchor=(1, 0, 0.5, 1))

    ax.axis("equal")
    st.pyplot(fig)

    # ---------------------------
    # Step 5: Spending Analysis & Advice
    # ---------------------------
    spending_analysis = analyze_spending(categorized_data)
    analysis_df = pd.DataFrame([
        {"Category": cat.capitalize(), 
         "Total ($)": data["total"], 
         "Percentage (%)": data["percent"], 
         "Overspending": "⚠️" if data["overspend"] else ""}
        for cat, data in spending_analysis.items()
    ])
    st.subheader("💡 Spending Analysis")
    st.dataframe(analysis_df)

    st.subheader("💡 Personalized Budget Advice")
    with st.spinner("Generating advice..."):
        advice_text = generate_budget_advice(categorized_data)
    st.text_area("Advice", advice_text, height=300)
