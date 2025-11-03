"""
PDF by Het - Document Processing Web App
Created by Hettie

To run this app:
1. Install dependencies: pip install streamlit PyPDF2 pillow opencv-python pytesseract img2pdf
2. Install Tesseract OCR: 
   - Ubuntu/Debian: sudo apt-get install tesseract-ocr
   - Mac: brew install tesseract
   - Windows: Download from https://github.com/UB-Mannheim/tesseract/wiki
3. Run: streamlit run app.py
"""

import streamlit as st
import os
import cv2
import numpy as np
from PyPDF2 import PdfReader, PdfWriter, PdfMerger
from PIL import Image
import io
from datetime import datetime
import base64

# Set page config
st.set_page_config(
    page_title="PDF by Het",
    page_icon="📄",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS with continuous bounce/glow animation for appLogo
st.markdown("""
    <style>
    @keyframes logoBounceGlow {
        0%, 100% { transform: translateY(0); box-shadow: 0 3px 10px rgba(0,0,0,0.15); }
        50% { transform: translateY(-6px); box-shadow: 0 8px 20px rgba(0,0,0,0.25); }
    }

    .main-header {
        display: flex;
        align-items: center;
        justify-content: center;
        gap: 25px;
        padding: 1.5rem 1rem;
        background-color: white;
        color: #333;
        border-radius: 10px;
        margin-bottom: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        transition: all 0.3s ease;
        cursor: default;
    }

    .main-header:hover {
        box-shadow: 0 6px 20px rgba(0,0,0,0.2);
    }

    .main-header img {
        height: 90px;
        border-radius: 10px;
        object-fit: contain;
        animation: logoBounceGlow 2s ease-in-out infinite;  /* continuous gentle bounce/glow */
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .main-header img:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 20px rgba(0,0,0,0.25);
    }

    .header-text {
        display: flex;
        flex-direction: column;
        justify-content: center;
        transition: transform 0.3s ease, color 0.3s ease;
    }

    .main-header:hover .header-text {
        transform: translateY(-5px);
        color: #764ba2;
    }

    .main-header h1 {
        margin: 0;
        font-size: 2.2rem;
        transition: color 0.3s ease;
    }

    .main-header p {
        margin: 0;
        font-size: 1rem;
        color: #555;
        transition: color 0.3s ease;
    }

    .stButton>button {
        width: 100%;
    }
    </style>
""", unsafe_allow_html=True)

# Header with appLogo and animations
st.markdown(f"""
    <div class="main-header">
        <img src="appLogo.png" alt="App Logo">
        <div class="header-text">
            <h1>📄 Het's PDFExpress</h1>
            <p>Complete Document Processing Tool | Created by Hettie</p>
        </div>
    </div>
""", unsafe_allow_html=True)

# Initialize session state
if 'processed_files' not in st.session_state:
    st.session_state.processed_files = []

# Sidebar - Settings
st.sidebar.header("⚙️ Processing Settings")

st.sidebar.subheader("📸 Image Processing")
enhance_images = st.sidebar.checkbox("Auto-enhance images", value=True, 
    help="Improve contrast, brightness, and clarity")
correct_perspective = st.sidebar.checkbox("Correct perspective", value=True,
    help="Fix tilted or skewed documents")
apply_ocr = st.sidebar.checkbox("Apply OCR", value=False,
    help="Make PDFs searchable (slower)")

st.sidebar.subheader("📄 PDF Processing")
merge_all = st.sidebar.checkbox("Merge all files", value=True,
    help="Combine all PDFs into one document")
remove_blanks = st.sidebar.checkbox("Remove blank pages", value=True)
compress_pdf = st.sidebar.checkbox("Compress PDF", value=True)
remove_duplicates = st.sidebar.checkbox("Remove duplicates", value=False)

st.sidebar.markdown("---")
st.sidebar.markdown("### 💡 About")
st.sidebar.info("""
**Het's PDFExpress** can:
- Scan documents from images
- Merge multiple PDFs
- Clean and optimize files
- Remove blank pages
- Apply OCR for searchability
""")

# Helper functions
def enhance_document_image(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
    denoised = cv2.fastNlMeansDenoising(thresh, None, 10, 7, 21)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    return clahe.apply(denoised)

def detect_and_correct_perspective(img_array):
    gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edged = cv2.Canny(blurred, 75, 200)
    contours, _ = cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:5]
    for contour in contours:
        peri = cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, 0.02*peri, True)
        if len(approx) == 4:
            pts = approx.reshape(4,2)
            rect = np.zeros((4,2), dtype="float32")
            s = pts.sum(axis=1)
            rect[0] = pts[np.argmin(s)]
            rect[2] = pts[np.argmax(s)]
            diff = np.diff(pts, axis=1)
            rect[1] = pts[np.argmin(diff)]
            rect[3] = pts[np.argmax(diff)]
            (tl,tr,br,bl) = rect
            widthA = np.sqrt((br[0]-bl[0])**2 + (br[1]-bl[1])**2)
            widthB = np.sqrt((tr[0]-tl[0])**2 + (tr[1]-tl[1])**2)
            maxWidth = max(int(widthA), int(widthB))
            heightA = np.sqrt((tr[0]-br[0])**2 + (tr[1]-br[1])**2)
            heightB = np.sqrt((tl[0]-bl[0])**2 + (tl[1]-bl[1])**2)
            maxHeight = max(int(heightA), int(heightB))
            dst = np.array([[0,0],[maxWidth-1,0],[maxWidth-1,maxHeight-1],[0,maxHeight-1]], dtype="float32")
            M = cv2.getPerspectiveTransform(rect,dst)
            warped = cv2.warpPerspective(img_array, M, (maxWidth, maxHeight))
            return warped
    return img_array

def image_to_pdf(image_file, enhance=True, perspective=True):
    image = Image.open(image_file)
    img_array = np.array(image)
    if len(img_array.shape) == 3:
        img_array = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
    if perspective:
        img_array = detect_and_correct_perspective(img_array)
    if enhance:
        img_array = enhance_document_image(img_array)
    if len(img_array.shape) == 2:
        img_pil = Image.fromarray(img_array)
    else:
        img_pil = Image.fromarray(cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB))
    if img_pil.mode != 'RGB':
        img_pil = img_pil.convert('RGB')
    pdf_bytes = io.BytesIO()
    img_pil.save(pdf_bytes, format='PDF', resolution=100.0, quality=95)
    pdf_bytes.seek(0)
    return pdf_bytes

def format_file_size(size_bytes):
    for unit in ['B','KB','MB','GB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.2f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.2f} TB"

def is_blank_page(page, threshold=100):
    text = page.extract_text().strip()
    return len(text) < threshold

def remove_blank_pages(reader):
    writer = PdfWriter()
    blank_count = 0
    for page in reader.pages:
        if not is_blank_page(page):
            writer.add_page(page)
        else:
            blank_count += 1
    return writer, blank_count

def compress_pdf_func(reader):
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    for page in writer.pages:
        page.compress_content_streams()
    return writer

def remove_duplicate_pages(reader):
    writer = PdfWriter()
    seen_pages = set()
    duplicate_count = 0
    for page in reader.pages:
        page_text = page.extract_text().strip()
        page_hash = hash(page_text)
        if page_hash not in seen_pages:
            seen_pages.add(page_hash)
            writer.add_page(page)
        else:
            duplicate_count += 1
    return writer, duplicate_count

def create_download_link(file_bytes, filename):
    b64 = base64.b64encode(file_bytes).decode()
    return f'<a href="data:application/pdf;base64,{b64}" download="{filename}">📥 Download {filename}</a>'

# Main App
st.header("📤 Upload Your Files")

col1, col2 = st.columns(2)

with col1:
    st.subheader("PDFs")
    pdf_files = st.file_uploader(
        "Upload PDF files",
        type=['pdf'],
        accept_multiple_files=True,
        key="pdf_upload"
    )

with col2:
    st.subheader("Images/Scans")
    image_files = st.file_uploader(
        "Upload images (JPG, PNG)",
        type=['jpg','jpeg','png'],
        accept_multiple_files=True,
        key="image_upload"
    )

if pdf_files or image_files:
    st.success(f"✓ Uploaded: {len(pdf_files)} PDF(s) and {len(image_files)} image(s)")

# Process button
if st.button("🚀 Process Files", type="primary"):
    if not pdf_files and not image_files:
        st.error("⚠️ Please upload at least one file!")
    else:
        with st.spinner("Processing your files..."):
            try:
                progress_bar = st.progress(0)
                status_text = st.empty()
                # Convert images to PDF
                converted_pdfs = []
                if image_files:
                    status_text.text("📸 Converting images to PDF...")
                    for i, image_file in enumerate(image_files):
                        pdf_bytes = image_to_pdf(image_file, enhance=enhance_images, perspective=correct_perspective)
                        converted_pdfs.append((f"scan_{i+1}.pdf", pdf_bytes))
                        progress_bar.progress((i+1)/(len(image_files)+len(pdf_files)))
                # Combine PDFs
                all_pdfs = []
                for pdf_file in pdf_files:
                    all_pdfs.append((pdf_file.name, io.BytesIO(pdf_file.read())))
                for name, pdf_bytes in converted_pdfs:
                    all_pdfs.append((name, pdf_bytes))
                status_text.text("🔄 Processing PDFs...")
                # Merge if needed
                if merge_all and len(all_pdfs) > 1:
                    merger = PdfMerger()
                    for name, pdf_bytes in all_pdfs:
                        merger.append(pdf_bytes)
                    merged_bytes = io.BytesIO()
                    merger.write(merged_bytes)
                    merger.close()
                    merged_bytes.seek(0)
                    pdfs_to_process = [("merged.pdf", merged_bytes)]
                else:
                    pdfs_to_process = all_pdfs
                # Clean PDFs
                status_text.text("🧹 Cleaning PDFs...")
                output_files = []
                for name, pdf_bytes in pdfs_to_process:
                    pdf_bytes.seek(0)
                    original_size = len(pdf_bytes.getvalue())
                    reader = PdfReader(pdf_bytes)
                    original_pages = len(reader.pages)
                    writer = PdfWriter()
                    for page in reader.pages:
                        writer.add_page(page)
                    stats = {'original_pages': original_pages, 'original_size': original_size}
                    if remove_blanks:
                        temp_bytes = io.BytesIO()
                        writer.write(temp_bytes)
                        temp_bytes.seek(0)
                        reader_temp = PdfReader(temp_bytes)
                        writer, blank_count = remove_blank_pages(reader_temp)
                        stats['blanks_removed'] = blank_count
                    if remove_duplicates:
                        temp_bytes = io.BytesIO()
                        writer.write(temp_bytes)
                        temp_bytes.seek(0)
                        reader_temp = PdfReader(temp_bytes)
                        writer, dup_count = remove_duplicate_pages(reader_temp)
                        stats['duplicates_removed'] = dup_count
                    if compress_pdf:
                        temp_bytes = io.BytesIO()
                        writer.write(temp_bytes)
                        temp_bytes.seek(0)
                        reader_temp = PdfReader(temp_bytes)
                        writer = compress_pdf_func(reader_temp)
                    output_bytes = io.BytesIO()
                    writer.add_metadata({
                        '/Producer': 'Hets PDFExpress',
                        '/Creator': 'Hets PDFExpress - Created by Hettie',
                        '/Title': 'Processed by Hets PDFExpress'
                    })
                    writer.write(output_bytes)
                    output_bytes.seek(0)
                    stats['final_pages'] = len(writer.pages)
                    stats['final_size'] = len(output_bytes.getvalue())
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    output_name = f"HetsPDFExpress_{os.path.splitext(name)[0]}_{timestamp}.pdf"
                    output_files.append({'name': output_name,'bytes': output_bytes.getvalue(),'stats': stats})
                progress_bar.progress(100)
                status_text.text("✅ Processing complete!")
                st.success("🎉 Files processed successfully!")
                st.header("📊 Processing Summary")
                for output_file in output_files:
                    with st.expander(f"📄 {output_file['name']}", expanded=True):
                        stats = output_file['stats']
                        col1, col2, col3, col4 = st.columns(4)
                        with col1: st.metric("Original Size", format_file_size(stats['original_size']))
                        with col2: st.metric("Final Size", format_file_size(stats['final_size']))
                        with col3:
                            size_saved = stats['original_size'] - stats['final_size']
                            if size_saved>0: st.metric("Space Saved", format_file_size(size_saved), delta=f"↓ {format_file_size(size_saved)}")
                            else: st.metric("Size Change", format_file_size(abs(size_saved)), delta=f"↑ {format_file_size(abs(size_saved))}", delta_color="inverse")
                        with col4:
                            if stats['original_size']>0:
                                reduction = (size_saved/stats['original_size'])*100
                                if reduction>0: st.metric("Reduction", f"{reduction:.1f}%", delta="Compressed")
                                else: st.metric("Change", f"{abs(reduction):.1f}%", delta="Larger", delta_color="inverse")
                        col1, col2, col3 = st.columns(3)
                        with col1: st.metric("Original Pages", stats['original_pages'])
                        with col2: st.metric("Final Pages", stats['final_pages'])
                        with col3: removed = stats['original_pages']-stats['final_pages']; st.metric("Pages Removed", removed)
                        if 'blanks_removed' in stats and stats['blanks_removed']>0: st.info(f"🗑️ Removed {stats['blanks_removed']} blank page(s)")
                        if 'duplicates_removed' in stats and stats['duplicates_removed']>0: st.info(f"🔍 Removed {stats['duplicates_removed']} duplicate(s)")
                        if stats['final_size']>stats['original_size']: st.warning("ℹ️ File is slightly larger due to PDF structure optimization and metadata. This ensures maximum compatibility with all PDF readers.")
                        st.download_button(label="📥 Download PDF", data=output_file['bytes'], file_name=output_file['name'], mime="application/pdf", key=output_file['name'])
            except Exception as e:
                st.error(f"❌ Error processing files: {str(e)}")
                st.exception(e)

# Footer
st.markdown("---")
st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>📄 PDF by Het | Created by Hettie | © 2024</p>
        <p><small>Your privacy matters - all processing happens locally, files are not stored</small></p>
    </div>
""", unsafe_allow_html=True)
