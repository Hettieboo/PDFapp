
# Het's PDFExpress

**Document Processing Web Application**
Created by Hettie

---

## Overview

**Het’s PDFExpress** is a document processing web application built with **Python** and **Streamlit**.
It enables users to upload, clean, enhance, and convert PDF or image files quickly and locally.

The app supports image enhancement, PDF merging, blank page detection, and compression, making it a complete solution for lightweight document workflows.

---

## Features

### PDF Processing

* Merge multiple PDF files
* Remove blank or duplicate pages
* Compress and optimize PDF size
* Add metadata such as title and creator
* Download processed files instantly

### Image Processing

* Enhance scanned images for better readability
* Correct perspective and alignment automatically
* Convert image files (JPEG, PNG) into PDFs
* (Optional) Apply OCR for text recognition using Tesseract

### Privacy

All file operations are performed locally on the user’s machine.
No data is uploaded or stored externally.

---

## Technology Stack

* **Frontend / UI:** Streamlit
* **Backend Language:** Python
* **PDF Processing:** PyPDF2
* **Image Processing:** OpenCV, Pillow
* **OCR (optional):** pytesseract
* **PDF Conversion:** img2pdf

---

## Installation

### 1. Install Dependencies

```bash
pip install streamlit PyPDF2 pillow opencv-python pytesseract img2pdf
```

### 2. Install Tesseract OCR (for OCR features)

**Ubuntu/Debian:**

```bash
sudo apt-get install tesseract-ocr
```

**macOS (Homebrew):**

```bash
brew install tesseract
```

**Windows:**
Download from: [https://github.com/UB-Mannheim/tesseract/wiki](https://github.com/UB-Mannheim/tesseract/wiki)

### 3. Run the Application

Make sure `appLogo.png` is in the same directory as `app.py`, then run:

```bash
streamlit run app.py
```

The app will open automatically in your web browser (default: `http://localhost:8501`).

---

## Project Structure

```
pdf-by-het/
│
├── app.py                # Main application file
├── appLogo.png           # Application logo
├── requirements.txt      # Optional dependency list
└── README.md             # Documentation
```

---

## Usage

1. Upload one or more PDF or image files.
2. Configure settings in the sidebar:

   * Image enhancement and perspective correction
   * Merge, compress, or clean PDF files
3. Click **Process Files** to start processing.
4. Download the processed output directly from the interface.

---

## Tips

* Keep image enhancement enabled for scanned documents.
* Install Tesseract if you plan to use OCR.
* The app is designed for local automation—no external dependencies beyond Python libraries.

---

## Author

**Hettie**
Developer and creator of Het’s PDFExpress

---

## License

© 2025 Hettie. All rights reserved.
This project is proprietary and not open for redistribution or reuse.

