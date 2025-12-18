import pdfplumber

def extract_text_from_pdf(pdf_file):
    """
    Extracts text from a PDF file using pdfplumber for better layout handling.
    
    Args:
        pdf_file: A file-like object (e.g., from streamlit.file_uploader)
    
    Returns:
        str: Extracted text.
    """
    text = ""
    try:
        # pdfplumber expects a path or file-like object. 
        # Streamlit's UploadedFile works as a file-like object.
        with pdfplumber.open(pdf_file) as pdf:
            for page in pdf.pages:
                page_text = page.extract_text()
                if page_text:
                    text += page_text + "\n"
    except Exception as e:
        return f"Error reading PDF: {e}"
        
    return text
