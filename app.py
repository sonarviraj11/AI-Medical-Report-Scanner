from flask import Flask, request, render_template
from concurrent.futures import ThreadPoolExecutor, as_completed
from Utils.Agent import Cardiologist, Psychologist, Pulmonologist, MultidisciplinaryTeam
import os
from werkzeug.utils import secure_filename

# Text & OCR libs
from PIL import Image
import pytesseract
from PyPDF2 import PdfReader

# Optional: convert PDF pages to images (for scanned PDFs)
try:
    from pdf2image import convert_from_path
    HAS_PDF2IMAGE = True
except Exception:
    HAS_PDF2IMAGE = False

# If tesseract binary isn't on PATH (Windows), set it here, example:
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"


ALLOWED_EXTENSIONS = {'txt', 'pdf', 'png', 'jpg', 'jpeg'}

app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
RESULT_PATH = 'results/final_diagnosis.txt'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(os.path.dirname(RESULT_PATH), exist_ok=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def get_text_from_txt(filepath):
    with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
        return f.read()


def get_text_from_pdf(filepath):
    """
    Try to extract selectable text with PyPDF2. If nothing extracted and
    pdf2image + pytesseract is available, fall back to OCR of each page.
    """
    text_parts = []
    try:
        reader = PdfReader(filepath)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text_parts.append(page_text)
    except Exception as e:
        # log if needed
        print("PyPDF2 extraction error:", e)

    full_text = "\n".join(text_parts).strip()
    if full_text:
        return full_text

    # Fallback: use pdf2image + pytesseract to OCR scanned PDF
    if HAS_PDF2IMAGE:
        try:
            images = convert_from_path(filepath)
            ocr_texts = []
            for img in images:
                ocr_texts.append(pytesseract.image_to_string(img))
            return "\n".join(ocr_texts).strip()
        except Exception as e:
            print("pdf2image/pytesseract fallback error:", e)

    return ""  # nothing found


def get_text_from_image(filepath):
    """
    Use PIL + pytesseract to OCR images.
    """
    try:
        img = Image.open(filepath)
        text = pytesseract.image_to_string(img)
        return text
    except Exception as e:
        print("Image OCR error:", e)
        return ""


def extract_text(filepath):
    ext = filepath.rsplit('.', 1)[1].lower()
    if ext == 'txt':
        return get_text_from_txt(filepath)
    elif ext == 'pdf':
        return get_text_from_pdf(filepath)
    elif ext in ('png', 'jpg', 'jpeg'):
        return get_text_from_image(filepath)
    else:
        return ""


@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        if 'report' not in request.files:
            return render_template("index.html", error="No file part in the request.")

        file = request.files['report']
        if file and allowed_file(file.filename):
            filename = secure_filename(file.filename)
            filepath = os.path.join(UPLOAD_FOLDER, filename)
            file.save(filepath)

            # Extract text depending on type
            medical_report = extract_text(filepath)
            if not medical_report.strip():
                return render_template("index.html", error="Could not extract text from the file. Try another file or ensure Tesseract/Poppler are installed for OCR of scanned PDFs/images.")

            # Run individual specialists (same as your existing flow)
            agents = {
                "Cardiologist": Cardiologist(medical_report),
                "Psychologist": Psychologist(medical_report),
                "Pulmonologist": Pulmonologist(medical_report)
            }

            responses = {}
            with ThreadPoolExecutor() as executor:
                futures = {executor.submit(agent.run): name for name, agent in agents.items()}
                for future in as_completed(futures):
                    agent_name = futures[future]
                    try:
                        responses[agent_name] = future.result()
                    except Exception as e:
                        responses[agent_name] = f"Agent {agent_name} failed: {e}"

            # Run multidisciplinary agent
            team_agent = MultidisciplinaryTeam(
                cardiologist_report=responses.get("Cardiologist", ""),
                psychologist_report=responses.get("Psychologist", ""),
                pulmonologist_report=responses.get("Pulmonologist", "")
            )
            final_diagnosis = team_agent.run()

            # Save the diagnosis
            final_diagnosis_text = "### Final Diagnosis:\n\n" + final_diagnosis
            with open(RESULT_PATH, 'w', encoding='utf-8') as result_file:
                result_file.write(final_diagnosis_text)

            return render_template("index.html", diagnosis=final_diagnosis_text)

        return render_template("index.html", error="Please upload a valid file (.txt, .pdf, .png, .jpg, .jpeg).")

    return render_template("index.html")


if __name__ == '__main__':
    app.run(debug=True)
