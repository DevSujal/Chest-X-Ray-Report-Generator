import os
import io
import json
import numpy as np
from flask import Flask, render_template, request, redirect, send_file
from PIL import Image
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.lib import utils as rl_utils
import textwrap
from dotenv import load_dotenv
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load environment variables from .env file
load_dotenv()

import utils
import config
import google.generativeai as genai

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from huggingface_hub import login

# Only login to HuggingFace if token is provided
hf_token = os.getenv("HF_TOKEN")
if hf_token:
    login(token=hf_token, add_to_git_credential=False)
else:
    print("Some features may not work without authentication.")

# Configure Google Generative AI
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY", "")
if GOOGLE_API_KEY:
    genai.configure(api_key=GOOGLE_API_KEY)
else:
    print("AI-enhanced reports will use fallback mode.")

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = "static/uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

# Load your image captioning mo del
model = utils.get_model_instance(utils.load_dataset().vocab)
utils.load_checkpoint(model)
model.eval()

# Load GPT-2 once
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")
gpt2_model.eval()

os.makedirs(app.config["UPLOAD_FOLDER"], exist_ok=True)


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


def verify_xray_image(image_file):
    """
    Verify if the uploaded image is actually a chest X-ray using Gemini Vision API
    
    Args:
        image_file: File object from Flask request.files
    
    Returns:
        tuple: (is_valid: bool, message: str)
    """
    if not GOOGLE_API_KEY:
        # Skip verification if API key is not configured
        return True, "API key not configured, skipping verification"
    
    try:
        # Read image file
        image_file.stream.seek(0)
        image_bytes = image_file.read()
        image_file.stream.seek(0)  # Reset for later use
        
        # Open with PIL to get format
        pil_image = Image.open(io.BytesIO(image_bytes))
        
        # Create Gemini model
        model_vision = genai.GenerativeModel('gemini-2.5-flash')
        
        # Prepare the image for Gemini
        image_parts = [
            {
                "mime_type": f"image/{pil_image.format.lower() if pil_image.format else 'jpeg'}",
                "data": image_bytes
            }
        ]
        
        # Create prompt for verification
        prompt = """
        Analyze this image carefully and determine if it is a medical chest X-ray (radiograph).
        
        A valid chest X-ray should show:
        - Thoracic cavity including lungs, heart, ribs, and surrounding structures
        - Grayscale medical imaging appearance
        - Proper radiographic orientation (frontal or lateral view)
        
        Invalid images include:
        - Other types of medical scans (CT, MRI, ultrasound, etc.)
        - X-rays of other body parts (skull, limbs, abdomen, etc.)
        - Non-medical images (photos, drawings, random images)
        - Poor quality or corrupted images
        
        Respond in the following JSON format:
        {
            "is_chest_xray": true/false,
            "confidence": "high/medium/low",
            "reason": "brief explanation of your determination"
        }
        """
        
        # Generate response
        response = model_vision.generate_content([prompt, image_parts[0]])
        response_text = response.text.strip()
        
        # Parse response (handle both JSON and plain text responses)
        import re
        
        # Try to extract JSON
        json_match = re.search(r'\{.*?\}', response_text, re.DOTALL)
        if json_match:
            result = json.loads(json_match.group())
            is_valid = result.get("is_chest_xray", False)
            reason = result.get("reason", "Unknown reason")
            confidence = result.get("confidence", "unknown")
        else:
            # Fallback: check for keywords in response
            response_lower = response_text.lower()
            is_valid = any(keyword in response_lower for keyword in ["true", "yes", "is a chest x-ray", "valid chest"])
            reason = response_text[:200]  # First 200 chars
            confidence = "low"
        
        if is_valid:
            print(f"✅ Image verified as chest X-ray (confidence: {confidence})")
            return True, "Valid chest X-ray image"
        else:
            print(f"❌ Image rejected: {reason}")
            return False, f"Invalid image: {reason}"
            
    except Exception as e:
        print(f"⚠️ Image verification error: {e}")
        # In case of error, allow the image but log the error
        return True, f"Verification skipped due to error: {str(e)}"


def process_image(file):
    image_bytes = io.BytesIO(file.read())
    image = np.array(Image.open(image_bytes).convert("L"))
    image = np.expand_dims(image, axis=-1)
    image = image.repeat(3, axis=-1)
    image = config.basic_transforms(image=image)["image"]
    image = image.to(config.DEVICE)
    return image


def get_detailed_report_gpt2(caption, max_new_tokens=150):
    prompt = (
        "Given the following short AI-generated caption of a chest X-ray, write a detailed, professional, patient-friendly radiology report with explanations, findings, and recommendations.\n"
        f"Caption: {caption}\n"
        "Explanation:"
    )
    input_ids = gpt2_tokenizer.encode(prompt, return_tensors="pt")
    with torch.no_grad():
        output_ids = gpt2_model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.9,
            pad_token_id=gpt2_tokenizer.eos_token_id,
        )
    generated_text = gpt2_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Only keep the explanation
    detailed_report = generated_text.split("Explanation:")[-1].strip()
    if not detailed_report:
        detailed_report = "Explanation could not be generated at this time."
    return detailed_report


def extract_clinical_terms_from_caption(caption, max_terms=5):
    clinical_terms_list = [
        "granuloma",
        "consolidation",
        "effusion",
        "nodule",
        "atelectasis",
        "infiltrate",
        "fibrosis",
        "opacity",
        "pneumothorax",
        "cardiomegaly",
        "edema",
        "calcification",
        "pleural thickening",
        "mass",
        "emphysema",
        "pneumonia",
        "sarcoidosis",
        "hyperinflation",
        "collapse",
        "lesion",
        "pleural effusion",
        "interstitial markings",
        "hilar enlargement",
        "lymphadenopathy",
        "bronchiectasis",
        "cavity",
        "scar",
        "infection",
        "pleural fluid",
        "pleural plaque",
        "pleural calcification",
        "reticulation",
        "honeycombing",
        "volume loss",
        "air trapping",
        "bullae",
        "consolidations",
        "nodules",
        "masses",
    ]
    caption_lower = caption.lower()
    found_terms = []
    for term in clinical_terms_list:
        if term in caption_lower and term not in found_terms:
            found_terms.append(term)
        if len(found_terms) == max_terms:
            break
    if found_terms:
        return ", ".join(found_terms)
    else:
        return "No clinical terms could be extracted."


def wrap_text(text, width=80):
    # Helper for wrapping long text for the PDF
    return textwrap.fill(text, width=width)


def generate_good_report(text):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash-lite")

        prompt = f"""
            Use this prompt with the Gemini API:

            Given the following radiology report summary (1-2 sentences), generate a structured report in the following format.

            CLINICAL TAGS

            <comma-separated main findings or conditions, both normal and abnormal>

            KEY FINDINGS

            <one or two clear, concise summary sentences covering key findings>

            ABSTRACT

            COMPARISON: <Describe comparison if present, otherwise state "Not applicable" or "None.">
            INDICATION: <Describe indication/reason for the exam, or state "None provided." if absent.>
            FINDINGS: <Summarize findings in one or two sentences, or state "None reported." if absent.>
            IMPRESSION: <Summarize impression using all clinical information, rephrased as a single concise paragraph.>

            Example input:
            "Heart size and pulmonary vasculature are normal. No evidence of pneumonia, pleural effusion, or pneumothorax."

            Example output:

            CLINICAL TAGS

            Normal heart size, normal pulmonary vasculature, no pneumonia, no pleural effusion, no pneumothorax

            KEY FINDINGS

            The heart size and pulmonary vasculature are normal. No signs of pneumonia, pleural effusion, or pneumothorax.

            ABSTRACT

            COMPARISON: None.
            INDICATION: Routine chest X-ray.
            FINDINGS: The heart and pulmonary vessels are normal, with no abnormal opacities or fluid present. Lungs are clear.
            IMPRESSION: The study shows no acute cardiac or pulmonary abnormality. No pneumonia, pneumothorax, or pleural effusion detected.

            Make sure your output always uses all the information available from the input sentence and clearly fills out each section as shown above.

            Original report : {text}
            """

        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        # Fallback if Gemini API fails
        return f"""CLINICAL TAGS

{text}

KEY FINDINGS

{text}

DETAILED REPORT

1. Initial findings: {text}
2. Please configure GOOGLE_API_KEY environment variable for enhanced AI-powered reports.
"""


def find_k_most_relevant_reports(query_text, k=5, reports_path="reports.json"):
    """
    Find k most similar reports from reports.json using TF-IDF and cosine similarity
    
    Args:
        query_text: The generated report text to compare
        k: Number of most similar reports to return
        reports_path: Path to the reports.json file
    
    Returns:
        List of tuples (report_id, similarity_score, report_data)
    """
    try:
        # Load reports
        with open(reports_path, 'r', encoding='utf-8') as f:
            reports_data = json.load(f)
        
        # Extract report IDs and text content
        report_ids = []
        report_texts = []
        
        for report_id, report in reports_data.items():
            report_ids.append(report_id)
            
            # Combine all abstract fields into one text
            abstract = report.get("Abstract", {})
            combined_text = " ".join([
                str(abstract.get("COMPARISON") or ""),
                str(abstract.get("INDICATION") or ""),
                str(abstract.get("FINDINGS") or ""),
                str(abstract.get("IMPRESSION") or "")
            ])
            report_texts.append(combined_text)
        
        # Add query text to the corpus
        all_texts = report_texts + [query_text]
        
        # Create TF-IDF vectors
        vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Calculate cosine similarity between query and all reports
        query_vector = tfidf_matrix[-1]
        report_vectors = tfidf_matrix[:-1]
        similarities = cosine_similarity(query_vector, report_vectors).flatten()
        
        # Get top k indices
        top_k_indices = similarities.argsort()[-k:][::-1]
        
        # Prepare results
        results = []
        for idx in top_k_indices:
            report_id = report_ids[idx]
            similarity_score = similarities[idx]
            report_data = reports_data[report_id]
            results.append((report_id, similarity_score, report_data))
        
        return results
        
    except Exception as e:
        print(f"❌ Error finding similar reports: {e}")
        return []


def print_similar_reports(query_text, k=5):
    """
    Find and beautifully print k most similar reports to console
    
    Args:
        query_text: The generated report text to compare
        k: Number of most similar reports to display
    """
    print("\n" + "="*100)
    print(f"🔍 FINDING {k} MOST RELEVANT REPORTS")
    print("="*100)
    print(f"\n📄 Query Report:\n{query_text}\n")
    print("-"*100)
    
    similar_reports = find_k_most_relevant_reports(query_text, k)
    
    if not similar_reports:
        print("❌ No similar reports found or error occurred.")
        return
    
    print(f"\n✨ TOP {k} MOST SIMILAR REPORTS:\n")
    
    for rank, (report_id, similarity, report_data) in enumerate(similar_reports, 1):
        print(f"\n{'='*100}")
        print(f"🏆 RANK #{rank} | Report ID: {report_id} | Similarity Score: {similarity:.4f}")
        print(f"{'='*100}")
        
        # Print basic info
        print(f"📅 Date: {report_data.get('Date', 'N/A')}")
        print(f"🏥 Specialty: {report_data.get('Specialty', 'N/A')}")
        print(f"👨‍⚕️ Authors: {', '.join(report_data.get('Authors', ['N/A']))}")
        
        # Print abstract sections
        abstract = report_data.get('Abstract', {})
        print(f"\n📋 ABSTRACT:")
        print(f"  • COMPARISON: {abstract.get('COMPARISON', 'N/A')}")
        print(f"  • INDICATION: {abstract.get('INDICATION', 'N/A')}")
        print(f"  • FINDINGS: {abstract.get('FINDINGS', 'N/A')}")
        print(f"  • IMPRESSION: {abstract.get('IMPRESSION', 'N/A')}")
        
        # Print images
        images = report_data.get('Images', [])
        if images:
            print(f"\n🖼️  Images: {', '.join(images)}")
    
    print("\n" + "="*100)
    print("✅ Search Complete!")
    print("="*100 + "\n")


@app.route("/")
def index():
    return render_template("result1.html")


@app.route("/generator", methods=["GET", "POST"])
def generator():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        age = request.form.get("age", "").strip()
        if "image" not in request.files:
            return redirect(request.url)
        file = request.files["image"]
        if file.filename == "":
            return redirect(request.url)
        if file and allowed_file(file.filename):
            
            # # SECURITY LAYER: Verify if image is a chest X-ray
            # is_valid, verification_message = verify_xray_image(file)
            
            # if not is_valid:
            #     print(f"🚫 Image validation failed: {verification_message}")
            #     return render_template(
            #         "error.html",
            #         error_message=verification_message,
            #         name=name,
            #         age=age
            #     )
            
            # print(f"✅ Image validation passed: {verification_message}")
            
            # Process the validated image
            image_tensor = process_image(file)
            report = model.generate_caption(image_tensor.unsqueeze(0), max_length=25)
            report_text = " ".join(report)
            detailed_report = get_detailed_report_gpt2(report_text)
            clinical_terms = extract_clinical_terms_from_caption(report_text)
            detailed_report = generate_good_report(report_text)
            
            # Find and print k most similar reports
            print_similar_reports(report_text, k=5)
            
            file.stream.seek(0)
            filename = file.filename
            save_path = os.path.join(app.config["UPLOAD_FOLDER"], filename)
            file.save(save_path)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            return render_template(
                "result2.html",
                caption=report_text,
                detailed_report=detailed_report,
                clinical_terms=clinical_terms,
                image_path="/" + save_path,
                now=now,
                name=name,
                age=age,
            )
    return render_template("result1.html")


@app.route("/download-report")
def download_report():
    image_path = request.args.get("image", "").lstrip("/")
    report = request.args.get("report", "No report provided")
    now = request.args.get("now", "")
    detailed_report = request.args.get("detailed_report", "")
    clinical_terms = request.args.get("clinical_terms", "")
    name = request.args.get("name", "")
    age = request.args.get("age", "")
    if not image_path or not os.path.exists(image_path):
        return "Image not found", 404

    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Margins and positions
    left_margin = 50
    right_margin = 400
    top_margin = height - 50
    line_gap = 18

    # Header
    c.setFont("Helvetica-Bold", 16)
    c.drawString(left_margin, top_margin, "AI Chest X-Ray Diagnostic Report")

    c.setFont("Helvetica", 10)
    c.drawString(left_margin, top_margin - line_gap, f"Generated on: {now}")
    c.drawString(left_margin, top_margin - 2 * line_gap, f"Name: {name}    Age: {age}")

    y = top_margin - 4 * line_gap

    # AI Caption
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y, "AI Caption:")
    y -= line_gap
    c.setFont("Helvetica", 12)
    for line in textwrap.wrap(report, 70):
        c.drawString(left_margin, y, line)
        y -= line_gap

    # Clinical Terms
    y -= line_gap // 2
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y, "Extracted Clinical Terms:")
    y -= line_gap
    c.setFont("Helvetica", 12)
    for line in textwrap.wrap(clinical_terms, 70):
        c.drawString(left_margin, y, line)
        y -= line_gap

    # Detailed Report
    y -= line_gap // 2
    c.setFont("Helvetica-Bold", 12)
    c.drawString(left_margin, y, "Explanation:")
    y -= line_gap
    c.setFont("Helvetica", 12)
    for line in textwrap.wrap(detailed_report, 70):
        c.drawString(left_margin, y, line)
        y -= line_gap

    # Place image on the right, vertically aligned with the top of the text
    try:
        img = Image.open(image_path)
        max_img_width = 200
        max_img_height = 200
        img.thumbnail((max_img_width, max_img_height), Image.ANTIALIAS)
        img_io = io.BytesIO()
        img.save(img_io, format="PNG")
        img_io.seek(0)
        img_x = right_margin
        img_y = top_margin - max_img_height
        c.drawImage(
            ImageReader(img_io), img_x, img_y, width=img.width, height=img.height
        )
    except Exception as e:
        print("Image error:", e)

    c.showPage()
    c.save()
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="Xray_Report.pdf",
        mimetype="application/pdf",
    )


if __name__ == "__main__":
    app.run(debug=True)
