import os
import io
import numpy as np
from flask import Flask, render_template, request, redirect, send_file
from PIL import Image
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
from reportlab.lib import utils as rl_utils
import textwrap
import utils
import config

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

from huggingface_hub import login
login()  # Uses cached token

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load your image captioning model
model = utils.get_model_instance(utils.load_dataset().vocab)
utils.load_checkpoint(model)
model.eval()

# Load GPT-2 once
gpt2_tokenizer = AutoTokenizer.from_pretrained("gpt2")
gpt2_model = AutoModelForCausalLM.from_pretrained("gpt2")
gpt2_model.eval()

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def process_image(file):
    image_bytes = io.BytesIO(file.read())
    image = np.array(Image.open(image_bytes).convert('L'))
    image = np.expand_dims(image, axis=-1)
    image = image.repeat(3, axis=-1)
    image = config.basic_transforms(image=image)['image']
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
            pad_token_id=gpt2_tokenizer.eos_token_id
        )
    generated_text = gpt2_tokenizer.decode(output_ids[0], skip_special_tokens=True)
    # Only keep the explanation
    detailed_report = generated_text.split("Explanation:")[-1].strip()
    if not detailed_report:
        detailed_report = "Explanation could not be generated at this time."
    return detailed_report

def extract_clinical_terms_from_caption(caption, max_terms=5):
    clinical_terms_list = [
        "granuloma", "consolidation", "effusion", "nodule", "atelectasis", "infiltrate",
        "fibrosis", "opacity", "pneumothorax", "cardiomegaly", "edema", "calcification",
        "pleural thickening", "mass", "emphysema", "pneumonia", "sarcoidosis",
        "hyperinflation", "collapse", "lesion", "pleural effusion", "interstitial markings",
        "hilar enlargement", "lymphadenopathy", "bronchiectasis", "cavity", "scar", "infection",
        "pleural fluid", "pleural plaque", "pleural calcification", "reticulation", "honeycombing",
        "volume loss", "air trapping", "bullae", "consolidations", "nodules", "masses"
    ]
    caption_lower = caption.lower()
    found_terms = []
    for term in clinical_terms_list:
        if term in caption_lower and term not in found_terms:
            found_terms.append(term)
        if len(found_terms) == max_terms:
            break
    if found_terms:
        return ', '.join(found_terms)
    else:
        return "No clinical terms could be extracted."

def wrap_text(text, width=80):
    # Helper for wrapping long text for the PDF
    return textwrap.fill(text, width=width)

@app.route('/')
def index():
    return render_template('result1.html')

@app.route('/generator', methods=['GET', 'POST'])
def generator():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        age = request.form.get('age', '').strip()
        if 'image' not in request.files:
            return redirect(request.url)
        file = request.files['image']
        if file.filename == '':
            return redirect(request.url)
        if file and allowed_file(file.filename):
            image_tensor = process_image(file)
            report = model.generate_caption(image_tensor.unsqueeze(0), max_length=25)
            report_text = ' '.join(report)
            detailed_report = get_detailed_report_gpt2(report_text)
            clinical_terms = extract_clinical_terms_from_caption(report_text)
            file.stream.seek(0)
            filename = file.filename
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            return render_template(
                'result2.html',
                caption=report_text,
                detailed_report=detailed_report,
                clinical_terms=clinical_terms,
                image_path='/' + save_path,
                now=now,
                name=name,
                age=age
            )
    return render_template('result1.html')

@app.route('/download-report')
def download_report():
    image_path = request.args.get('image', '').lstrip('/')
    report = request.args.get('report', 'No report provided')
    now = request.args.get('now', '')
    detailed_report = request.args.get('detailed_report', '')
    clinical_terms = request.args.get('clinical_terms', '')
    name = request.args.get('name', '')
    age = request.args.get('age', '')
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
    c.drawString(left_margin, top_margin - 2*line_gap, f"Name: {name}    Age: {age}")

    y = top_margin - 4*line_gap

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
        img.save(img_io, format='PNG')
        img_io.seek(0)
        img_x = right_margin
        img_y = top_margin - max_img_height
        c.drawImage(ImageReader(img_io), img_x, img_y, width=img.width, height=img.height)
    except Exception as e:
        print("Image error:", e)

    c.showPage()
    c.save()
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name="Xray_Report.pdf", mimetype='application/pdf')

if __name__ == '__main__':
    app.run(debug=True)
