import os
import io
import numpy as np
from flask import Flask, render_template, request, redirect, send_file
from PIL import Image
from datetime import datetime
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
from reportlab.lib.utils import ImageReader
import utils
import config

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Load model once on startup
model = utils.get_model_instance(utils.load_dataset().vocab)
utils.load_checkpoint(model)
model.eval()

# Ensure upload folder exists
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

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/generator', methods=['GET', 'POST'])
def generator():
    if request.method == 'POST':
        if 'image' not in request.files:
            return redirect(request.url)

        file = request.files['image']

        if file.filename == '':
            return redirect(request.url)

        if file and allowed_file(file.filename):
            # Preprocess image for model
            image_tensor = process_image(file)

            # Generate report
            report = model.generate_caption(image_tensor.unsqueeze(0), max_length=25)
            report_text = ' '.join(report)

            # Save file for display
            file.stream.seek(0)
            filename = file.filename
            save_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(save_path)

            # Current timestamp
            now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            return render_template('result.html', image_path='/' + save_path, report=report_text, now=now)

    return render_template('result.html')

@app.route('/download-report')
def download_report():
    image_path = request.args.get('image', '').lstrip('/')
    report = request.args.get('report', 'No report provided')
    now = request.args.get('now', '')

    if not image_path or not os.path.exists(image_path):
        return "Image not found", 404

    # Create PDF
    buffer = io.BytesIO()
    c = canvas.Canvas(buffer, pagesize=letter)
    width, height = letter

    # Draw title
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, height - 50, "AI Chest X-Ray Diagnostic Report")

    # Draw timestamp
    c.setFont("Helvetica", 10)
    c.drawString(50, height - 70, f"Generated on: {now}")

    # Draw report
    text = c.beginText(50, height - 100)
    text.setFont("Helvetica", 12)
    text.textLines("Report:\n" + report)
    c.drawText(text)

    # Draw image
    try:
        img = Image.open(image_path)
        img = img.resize((300, 300))
        img_io = io.BytesIO()
        img.save(img_io, format='PNG')
        img_io.seek(0)
        c.drawImage(ImageReader(img_io), 50, height - 450, width=300, height=300)
    except Exception as e:
        print("Image error:", e)

    c.showPage()
    c.save()
    buffer.seek(0)

    return send_file(buffer, as_attachment=True, download_name="Xray_Report.pdf", mimetype='application/pdf')

if __name__ == '__main__':
    app.run(debug=True)
