# 🩺 Chest X-Ray Report Generator

The main functionality of the app is to **generate AI-powered diagnostic reports from chest X-ray images** using a deep learning model (based on CheXNet and DenseNet121) with an interactive and beautiful web-based UI powered by Flask.

---

## 🚀 Features

- Upload chest X-ray images via a web interface  
- AI model generates diagnostic reports automatically  
- Real-time "Scanning"  
- Stylish and responsive Tailwind CSS UI  
- Download generated reports as PDF  
- BLEU-score-based evaluation for model accuracy  

---

## 📦 Installation

1. **Clone the repository:**

```bash
git clone https://github.com/SHIVANSH-A/Chest-X-Ray-Report-Generator.git
cd Chest-X-Ray-Report-Generator
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

---

## 🧠 Dataset

**IU X-Ray**  
Download the IU X-Ray Dataset from:  
🔗 https://openi.nlm.nih.gov

Place the contents in the `dataset` folder:

```
dataset/
├── reports/
└── images/
```

---

## 🏋️‍♂️ Checkpoints

This model uses CheXNet (DenseNet121) as the encoder.  
Place pretrained weights inside the `weights/` directory.

```
weights/
└── chexnet_densenet121.pth  # Example
```

---

## ⚙️ Configurations

Modify training and model parameters in `config.py`.  
Each task-specific config file is modular and easy to tune.

---

## 🧪 Training and Evaluation

### 🔧 Train the Model

```bash
python train.py
```

You can continue training from a checkpoint or start fresh depending on your configuration.

### 📊 Evaluate the Model

The current evaluation is based on the **BLEU** metric.  
You can customize the metric in `eval.py`, specifically in the `check_accuracy` method.

```bash
python eval.py
```

---

## 🌐 Flask Web App

### ✅ How it works:

- Built using Flask + Tailwind CSS  
- Upload an image from the home page  
- Image is scanned visually using a loading animation  
- Diagnostic report appears after a few seconds  
- Option to download the generated report as a PDF  

### ▶️ Run the App

```bash
python app.py
```

Then open your browser and go to:  
[http://127.0.0.1:5000](http://127.0.0.1:5000)

---

## 📁 Structure

```
templates/
├── index.html       # Landing page
├── result.html      # Scanning animation + report

static/
├── css/             # Tailwind styles (via CDN)

app.py               # Flask backend
utils.py             # Helper functions
```

---

## 📄 PDF Report Download

After generating a report, users can click a **Download PDF** button that includes:

- The uploaded X-ray image  
- The AI-generated report  
- Timestamp of the scan  

PDF is generated server-side using `reportlab`.

---

## 🤝 Contributing

PRs and suggestions are welcome. This is both an academic and learning project.

---

## 🧑‍🎓 Acknowledgements

- IU X-Ray dataset from OpenI  
- CheXNet architecture (DenseNet121)  
- BLEU metric for evaluation  
- Flask, Tailwind CSS, and ReportLab  

---


```

