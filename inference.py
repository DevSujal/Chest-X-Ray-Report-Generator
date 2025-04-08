import io
import utils
import config
import numpy as np

from PIL import Image
from flask_ngrok import run_with_ngrok
from flask import Flask, jsonify, request


app = Flask(__name__)

run_with_ngrok(app)

@app.route('/report/generate', methods=['POST'])
def generate_report():
    global model

    model.eval()

    if request.method == 'POST':
        file = request.files['file']

        image = io.BytesIO(file.read())

        image = np.array(Image.open(image).convert('L'))
        image = np.expand_dims(image, axis=-1)
        image = image.repeat(3, axis=-1)
        image = config.basic_transforms(image=image)['image']

        image = image.to(config.DEVICE)

        report = model.generate_caption(image.unsqueeze(0), max_length=25)

        return jsonify({'report': ' '.join(report)})



if __name__ == '__main__':
    model = utils.get_model_instance(utils.load_dataset().vocab)

    utils.load_checkpoint(model)

    app.run()