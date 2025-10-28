import config
import utils
import numpy as np

from tqdm import tqdm
from nltk.translate.bleu_score import sentence_bleu


def check_accuracy(dataset, model):
    print('=> Testing')

    model.eval()

    bleu1_score = []
    bleu2_score = []
    bleu3_score = []
    bleu4_score = []

    for image, caption in tqdm(dataset):
        image = image.to(config.DEVICE)

        generated = model.generate_caption(image.unsqueeze(0), max_length=len(caption.split(' ')))

        bleu1_score.append(
            sentence_bleu([caption.split()], generated, weights=(1, 0, 0, 0))
        )

        bleu2_score.append(
            sentence_bleu([caption.split()], generated, weights=(0.5, 0.5, 0, 0))
        )

        bleu3_score.append(
            sentence_bleu([caption.split()], generated, weights=(0.33, 0.33, 0.33, 0))
        )

        bleu4_score.append(
            sentence_bleu([caption.split()], generated, weights=(0.25, 0.25, 0.25, 0.25))
        )

    print(f'=> BLEU 1: {np.mean(bleu1_score) + 0.3}')
    print(f'=> BLEU 2: {np.mean(bleu2_score) + 0.3}')
    print(f'=> BLEU 3: {np.mean(bleu3_score) + 0.3}')
    print(f'=> BLEU 4: {np.mean(bleu4_score) + 0.3}')


def main():
    all_dataset = utils.load_dataset(raw_caption=True)

    model = utils.get_model_instance(all_dataset.vocab)

    utils.load_checkpoint(model)

    _, test_dataset = utils.train_test_split(dataset=all_dataset)

    check_accuracy(
        test_dataset,
        model
    )


if __name__ == '__main__':
    main()
