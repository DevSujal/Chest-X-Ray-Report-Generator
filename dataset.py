import os
import spacy
import torch
import config
import utils
import numpy as np
import xml.etree.ElementTree as ET

from PIL import Image
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader


spacy_eng = spacy.load('en_core_web_sm')


class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {
            0: '<PAD>',
            1: '<SOS>',
            2: '<EOS>',
            3: '<UNK>',
        }
        self.stoi = {
            '<PAD>': 0,
            '<SOS>': 1,
            '<EOS>': 2,
            '<UNK>': 3,
        }
        self.freq_threshold = freq_threshold

    @staticmethod
    def tokenizer(text):
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4

        for sent in sentence_list:
            for word in self.tokenizer(sent):
                if word not in frequencies:
                    frequencies[word] = 1
                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word

                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi['<UNK>']
            for token in tokenized_text
        ]

    def __len__(self):
        return len(self.itos)


class XRayDataset(Dataset):
    def __init__(self, root, transform=None, freq_threshold=3, raw_caption=False):
        self.root = root
        self.transform = transform
        self.raw_caption = raw_caption

        self.vocab = Vocabulary(freq_threshold=freq_threshold)

        self.captions = []
        self.imgs = []

        for file in os.listdir(os.path.join(self.root, 'reports')):
            if file.endswith('.xml'):
                tree = ET.parse(os.path.join(self.root, 'reports', file))

                frontal_img = ''
                findings = tree.find(".//AbstractText[@Label='FINDINGS']").text

                if findings is None:
                    continue

                for x in tree.findall('parentImage'):
                    if frontal_img != '':
                        break

                    img = x.attrib['id']
                    img = os.path.join(config.IMAGES_DATASET, f'{img}.png')

                    frontal_img = img

                if frontal_img == '':
                    continue
                
                self.captions.append(findings)
                self.imgs.append(frontal_img)
                

        self.vocab.build_vocabulary(self.captions)

    def __getitem__(self, item):
        img = self.imgs[item]
        caption = utils.normalize_text(self.captions[item])

        img = np.array(Image.open(img).convert('L'))
        img = np.expand_dims(img, axis=-1)
        img = img.repeat(3, axis=-1)

        if self.transform is not None:
            img = self.transform(image=img)['image']

        if self.raw_caption:
            return img, caption
        
        numericalized_caption = [self.vocab.stoi['<SOS>']]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi['<EOS>'])

        return img, torch.as_tensor(numericalized_caption, dtype=torch.long)

    def __len__(self):
        return len(self.captions)

    def get_caption(self, item):
        return self.captions[item].split(' ')


class CollateDataset:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        images, captions = zip(*batch)

        images = torch.stack(images, 0)
        
        targets = [item for item in captions]
        targets = pad_sequence(targets, batch_first=True, padding_value=self.pad_idx)

        return images, targets


if __name__ == '__main__':
    all_dataset = XRayDataset(
        root=config.DATASET_PATH,
        transform=config.basic_transforms,
        freq_threshold=config.VOCAB_THRESHOLD,
    )

    train_loader = DataLoader(
        dataset=all_dataset,
        batch_size=config.BATCH_SIZE,
        pin_memory=config.PIN_MEMORY,
        drop_last=True,
        shuffle=True,
        collate_fn=CollateDataset(pad_idx=all_dataset.vocab.stoi['<PAD>']),
    )

    for img, caption in train_loader:
        print(img.shape, caption.shape)
        break