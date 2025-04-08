import re
import torch
import config
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from collections import OrderedDict


class DenseNet121(nn.Module):
    def __init__(self, out_size=14, checkpoint=None):
        super(DenseNet121, self).__init__()

        self.densenet121 = models.densenet121(weights='DEFAULT')

        num_classes = self.densenet121.classifier.in_features

        self.densenet121.classifier = nn.Sequential(
            nn.Linear(num_classes, out_size),
            nn.Sigmoid()
        )

        if checkpoint != None:
            checkpoint = torch.load(checkpoint, map_location=torch.device('cpu'))


            state_dict = checkpoint['state_dict']

            new_state_dict = OrderedDict()

            for k, v in state_dict.items():
                if 'module' not in k:
                    k = f'module.{k}'
                else:
                    k = k.replace('module.densenet121.features', 'features')
                    k = k.replace('module.densenet121.classifier', 'classifier')
                    k = k.replace('.norm.1', '.norm1')
                    k = k.replace('.conv.1', '.conv1')
                    k = k.replace('.norm.2', '.norm2')
                    k = k.replace('.conv.2', '.conv2')

                    new_state_dict[k] = v

            self.densenet121.load_state_dict(new_state_dict)

    def forward(self, x):
        return self.densenet121(x)


class EncoderCNN(nn.Module):
    def __init__(self, checkpoint=None):
        super(EncoderCNN, self).__init__()

        self.model = DenseNet121(
            checkpoint=checkpoint
        )

        for param in self.model.densenet121.parameters():
            param.requires_grad_(False)

    def forward(self, images):
        features = self.model.densenet121.features(images)

        batch, maps, size_1, size_2 = features.size()

        features = features.permute(0, 2, 3, 1)
        features = features.view(batch, size_1 * size_2, maps)

        return features


class Attention(nn.Module):
    def __init__(self, features_size, hidden_size, output_size=1):
        super(Attention, self).__init__()

        self.W = nn.Linear(features_size, hidden_size)
        self.U = nn.Linear(hidden_size, hidden_size)
        self.v = nn.Linear(hidden_size, output_size)

    def forward(self, features, decoder_output):
        decoder_output = decoder_output.unsqueeze(1)

        w = self.W(features)
        u = self.U(decoder_output)

        scores = self.v(torch.tanh(w + u))
        weights = F.softmax(scores, dim=1)
        context = torch.sum(weights * features, dim=1)

        weights = weights.squeeze(2)

        return context, weights


class DecoderRNN(nn.Module):
    def __init__(self, features_size, embed_size, hidden_size, vocab_size):
        super(DecoderRNN, self).__init__()

        self.vocab_size = vocab_size

        self.embedding = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTMCell(embed_size + features_size, hidden_size)

        self.fc = nn.Linear(hidden_size, vocab_size)

        self.attention = Attention(features_size, hidden_size)

        self.init_h = nn.Linear(features_size, hidden_size)
        self.init_c = nn.Linear(features_size, hidden_size)

    def forward(self, features, captions):
        embeddings = self.embedding(captions)

        h, c = self.init_hidden(features)

        seq_len = len(captions[0]) - 1
        features_size = features.size(1)
        batch_size = captions.size(0)

        outputs = torch.zeros(batch_size, seq_len, self.vocab_size).to(config.DEVICE)
        atten_weights = torch.zeros(batch_size, seq_len, features_size).to(config.DEVICE)

        for i in range(seq_len):
            context, attention = self.attention(features, h)

            inputs = torch.cat((embeddings[:, i, :], context), dim=1)

            h, c = self.lstm(inputs, (h, c))
            h = F.dropout(h, p=0.5)

            output = self.fc(h)

            outputs[:, i, :] = output
            atten_weights[:, i, :] = attention

        return outputs, atten_weights

    def init_hidden(self, features):
        features = torch.mean(features, dim=1)

        h = self.init_h(features)
        c = self.init_c(features)

        return h, c


class EncoderDecoderNet(nn.Module):
    def __init__(self, features_size, embed_size, hidden_size, vocabulary, encoder_checkpoint=None):
        super(EncoderDecoderNet, self).__init__()

        self.vocabulary = vocabulary

        self.encoder = EncoderCNN(
            checkpoint=encoder_checkpoint
        )
        self.decoder = DecoderRNN(
            features_size=features_size,
            embed_size=embed_size,
            hidden_size=hidden_size,
            vocab_size=len(self.vocabulary)
        )

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs, _ = self.decoder(features, captions)

        return outputs

    def generate_caption(self, image, max_length=25):
        caption = []

        with torch.no_grad():
            features = self.encoder(image)
            h, c = self.decoder.init_hidden(features)

            word = torch.tensor(self.vocabulary.stoi['<SOS>']).view(1, -1).to(config.DEVICE)
            embeddings = self.decoder.embedding(word).squeeze(0)

            for _ in range(max_length):
                context, _ = self.decoder.attention(features, h)

                inputs = torch.cat((embeddings, context), dim=1)

                h, c  = self.decoder.lstm(inputs, (h, c))

                output = self.decoder.fc(F.dropout(h, p=0.5))
                output = output.view(1, -1)

                predicted = output.argmax(1)

                if self.vocabulary.itos[predicted.item()] == '<EOS>':
                    break

                caption.append(predicted.item())

                embeddings = self.decoder.embedding(predicted)

        return [self.vocabulary.itos[idx] for idx in caption]
