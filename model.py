import io
import spacy
from collections import Counter
from collections import defaultdict 
import torch.nn as nn
import torch.optim as optim
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision
import torch
from  torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer
from torch.utils.data import DataLoader, Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image


class EncoderCNN(nn.Module):
    def __init__(self, embed_size, train=False):
        super(EncoderCNN, self).__init__()
        self.use_pretrained = not train
        self.inception = models.inception_v3(pretrained=self.use_pretrained, aux_logits=True) # TODO - aux_logits=False giving issues
        self.inception.fc = nn.Linear(self.inception.fc.in_features, embed_size)
        # make sure output of cnn model is embed size
        
        for name, param in self.inception.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = train
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
                
        
        # self.main = nn.Sequential(
        #     inception,
        #     nn.ReLU(),
        #     nn.Dropout(0.5)
        # )

    def forward(self, images):
        inception_output = self.inception(images)
        try:
            return self.dropout(self.relu(inception_output.logits))
        except AttributeError:
            # inference
            return self.dropout(self.relu(inception_output))
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        # import pdb;pdb.set_trace()
        # print(embeddings.size())
        # print(features.unsqueeze(0).size())
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs
    

class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoder(images)
        # print(features.size())
        # print(captions.size())
        outputs = self.decoder(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            encoded_image = self.encoder(image.unsqueeze(0)).unsqueeze(0)
            states = None
            

            for _ in range(max_length):
                hiddens, states = self.decoder.lstm(encoded_image, states)
                output = self.decoder.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                predicted_word = vocabulary.lookup_token(predicted.item())
                result_caption.append(predicted_word)
                encoded_image = self.decoder.embed(predicted).unsqueeze(0)

                if predicted_word == "<eos>":
                    break

        return result_caption
