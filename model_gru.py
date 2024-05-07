# Referenced from https://github.com/aladdinpersson/Machine-Learning-Collection/blob/master/ML/Pytorch/more_advanced/image_captioning/model.py

import torch.nn as nn
import torchvision.models as models
import torch


class EncoderCNN(nn.Module):
    def __init__(self, embed_size):
        super(EncoderCNN, self).__init__()
        self.cnn = models.resnext101_32x8d(weights="DEFAULT")
        
        self.cnn.fc = nn.Linear(self.cnn.fc.in_features, embed_size)
        # make sure output of cnn model is embed size
        
        for name, param in self.cnn.named_parameters():
            if "fc.weight" in name or "fc.bias" in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
        
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)
                

    def forward(self, images):
        cnn_out = self.cnn(images)
        return self.dropout(self.relu(cnn_out))
    

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(DecoderRNN, self).__init__()
        
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.dropout = nn.Dropout(0.5)
        
        self.gru = nn.GRU(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        # unsqueeze to at time step as first dimension
        # torch.cat with caption embeddings to make timestep-dimensioned inputs
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.gru(embeddings)
        outputs = self.linear(hiddens)
        return outputs
    

class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(CNNtoRNN, self).__init__()
        self.encoder = EncoderCNN(embed_size)
        self.decoder = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, images, captions):
        features = self.encoder(images)
        outputs = self.decoder(features, captions)
        return outputs

    def caption_image(self, image, vocabulary, max_length=50):
        result_caption = []

        with torch.no_grad():
            encoded_image = self.encoder(image.unsqueeze(0)).unsqueeze(0)
            states = None
            

            for _ in range(max_length):
                hiddens, states = self.decoder.gru(encoded_image, states)
                output = self.decoder.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                predicted_word = vocabulary.lookup_token(predicted.item())
                result_caption.append(predicted_word)
                encoded_image = self.decoder.embed(predicted).unsqueeze(0)

                if predicted_word == "<eos>":
                    break

        return result_caption
