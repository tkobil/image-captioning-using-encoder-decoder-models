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

from dataset import TrainDataset, TestDataset, Collate
from model import CNNtoRNN

IMAGES_PATH = './flickr8/Images'
CAPTIONS_PATH = './flickr8/captions.txt'
BATCH_SIZE = 200
NUM_EPOCHS = 5

train_dataset = TrainDataset(CAPTIONS_PATH, IMAGES_PATH, transform=transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    ))

test_dataset = TestDataset(CAPTIONS_PATH, IMAGES_PATH, transform=transforms.Compose(
        [
            transforms.Resize((356, 356)),
            transforms.RandomCrop((299, 299)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
        ]
    ))

train_loader = DataLoader(
    dataset=train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    collate_fn=Collate(pad_idx=train_dataset.caption_vocab['<pad>']),
)

test_loader = DataLoader(
    dataset=test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    pin_memory=True,
    collate_fn=Collate(pad_idx=test_dataset.caption_vocab['<pad>']),
)

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Note, train and test dataset vocab is the same!

model = CNNtoRNN(embed_size=256, hidden_size=256, vocab_size=len(train_dataset.caption_vocab), num_layers=1).to(device)
criterion = nn.CrossEntropyLoss(ignore_index=train_dataset.caption_vocab['<pad>'])
optimizer = optim.Adam(model.parameters(), lr=3e-4)


epoch_losses = []

sample_img = test_dataset[0][0].to(device)
sample_vocab = test_dataset.caption_vocab.to(device)

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0
    model.train()
    for idx, (imgs, captions) in enumerate(train_loader):
        imgs = imgs.to(device)
        captions = captions.to(device)
        outputs = model(imgs, captions[:-1])
        loss = criterion(
            outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
        )
        epoch_loss += loss.item()
        optimizer.zero_grad()
        loss.backward(loss)
        optimizer.step()

    epoch_losses.append(epoch_loss)
    print(f"EPOCH {epoch} LOSS: {epoch_loss}")
    
    model.eval()
    # TODO - loop over test_loader
    # to evaluate metrics...
    print(model.caption_image(sample_img, sample_vocab))


        
