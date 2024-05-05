import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torch
from torch.utils.data import DataLoader

from dataset import TrainDataset, TestDataset, Collate
from model import CNNtoRNN

IMAGES_PATH = './flickr8/Images'
CAPTIONS_PATH = './flickr8/captions.txt'
BATCH_SIZE = 50
NUM_EPOCHS = 20

train_dataset = TrainDataset(CAPTIONS_PATH, IMAGES_PATH, transform=transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ))

test_dataset = TestDataset(CAPTIONS_PATH, IMAGES_PATH, transform=transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
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

sample_img = test_dataset[10][1].to(device)
sample_img_label = test_dataset[10][0]
sample_img_caption = test_dataset[10][2]
sample_vocab = test_dataset.caption_vocab.to(device)

min_loss = float('inf')

with open('epoch_loss.csv', 'w') as epoch_loss_file:
    epoch_loss_file.write(f'epoch, epoch_loss')
    
    for epoch in range(NUM_EPOCHS):
        epoch_loss = 0
        model.train()
        for idx, (img_labels, imgs, captions) in enumerate(train_loader):
            imgs = imgs.to(device)
            captions = captions.to(device)
            # We don't send in the last token of each caption
            # because we want the model to predict the last token.
            outputs = model(imgs, captions[:-1])
            
            # cross entropy loss of probability distribution of each token
            # vs each token in the caption
            loss = criterion(
                outputs.reshape(-1, outputs.shape[2]), captions.reshape(-1)
            )
            epoch_loss += loss.item()
            optimizer.zero_grad()
            loss.backward(loss)
            optimizer.step()

        epoch_losses.append(epoch_loss)
        print(f"EPOCH {epoch} LOSS: {epoch_loss}")
        epoch_loss_file.write(f'{epoch}, {epoch_loss}\n')
        
        if epoch_loss < min_loss:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': epoch_loss,
                }, f'./epoch={epoch}_model.pt')
        
        model.eval()
        print(model.caption_image(sample_img, sample_vocab))




        
