from collections import Counter
from collections import defaultdict 
import torchvision.transforms as transforms
import torch
from torchtext.vocab import vocab
from torchtext.data.utils import get_tokenizer
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
from PIL import Image

class FlickrDataset(Dataset):
    def __init__(self, captions_path, images_path, transform=transforms.ToTensor()):
        self.tokenizer = get_tokenizer('spacy', language='en_core_web_sm')
        self.train_images, self.train_captions, self.test_images, self.test_captions = self._load_img_and_captions(captions_path)
        self.caption_vocab = self._get_vocab()
        self.train_image_paths = [f"{images_path}/{img_name}" for img_name in self.train_images]
        self.test_image_paths = [f"{images_path}/{img_name}" for img_name in self.test_images]
        self.transform = transform
        
    def _get_vocab(self):
        counter = Counter()
        for caption in self.train_captions + self.test_captions:
            counter.update(self.tokenizer(caption))
                
        caption_vocab = vocab(counter, specials=['<unk>', '<pad>', '<bos>', '<eos>'])
        caption_vocab.set_default_index(caption_vocab["<unk>"])
        return caption_vocab
    
    
    def _load_img_and_captions(self, captions_path):
        train_images = []
        train_captions = []
        
        test_images = []
        test_captions = []
        
        img_caption_dict = self._load_img_caption_dict(captions_path)
        for img_key in img_caption_dict:
            for caption in img_caption_dict[img_key][:-1]:
                train_images.append(img_key)
                train_captions.append(caption)
            
            test_images.append(img_key)
            test_captions.append(img_caption_dict[img_key][-1])
            
        
        return train_images, train_captions, test_images, test_captions

    
    def _load_img_caption_dict(self, captions_path):
        img_capt_dict = defaultdict(list)
        with open(captions_path, 'r') as captions_file:
            for line in captions_file.readlines():
                if line.startswith("image"):
                    # header
                    continue
                
                else:
                    current_line = line.split(',')
                    img = current_line[0]
                    capt = current_line[1]
                    img_capt_dict[img].append(capt)
                    
        return img_capt_dict  

    
class TrainDataset(FlickrDataset):
    def __init__(self, captions_path, images_path, transform=transforms.ToTensor()):
        super(TrainDataset, self).__init__(captions_path, images_path, transform)
    
    def __len__(self):
        return len(self.train_captions)

    
    def __getitem__(self, index):
        image_label = self.train_image_paths[index]
        image = Image.open(image_label).convert("RGB")
        image_tensor = self.transform(image)
                
        caption = self.train_captions[index]
        tokens = self.tokenizer(caption)
        tensor = torch.cat([
            torch.tensor([self.caption_vocab['<bos>']]),
            torch.tensor([self.caption_vocab[token] for token in tokens]),
            torch.tensor([self.caption_vocab['<eos>']])
        ])
                
        
        return image_label, image_tensor, tensor
    
class TestDataset(FlickrDataset):
    def __init__(self, captions_path, images_path, transform=transforms.ToTensor()):
        super(TestDataset, self).__init__(captions_path, images_path, transform)
        
    def __len__(self):
        return len(self.test_captions)

    
    def __getitem__(self, index):
        image_label = self.test_image_paths[index]
        image = Image.open(image_label).convert("RGB")
        image_tensor = self.transform(image)
                
        caption = self.test_captions[index]
        tokens = self.tokenizer(caption)
        tensor = torch.cat([
            torch.tensor([self.caption_vocab['<bos>']]),
            torch.tensor([self.caption_vocab[token] for token in tokens]),
            torch.tensor([self.caption_vocab['<eos>']])
        ])
                
        
        return image_label, image_tensor, tensor


class Collate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        img_labels = torch.stack([item[0] for item in batch], dim=0)
        imgs = torch.stack([item[1] for item in batch], dim=0)
        targets = [item[2] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return img_labels, imgs, targets
    