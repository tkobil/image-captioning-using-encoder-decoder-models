import sys
import torch
import torchvision.transforms as transforms
from dataset import TestDataset
from model import CNNtoRNN
import matplotlib.pyplot as plt  
from nltk.translate.bleu_score import sentence_bleu
from nltk.translate.gleu_score import sentence_gleu
from nltk.translate.meteor_score import single_meteor_score
import nltk

IMAGES_PATH = './flickr8/Images'
CAPTIONS_PATH = './flickr8/captions.txt'

device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

test_dataset = TestDataset(CAPTIONS_PATH, IMAGES_PATH, transform=transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    ))

def get_model(model_name):
    if model_name == "ResNextCNNtoRNNSingleLayer":
        model =  CNNtoRNN(embed_size=256, hidden_size=256, vocab_size=len(test_dataset.caption_vocab), num_layers=1).to(device)
        checkpoint = torch.load('./model.pt')
        model.load_state_dict(checkpoint['model_state_dict'])
        return model

model_name = sys.argv[1]
model = get_model(model_name)


vocab = test_dataset.caption_vocab.to(device)
results = []

for idx, (img_label, img, caption) in enumerate(test_dataset):
    sample_img = img.to(device)
    sample_img_label = img_label
    sample_img_caption = caption
    generated_caption = model.caption_image(sample_img, vocab)
    result = {
        'img_label': img_label,
        'img': img.T,
        'target_caption': [vocab.lookup_token(token) for token in caption],
        'generated_caption': generated_caption
    }
    results.append(result)
    if idx == 5:
        break

nltk.download('wordnet')

# Calculate BLEU-4, gleu, meteor Score
sum_bleu_scores = 0
sum_gleu_scores = 0
sum_meteor_scores = 0
number_samples = 0
for result in results:
    number_samples += 1
    references = [result['target_caption']]
    candidates = result['generated_caption']
    sum_bleu_scores += sentence_bleu(references, candidates)
    sum_gleu_scores = sentence_gleu(references, candidates)
    sum_meteor_scores += single_meteor_score(result['target_caption'], result['generated_caption'])
    
avg_bleu_score = sum_bleu_scores / number_samples
print(f"BLEU SCORE: {avg_bleu_score}")

avg_gleu_score = sum_gleu_scores / number_samples
print(f"GLEU SCORE: {avg_gleu_score}")

avg_meteor_score = sum_meteor_scores / number_samples
print(f"METEOR SCORE: {avg_meteor_score}")

pytorch_total_params = sum(p.numel() for p in model.parameters())
pytorch_total_trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
print(f"TOTAL PARAMS: {pytorch_total_params}")
print(f"TOTAL TRAINABLE PARAMS {pytorch_total_trainable_params}")