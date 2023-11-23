import numpy as np
import torch

from tqdm import tqdm
import tonic.transforms as transforms
import torchvision
import torchvision.transforms as transforms
from snntorch import utils
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

device = 'cuda' if torch.cuda.is_available() else 'cpu'

seed = 0
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
torch.manual_seed(seed)

print("Torch version:", torch.__version__)

import clip

clip.available_models()
ann, preprocess = clip.load('RN50') # 'RN50', 'RN101', 'RN50x4', 'RN50x16', 'RN50x64', 'ViT-B/32', 'ViT-B/16', 'ViT-L/14', 'ViT-L/14@336px'
ann.float().cuda().eval()

from clip.model_snn import CLIP as CLIP_SNN

weights = []
bias = []
for n,p in ann.visual.named_parameters():
    if hasattr(p, 'weight'):
        weights.append(p.weight.detach().cpu().numpy())
    if hasattr(p, 'bias'):
        bias.append(p.bias.detach().cpu().numpy())

snn = CLIP_SNN(
        ann.embed_dim, 
        ann.image_resolution,
        ann.vision_layers, ann.vision_width, ann.vision_patch_size,
        ann.context_length, ann.vocab_size, 
        ann.transformer_width, ann.transformer_heads, ann.transformer_layers
    )
for n,p in snn.visual.named_parameters():
    if hasattr(p, 'weight'):
        p.weight.data = torch.from_numpy(weights.pop(0)).to(device)
    if hasattr(p, 'bias'):
        p.bias.data = torch.from_numpy(bias.pop(0)).to(device)

torch.cuda.empty_cache()
snn = snn.to(device)

# Basic training parameters
num_epochs = 80
batch_size = 32
lr = 5e-4
out_dim = 10

# LIF neuron parameters
rep = 5

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize(mean=[0.507, 0.487, 0.441],std=[0.267, 0.256, 0.276])])

transform = transforms.Compose(
    [ transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
     transforms.Normalize(mean=[0.507, 0.487, 0.441], std=[0.267, 0.256, 0.276])])

trainset = torchvision.datasets.CIFAR100(root='./data', train=True,
                                        download=True, transform=transform)
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=2)

testset = torchvision.datasets.CIFAR100(root='./data', train=False,
                                       download=True, transform=transform)
testloader = torch.utils.data.DataLoader(testset, batch_size=batch_size,
                                         shuffle=False, num_workers=2)

input_shape=(3, 32, 32)

optimizer = torch.optim.Adam(snn.parameters(), lr=lr, betas=(0.9,0.99))
loss_fn = torch.nn.CrossEntropyLoss()

# ==============
# ==== Test ====
# ==============
test_loss = 0.0
correct, total = 0,0
snn.train(False)

with torch.no_grad():
    for data, target in tqdm(testloader):
        image_input = data.to(device).repeat(rep, 1, 1, 1, 1)
        target = target
        text_descriptions = [f"a bad photo of a {trainset.classes[label]}" for label in target]
        text_tokens = clip.tokenize(text_descriptions).cuda()
        
        # image_features = ann.encode_image(image_input[0]).float()
        image_features = snn.encode_image(image_input).float()

        text_features = ann.encode_text(text_tokens).float()
        text_features /= text_features.norm(dim=-1, keepdim=True)
        text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        top_probs, top_labels = text_probs.cpu().topk(1, dim=-1)

        # loss = loss_fn(top_labels.float(), target)
        # test_loss += loss.item() * data.size(0)
        correct += torch.eq(top_labels.unsqueeze(1), target).sum()
        total += target.size(0)
    print(f'Testing Loss:{test_loss/len(testloader)}')
    print(f'Correct Predictions: {correct/total}')


