import sys
sys.path.append("../")
import torch
from data.dataset import Im2LatexDataset
from model.model import Img2MathModel

img_dims = [256, 256]
data = Im2LatexDataset(path_to_data="../data/",
                       tokenizer="../data/tokenizer.json", img_dims=img_dims, batch_size=16, device=torch.device('mps'))
img, label = next(iter(data.train))
vocab_size = len(data.tokenizer.get_vocab())
enc_output = torch.randn(1, 512, 512)
model = Img2MathModel.load_from_checkpoint('../train/img2math/362edsd8/checkpoints/epoch=9-step=65180.ckpt')
model.eval()
generation = model.generate(img[0])
print(generation)
