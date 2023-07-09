import sys
sys.path.append("../")
import torch
from data.dataset import Im2LatexDataset
from model.model import Img2MathModel


device = torch.device('mps')

def main():
    img_dims = [256, 256]
    data = Im2LatexDataset(path_to_data="../data/",
                           tokenizer="../data/tokenizer.json", img_dims=img_dims, batch_size=4, device=device)
    batch = next(iter(data.train))
    img, label = batch
    img = img.to(device)
    label = label.to(device)
    model = Img2MathModel.load_from_checkpoint('../train/img2math/362edsd8/checkpoints/epoch=9-step=65180.ckpt', map_location=device)
    preds = model(img, trg_seq=label['input_ids'])
    print(preds)

if __name__ == '__main__':
    main()
