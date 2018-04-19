import os
import argparse
from solver import Solver
from data_loader import get_loader
from torch.backends import cudnn
from inference import Inference
from PIL import Image
from model import Generator
import torch
from torchvision.utils import save_image
from torchvision import transforms
from torch.autograd import Variable

def denorm(x):
        out = (x + 1) / 2
        return out.clamp_(0, 1)

def to_var(x, volatile=True):
    if torch.cuda.is_available():
        x = x.cuda()
    return Variable(x, volatile=True)

def features_one_hot(features):
    selected_features = ['Black_Hair', 'Blond_Hair', 'Brown_Hair', 'Male', 'Young']
    features_label = []
    for f in selected_features:

        if f in features:
            features_label.append(1)
        else:
            features_label.append(0)
    label = torch.FloatTensor([features_label])
    
    return to_var(label)

def generate_new_image(image_path, features, output_path, crop_size=178, image_size=128, 
        g_conv_dim=64, c_dim=5, g_repeat_num=6):
    checkpoint_path = "stargan_celebA/models/10_4000_G.pth"
    transform = transforms.Compose([
            transforms.CenterCrop(crop_size),
            transforms.Scale(image_size, interpolation=Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    image = to_var(transform(Image.open(image_path)).view(-1, 3, 128, 128))
    label = features_one_hot(features)

    generator = Generator(g_conv_dim, c_dim, g_repeat_num)
    if torch.cuda.is_available():
        generator.cuda()

    generator.load_state_dict(torch.load(checkpoint_path))
    generator.eval()
    print(image.size())
    new_image = generator(image, label)
    save_image(denorm(new_image.data), output_path, nrow=1, padding=0)
    
    new_image = Image.open(output_path)
    return new_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--image")
    parser.add_argument("-f", "--features",action='append', default=[])
    parser.add_argument("-o", "--output")

    args = parser.parse_args()
    if not args.features:
        args.features = ['Black_Hair', 'Male', 'Young']
    generate_new_image(args.image, args.features, args.output)e