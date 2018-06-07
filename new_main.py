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
from crop_face import *

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

def generate_new_image(image, features, crop_size=178, image_size=128,
        g_conv_dim=64, c_dim=5, g_repeat_num=6, save_path="tmp.jpg"):
    checkpoint_path = "model/G.ckpt"
    transform = transforms.Compose([
            transforms.Scale(image_size, interpolation=Image.ANTIALIAS),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    image = to_var(transform(image).view(-1, 3, 128, 128))
    label = features_one_hot(features)

    generator = Generator(g_conv_dim, c_dim, g_repeat_num)
    # set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    image.to(device)
    generator.to(device)

    generator.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
    # generator.eval()

    new_image = generator(image, label)
    save_image(denorm(new_image.data), save_path, nrow=1, padding=0)
    
    new_image = Image.open(save_path)
    return new_image

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("-i", "--image")
    parser.add_argument("-f", "--features", default=['Black_Hair', 'Male', 'Young'])

    args = parser.parse_args()

    image_name = args.image
    origin_img = Image.open(image_name)
    result = faceCrop(image_name)
    if not result:
        origin_img = generate_new_image(origin_img, args.features)

    for idx,(coordinates, face) in enumerate(result):
        old_size = face.size
        coordinates[2] += coordinates[0]
        coordinates[3] += coordinates[1]
        face = face.resize((128, 128))

        new_face = generate_new_image(face, args.features)
        new_face = new_face.resize(old_size)
        origin_img.paste(new_face, coordinates)

    origin_img.save("result.jpg")

