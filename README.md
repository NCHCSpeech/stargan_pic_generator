# stargan_pic_generator
A picture generator which translates pictures according to the sentence it receives.

**(This Project is based on [StarGAN](https://github.com/yunjey/StarGAN))**

## Prerequisites
* [Python 3.5+](https://www.continuum.io/downloads)
* [PyTorch 0.2.0](http://pytorch.org/)
* [TensorFlow 1.3+](https://www.tensorflow.org/) (optional for tensorboard)


## Getting Start
#### 1. Clone the repository
```bash
$ git clone https://github.com/yunjey/StarGAN.git
$ cd StarGAN/
```
#### 2. Download the dataset (CelebA dataset)
```bash
$ bash download.sh
```

#### 3. Generate pictures
```bash
$ python3 generate_pic.py
```
