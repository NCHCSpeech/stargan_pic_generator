# stargan_pic_generator
A picture generator which translates pictures according to the sentence it receives.

**(This Project is based on [StarGAN](https://github.com/yunjey/StarGAN))**

## Prerequisites
* [Python 3.5+](https://www.continuum.io/downloads)
* [PyTorch 0.4.0](http://pytorch.org/)
* [TensorFlow 1.3+](https://www.tensorflow.org/) (optional for tensorboard)


## Getting Start
#### 1. Clone the repository
```bash
$ git clone https://github.com/NCHCSpeech/stargan_pic_generator.git
$ cd stargan_pic_generator
```
#### 2. Generate pictures
```bash
$ python3 new_main.py -i image/path [-f ['hair_option', 'sex_option', 'age_option']] [-e true|false]

```

- hair_option
  * Black_Hair
  * Brown_Hair
  * Blond_Hair

- sex_option
  * Male
  * Female

- age_option
  * Old
  * Young
