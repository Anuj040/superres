# A2N in tensorflow
This repository is a tensorflow-keras implementation of [Attention in Attention Network for Image Super-Resolution](https://arxiv.org/abs/2104.09497) by H. Chen _et al._,
with code reference from [A2N](https://github.com/haoyuc/A2N). 
Currently, it is the bare minimum (_default args on the original repository_) implementation. I plan to extend it further. Meanwhile, please feel free to use, fork the 
code or leave comments if you find any inconsitency.

## Environment
* tensorflow == 2.4.1 (checked with 2.5.0)
* tensorflow-datasets == 4.3.0

## Installation Guideline

```sh
git clone git@github.com:Anuj040/superres.git [-b <branch_name>]
cd superres (Work Directory)

# local environment settings
pyenv local 3.8.5                                 # Installation issues with 3.9.2
python -m pip install poetry
poetry config virtualenvs.create true --local
poetry config virtualenvs.in-project true --local

# In case older version of pip throws installation errors
poetry run python -m pip install --upgrade pip 

# local environment preparation
poetry install

```

## Working with the code
Before running the code please make sure that your datasets follow the below directory hierarchy.

```sh
    superres {work_directory}
    ├── A2N
    ├── datasets                  
    │   ├──train                                   # high resolution images
    │       ├──image1.png
    │       ├──image2.png
    │         :
    │   ├──val
    │       ├──input                               # low resolution images
    │         ├──files  
    │           ├──test_image1.png
    │           ├──test_image2.png
    │           :  
    │       ├──gt                                  # ground truth (high resolution images)
    │         ├──files  
    │           ├──test_image1.png
    │           ├──test_image2.png
    │           :      
    └── ...
```
All the code executions shall happen from work directory.

### Training mode
  ```sh
  poetry run python A2N/start.py
  ```

## Notes
1. The current implementation provides the option of using [perceptual loss](https://arxiv.org/abs/1603.08155) along with _mae_ loss. To only use perceptual loss, please make necessary changes (in _A2N/model.py_ or _A2N/trainer/trainer.py_)
2. Currently perceptual loss is implemented using _VGG19_. I might include other feature extractors as well.
3. To include the perceptual loss or gan in model training, please include ```--percep``` or ```--gan``` as command line arguments.
4. The *loss weights* have been updated to the optimum for my dataset. 
5. Including perceptual loss, slightly improves the model performance with insignificant change to epoch times. Whereas, *gan training* did not provide any appreciable adavantage. Moreover the epoch time was significantly longer (upto 1.8 times).
6. Apart from the loss definitions used in the original implementation, _sobel loss_ has also been included in the training pipeline. This loss incentivises sharp 
edges for crispier boundaries. In my implementation, it slightly improved the model performance.
