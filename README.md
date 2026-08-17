# Classification and Captioning With Keras

Classifies aircraft damage images as `dent` or `crack` using transfer learning on a pretrained VGG16 model, then generates captions and summaries for those images using the pretrained BLIP model.

## File Tree

```
Classification_and_Captioning_With_Keras/
├── Final_Project_Classification_and_Captioning.py
└── README.md
```

## Requirements

```
pandas==2.2.3
tensorflow_cpu==2.17.1
pillow==11.1.0
matplotlib==3.9.2
transformers==4.38.2
torch==2.2.0+cpu
torchvision==0.17.0+cpu
torchaudio==2.2.0+cpu
```

## Run

```
python Final_Project_Classification_and_Captioning.py
```

Downloads the aircraft damage dataset automatically on first run.

## What it does

1. Downloads and extracts the aircraft damage dataset (train/valid/test, `dent` vs `crack`).
2. Builds a classifier on top of a frozen, pretrained VGG16 (ImageNet weights).
3. Trains and evaluates the classifier, plots loss/accuracy curves.
4. Visualizes a sample prediction vs. true label.
5. Uses a pretrained BLIP model (via a custom Keras layer) to generate a caption and a summary for a sample image.
