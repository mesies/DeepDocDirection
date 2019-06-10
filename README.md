# DeepDocDirection
A simple convolutional neural network built with keras which detects the true direction(up, down, right, left) of scanned documents.

## Dataset

I used personal documents as a dataset, so I couldn't include them. Dataset must contain all directions.
Example dataset for training could be
```https://www.cs.cmu.edu/~aharley/rvl-cdip/```
with some images flipped down, right etc.

## Results

Average precision is ~97% when trained with 273 scanned documents.

## Usage
```buildoutcfg
python test_network_cli.py -m pre_trained_model -i image_or_pdf -d dimension_of_image_in_training
```
