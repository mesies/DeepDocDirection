import getopt
import sys

import cv2
import imutils
import numpy
import numpy as np
from keras.models import load_model
from keras.preprocessing.image import img_to_array
from pdf2image import convert_from_path


def eval_image(image_path_of_eval_image, model, image_dimension):
    if image_path_of_eval_image.endswith('.pdf'):
        pages = convert_from_path(image_path_of_eval_image, dpi=500)
        page_number = 0
        for page in pages:
            _eval_image(page.convert('RGB'), model, image_dimension, 'Page number ' + str(page_number), 0)
            page_number += 1

    else:
        _eval_image(image_path_of_eval_image, model, image_dimension)


def _eval_image(image_path_of_eval_image, model, image_dim, image_name='', readfrompath=1):
    if image_name == '':
        image_name = image_path_of_eval_image

    if readfrompath == 1:
        image = cv2.imread(image_path_of_eval_image)
    else:
        _image = image_path_of_eval_image
        _image = numpy.array(_image)
        # Convert RGB to BGR
        image = _image[:, :, ::-1].copy()
    orig = image.copy()

    # pre-process the image for classification
    image = cv2.resize(image, (image_dim, image_dim))
    image = image.astype("float") / 255.0
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)

    # classify the input image
    (up, right, down, left) = model.predict(image)[0]
    print("[INFO] printing probability for " + image_name)
    print("[INFO] UP : " + str(up))
    print("[INFO] RIGHT : " + str(right))
    print("[INFO] DOWN : " + str(down))
    print("[INFO] LEFT : " + str(left))

    # build the label
    label = "up"
    probability = up  # + 0.4

    if probability < right:
        label = "right"
        probability = right

    if probability < down:
        label = "down"
        probability = down

    if probability < left:
        label = "left"
        probability = left

    label = "{}: {:.2f}%".format(label, probability * 100)

    # draw the label on the image
    output = imutils.resize(orig, width=600)
    cv2.putText(output, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                0.7, (0, 255, 0), 2)

    # show the output image
    cv2.imshow("Output", output)
    cv2.waitKey(0)


my_opts, args = getopt.getopt(sys.argv[1:], "i:o:")

model_path = "trained_model"

image_path = "combined.pdf"

# Load model from disk
print("[INFO] loading network...")
loaded_model = load_model(model_path)
image_dim = 320

for o, a in my_opts:
    if o == '-i':
        image_cli_path = a
    elif o == '-m':
        model_path = a
    elif o == '-d':
        image_dim = a
    else:
        print("Usage: -i -> path of image to predict,")
        print("       -m -> path of trained model,")
        print("       -d -> image dimension of trained model")

eval_image(image_cli_path, loaded_model, image_dim)
