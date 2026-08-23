from PIL import Image

import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input

IMAGE_SIZE = (224, 224)


def preprocess_image(image):

    image = Image.open(image)

    image = image.convert("RGB")

    image = image.resize(
        IMAGE_SIZE
    )

    image = np.array(
        image,
        dtype=np.float32
    )
      image = preprocess_input(image)


    image = np.expand_dims(
        image,
        axis=0
    )

    return image
