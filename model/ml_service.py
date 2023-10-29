import json
import os
import time

import numpy as np
import redis
from settings import REDIS_IP, REDIS_PORT, REDIS_DB_ID, REDIS_QUEUE, SERVER_SLEEP
from tensorflow.keras.applications import resnet50
from tensorflow.keras.applications.resnet50 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image

# TODO
# Connect to Redis and assign to variable `db``
# Make use of settings.py module to get Redis settings like host, port, etc.
db = redis.Redis(host=REDIS_IP, port=REDIS_PORT, db=REDIS_DB_ID, decode_responses=True)

# TODO
# Load your ML model and assign to variable `model`
# See https://drive.google.com/file/d/1ADuBSE4z2ZVIdn66YDSwxKv-58U7WEOn/view?usp=sharing
# for more information about how to use this model.
model = resnet50.ResNet50(include_top=True, weights="imagenet")
print("Hi! I'm the model!")
print("I'm ready to go!")

def predict(image_name):
    """
    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.

    Parameters
    ----------
    image_name : str
        Image filename.

    Returns
    -------
    class_name, pred_probability : tuple(str, float)
        Model predicted class as a string and the corresponding confidence
        score as a number.
    """
    class_name = None
    pred_probability = None

    # TODO
    # raise NotImplementedError

    return class_name, pred_probability


def classify_process():
    """
    Loop indefinitely asking Redis for new jobs.
    When a new job arrives, takes it from the Redis queue, uses the loaded ML
    model to get predictions and stores the results back in Redis using
    the original job ID so other services can see it was processed and access
    the results.

    Load image from the corresponding folder based on the image name
    received, then, run our ML model to get predictions.
    """
    while True:
        # Inside this loop you should add the code to:
        #   1. Take a new job from Redis
        #   2. Run your ML model on the given data
        #   3. Store model prediction in a dict with the following shape:
        #      {
        #         "prediction": str,
        #         "score": float,
        #      }
        #   4. Store the results on Redis using the original job ID as the key
        #      so the API can match the results it gets to the original job
        #      sent
        # Hint: You should be able to successfully implement the communication
        #       code with Redis making use of functions `brpop()` and `set()`.
        # TODO
        # raise NotImplementedError

        queue_name, message = db.brpop(REDIS_QUEUE, 0)
        message = message.decode("utf-8")

        message_json = json.loads(message)
        image_name = message_json["image_name"]

        prediction, score = predict(image_name)

        redis_id = message_json["id"]
        db.set(redis_id, json.dumps({"prediction": prediction, "score": score}))

        # Sleep for a bit
        time.sleep(SERVER_SLEEP)


if __name__ == "__main__":
    # Now launch process
    print("Launching ML service...")
    classify_process()
