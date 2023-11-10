import json
import os
import time

import numpy as np
import redis
import settings
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.applications.resnet50 import decode_predictions, preprocess_input
from tensorflow.keras.preprocessing import image

# TODO
# Connect to Redis and assign to variable `db``
# Make use of settings.py module to get Redis settings like host, port, etc.
db = redis.Redis(host=settings.REDIS_IP,
                 port=settings.REDIS_PORT,
                 db=settings.REDIS_DB_ID,
                 decode_responses=True)

# TODO
# Load your ML model and assign to variable `model`
# See https://drive.google.com/file/d/1ADuBSE4z2ZVIdn66YDSwxKv-58U7WEOn/view?usp=sharing
# for more information about how to use this model.

model = ResNet50(include_top=True, weights="imagenet")
print("Model loaded!")

# THE FOLLOWING COMMENTED CODE IS USED FOR DELIVER RESULTS WITH BATCH PROCESSING.


# def predict(messages_json):
#     """
#     Returns the predicted class and confidence score for each image filename in
#     `messages_json`. The final format can be passed directly to Redis.

#     Parameters
#     ----------
#     messages_json : list(dict)
#         Image filenames with their corresponding IDs.

#     Returns
#     -------
#     results_with_ids : dict
#         Model predicted class and confidence score for each ID.
#     """
#     if settings.UPLOAD_FOLDER[-1] != "/":
#         settings.UPLOAD_FOLDER += "/"

#     image_names = [message_json["image_name"] for message_json in messages_json]
#     ids = [message_json["id"] for message_json in messages_json]
#     images = []

#     for image_name in image_names:
#         img = image.load_img(settings.UPLOAD_FOLDER + image_name, target_size=(224, 224))
#         x = image.img_to_array(img)
#         images.append(x)
    
#     images = preprocess_input(np.array(images))
#     preds = model.predict(images, batch_size=6)
#     classes = decode_predictions(preds, top=1)

#     results_with_ids = {}
#     index = 0

#     for img_class in classes:
#         results = {"prediction": img_class[0][1], "score": round(img_class[0][2], 4).item()}
#         results_with_ids[str(ids[index])] = json.dumps(results)
#         index += 1
    
#     return results_with_ids

# def classify_process():
#     """
#     Loop indefinitely, asking Redis for new jobs.
#     When a new job list arrives, the jobs are removed from the Redis queue and
#     their corresponding images are classified using the `predict` function. The
#     results are sent back to Redis with the original job ID so that other services
#     can identify them.
#     """
#     while True:
#         messages = db.rpop(settings.REDIS_QUEUE, 12)
        
#         if messages is not None:
#             messages_json = [json.loads(message) for message in messages]

#             results_with_ids = predict(messages_json)

#             db.mset(results_with_ids)

#         # Sleep for a bit
#         time.sleep(settings.SERVER_SLEEP)


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
    if settings.UPLOAD_FOLDER[-1] != "/":
        settings.UPLOAD_FOLDER += "/"

    img = image.load_img(settings.UPLOAD_FOLDER + image_name, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)

    preds = model.predict(x)
    class_name = decode_predictions(preds, top=1)[0][0][1]
    pred_probability = round(decode_predictions(preds, top=1)[0][0][2], 4).item()

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

        message = db.rpop(settings.REDIS_QUEUE)
        # message = message.decode("utf-8")
        
        if message is not None:
            message_json = json.loads(message)
            image_name = message_json["image_name"]

            prediction, score = predict(image_name)
            results = json.dumps({"prediction": prediction, "score": score})

            redis_id = message_json["id"]
            db.set(redis_id, results)

        # Sleep for a bit
        time.sleep(settings.SERVER_SLEEP)


if __name__ == "__main__":
    # Now launch process
    print("Launching ML service...")
    classify_process()
