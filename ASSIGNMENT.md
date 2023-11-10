# Sprint project 03: Instructions
> Flask ML API

## Note: Here the volumes are named volumes.

You can easily manage files in such volumes with Docker Desktop.

### Named volumes and user permissions:

https://stackoverflow.com/questions/64027052/docker-compose-and-named-volume-permission-denied

### Host volumes and user permissions:

https://stackoverflow.com/questions/63993993/docker-persisted-volum-has-no-permissions-apache-solr/64006395#64006395

## Part 1 - Building the basic service

In this project, we will code and deploy an API for serving our own machine learning models. For this particular case, it will be a Convolutional Neural network for images. You don't need to fully understand how this model works because we will see that in detail later. For now, you can check how to use this model in the notebook [14 - THEORY - CNN Example Extra Material.ipynb](https://drive.google.com/file/d/1ADuBSE4z2ZVIdn66YDSwxKv-58U7WEOn/view?usp=sharing).

The project structure is already defined and you will see the modules already have some code and comments to help you get started.

Below is the full project structure:

```
├── api
│   ├── Dockerfile
│   ├── app.py
│   ├── middleware.py
│   ├── views.py
│   ├── settings.py
│   ├── utils.py
│   ├── templates
│   │   └── index.html
│   └── tests
│       ├── test_api.py
│       └── test_utils.py
├── model
│   ├── Dockerfile
│   ├── ml_service.py
│   ├── settings.py
│   └── tests
│       └── test_model.py
├── stress_test
│   └── locustfile.py
├── docker-compose.yml
├── README.md
└── tests
    └── test_integration.py
```

Let's take a quick overview of each module:

- api: It has all the needed code to implement the communication interface between the users and our service. It uses Flask and Redis to queue tasks to be processed by our machine learning model.
    - `api/app.py`: Setup and launch our Flask api.
    - `api/views.py`: Contains the API endpoints. You must implement the following endpoints:
        - *upload_image*: Displays a frontend in which the user can upload an image and get a prediction from our model.
        - *predict*: POST method which receives an image and sends back the model prediction. This endpoint is useful for integration with other services and platforms given we can access it from any other programming language.
        - *feedback*: Endpoint used to get feedback from users when the prediction from our model is incorrect.
    - `api/utils.py`: Implements some extra functions used internally by our api.
    - `api/settings.py`: It has all the API settings.
    - `api/templates`: Here we put the .html files used in the frontend.
    - `api/tests`: Test suite.
- model: Implements the logic to get jobs from Redis and process them with our Machine Learning model. When we get the predicted value from our model, we must encode it on Redis again so it can be delivered to the user.
    - `model/ml_service.py`: Runs a thread in which it gets jobs from Redis, processes them with the model, and returns the answers.
    - `model/settings.py`: Settings for our ML model.
    - `model/tests`: Test suite.
- tests: This module contains integration tests so we can properly check our system's end-to-end behavior is expected.

Your task will be to complete the corresponding code on those parts it's required across all the modules. You can validate it's working as expected using the already provided tests. We encourage you to also write extra test cases as needed.

You can also take a look at the file `System_architecture_diagram.png` to have a graphical description of the microservices and how the communication is performed.

### Recommended way to work across all those files

Our recommendation for you about the order in which you should complete these files is the following:

#### 1. `model` folder

Inside this module, complete:

1. `predict()` function under `model/ml_service.py` file. Then run the tests corresponding to this module and check if they are passing correctly.
2. Then, go for the `classify_process()` function also under `model/ml_service.py` file.

#### 2. `api` folder

Inside this module, complete:

1. `allowed_file()` function under `api/utils.py` file.
2. `feedback()` function under `api/views.py` file. The `/feedback` endpoint will allow API users report when a model prediction is wrong. You will have to store the reported image path and the model prediction to a plain text file inside the folder `/src/feedback` so we can access later to check those cases in which our Machine Learning model failed according to users.

Now run the tests corresponding to this module and check if they are passing correctly.

3. `model_predict()` function under `api/middleware.py` file. This will allow to communicate the API with our ML service.

## Part 2 - Stress testing with *Locust*

For this task, you must complete the file `locustfile.py` from the `stress_test` folder. Make sure to create at least one test for:
- `index` endpoint.
- `predict` endpoint.

You can use the same environment used for integration testing.

### Test scaled services

You can easily launch more instances for a particular service using `--scale SERVICE=NUM` when running `docker-compose up` command (see [here](https://docs.docker.com/compose/reference/up/)). Scale `model` service to 2 or even more instances and check the performance with locust.

Write a short report detailing the hardware specs from the server used to run the service and show a comparison in the results obtained for a different number of users being simulated and instances deployed.

### Report

Here are the results of stress tests for different numbers of simulated users (U) and different numbers of instances deployed. The numbers inside are the average execution time in milliseconds for each test (the average is calculated from ten values, approximately).

| Predict Test   | U = 1 | U = 2 | U = 3 | U = 4 |
|----------------|:-----:|:-----:|:-----:|:-----:|
| Instances = 1  |  367  |  292  |  265  |  399  |
| Instances = 2  |  268  |  270  |  276  |  288  |

| Feedback Test | U = 1 | U = 2 | U = 3 | U = 4 |
|---------------|:-----:|:-----:|:-----:|:-----:|
| Instances = 1 |  17   |   9   |   8   |   9   |
| Instances = 2 |   9   |   9   |   9   |  10   |

## [Optional] Part 3 - Batch processing

Replace the current model behavior to process the jobs in batches. Check if that improves the numbers when doing stress testing.

### Code

```python
def predict(messages_json):
    """
    Returns the predicted class and confidence score for each image filename in
    `messages_json`. The final format can be passed directly to Redis.

    Parameters
    ----------
    messages_json : list(dict)
        Image filenames with their corresponding IDs.

    Returns
    -------
    results_with_ids : dict
        Model predicted class and confidence score for each ID.
    """
    if settings.UPLOAD_FOLDER[-1] != "/":
        settings.UPLOAD_FOLDER += "/"

    image_names = [message_json["image_name"] for message_json in messages_json]
    ids = [message_json["id"] for message_json in messages_json]
    images = []

    for image_name in image_names:
        img = image.load_img(settings.UPLOAD_FOLDER + image_name, target_size=(224, 224))
        x = image.img_to_array(img)
        images.append(x)
    
    images = preprocess_input(np.array(images))
    preds = model.predict(images, batch_size=6)
    classes = decode_predictions(preds, top=1)

    results_with_ids = {}
    index = 0

    for img_class in classes:
        results = {"prediction": img_class[0][1], "score": round(img_class[0][2], 4).item()}
        results_with_ids[str(ids[index])] = json.dumps(results)
        index += 1
    
    return results_with_ids

def classify_process():
    """
    Loop indefinitely, asking Redis for new jobs.
    When a new job list arrives, the jobs are removed from the Redis queue and
    their corresponding images are classified using the `predict` function. The
    results are sent back to Redis with the original job ID so that other services
    can identify them.
    """
    while True:
        messages = db.rpop(settings.REDIS_QUEUE, 12)
        
        if messages is not None:
            messages_json = [json.loads(message) for message in messages]

            results_with_ids = predict(messages_json)

            db.mset(results_with_ids)

        # Sleep for a bit
        time.sleep(settings.SERVER_SLEEP)
```

For batch processing it is necessary to use the code above, it must replace the `model/ml_service.py` functions (in this case this different structure will fail some tests). Note that the batch_size is set to 6 and that 'classify_process' can remove up to 12 keys from Redis at once. These values can of course be higher. The table below shows the results of batch processing:


| Predict Test   | U = 1 | U = 2 | U = 3 | U = 4 |
|----------------|:-----:|:-----:|:-----:|:-----:|
| Instances = 1  |  281  |  263  |  294  |  279  |
| Instances = 2  |  277  |  283  |  252  |  295  |

| Feedback Test | U = 1 | U = 2 | U = 3 | U = 4 |
|---------------|:-----:|:-----:|:-----:|:-----:|
| Instances = 1 |  11   |   9   |   9   |  17   |
| Instances = 2 |  10   |   8   |   8   |  14   |



The numbers are not very different compared to the previous ones. I think is necessary to have a very big images dataset for the locust tests,
because in these tests, even with the gaussian noise implementation of `stress_test/locust.py', it can happen that two or more get requests try to upload the same image, causing a certain number of requests to fail to upload images and be counted as successful at the same time. Also, with a large number of different images, different values for batch_size can be tested.