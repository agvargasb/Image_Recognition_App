from locust import HttpUser, between, task
from wand.image import Image


class APIUser(HttpUser):
    wait_time = between(1, 5)

    # Put your stress tests here.
    # See https://docs.locust.io/en/stable/writing-a-locustfile.html for help.
    # TODO
    
    @task
    def index(self):
        self.client.get("/")

    def noise(self):
        with Image(filename="dog.jpeg") as img:
            img.noise("gaussian")
            img.save(filename="dog.jpeg")

    @task
    def predict(self):
        self.noise()
        self.client.post("/predict", files={"file": ("dog.jpeg", open("dog.jpeg", "rb"), "image/jpeg")})

    @task
    def feedback(self):
        self.client.post("/feedback", data={"report": '{"filename": "filename", "prediction": "wrong_prediction", "score": -1.0}\n'})
