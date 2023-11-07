from locust import HttpUser, between, task
from wand.image import Image


class APIUser(HttpUser):
    wait_time = between(1, 5)

    # Put your stress tests here.
    # See https://docs.locust.io/en/stable/writing-a-locustfile.html for help.
    # TODO
    
    @task
    def index(self):
        self.client.get("http://0.0.0.0/")

    def noise(self):
        with Image(filename="dog.jpeg") as img:
            img.noise("gaussian")
            img.save(filename="dog.jpeg")

    @task
    def predict(self):
        self.noise()
        self.client.post("http://0.0.0.0/predict", files={"file": ("dog.jpeg", open("dog.jpeg", "rb"), "image/jpeg")})
