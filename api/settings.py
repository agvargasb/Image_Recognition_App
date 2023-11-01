import os

# Run API in Debug mode
API_DEBUG = True

# We will store images uploaded by the user on this folder
UPLOAD_FOLDER = "static/uploads/"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# We will store user feedback on this file
FEEDBACK_FILEPATH = "feedback/feedback.txt"
os.makedirs(os.path.split(FEEDBACK_FILEPATH)[0], exist_ok=True)

with open("feedback/feedback.txt", 'a') as fp: 
    pass

# REDIS settings
# Queue name
REDIS_QUEUE = "service_queue"
# Port
REDIS_PORT = 6379
# DB Id
REDIS_DB_ID = 0
# Host IP
REDIS_IP = os.getenv("REDIS_IP", "redis")
# Sleep parameters which manages the
# interval between requests to our redis queue
API_SLEEP = 0.05
