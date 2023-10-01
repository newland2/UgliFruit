# This is a sample Python script.
import json

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from roboflow import Roboflow
rf = Roboflow(api_key="9jzqqVFC5hTDvuqcpFDh")
project = rf.workspace().project("ugli-fruit")
model = project.version(4).model

# infer on a local image
temp = model.predict("apple-single-red.jpg", confidence=40, overlap=30).json()
print(temp)
fruit = json.dumps(model.predict("apple-single-red.jpg", confidence=40, overlap=30).json())
fruitDict = json.loads(fruit)
print(fruitDict['predictions'][0]['class'])
# # visualize your pre
# diction
# model.predict("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
#print(model.predict("https://photos.google.com/u/1/photo/AF1QipM0kFBxG71Cco14FvjmnDmqKYMqhcnEk3ekPr7I", hosted=True, confidence=40, overlap=30).json())