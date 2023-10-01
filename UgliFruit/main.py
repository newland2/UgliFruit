import json



from roboflow import Roboflow
rf = Roboflow(api_key="9jzqqVFC5hTDvuqcpFDh")
project = rf.workspace().project("ugli-fruit")
model = project.version(4).model

# infer on a local image

fruit = json.dumps(model.predict("apple-single-red.jpg", confidence=40, overlap=30).json())
fruitDict = json.loads(fruit)
print(fruitDict['predictions'][0]['class'])#Finds the class of a given picutre ie AppleGood or AppleUgly

# infer on an image hosted elsewhere
#print(model.predict("https://photos.google.com/u/1/photo/AF1QipM0kFBxG71Cco14FvjmnDmqKYMqhcnEk3ekPr7I", hosted=True, confidence=40, overlap=30).json())