# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


from roboflow import Roboflow
rf = Roboflow(api_key="9jzqqVFC5hTDvuqcpFDh")
project = rf.workspace().project("ugli-fruit")
model = project.version(4).model

# infer on a local image
print(model.predict("apple-single-red.jpg", confidence=40, overlap=30).json())

# visualize your prediction
# model.predict("your_image.jpg", confidence=40, overlap=30).save("prediction.jpg")

# infer on an image hosted elsewhere
# print(model.predict("URL_OF_YOUR_IMAGE", hosted=True, confidence=40, overlap=30).json())