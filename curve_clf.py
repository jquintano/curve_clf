# import pickle
# from skimage.io import imread
# from skimage.transform import resize
# import numpy as np


# def main(img_path):
# 	loaded_model = pickle.load(open('./model.p', 'rb'))
# 	new_data = []
# 	img = imread(img_path)
# 	img = resize(img, (15, 15))
# 	flattened_img = img.flatten()
# 	new_data.append(flattened_img)
# 	new_data = np.asarray(new_data)
# 	prediction = loaded_model.predict(new_data)
# 	return prediction

# if __name__ == "__main__":
# 	main()

# import pickle
# import argparse
# from skimage.io import imread
# from skimage.transform import resize
# import numpy as np

# MODEL_PATH = './model.p'
# LABELS = ['convex', 'concave']

# def predict_image(img_path):
#     loaded_model = pickle.load(open(MODEL_PATH, 'rb'))
#     new_data = []
#     img = imread(img_path)
#     img = resize(img, (15, 15))
#     flattened_img = img.flatten()
#     new_data.append(flattened_img)
#     new_data = np.asarray(new_data)
#     prediction = loaded_model.predict(new_data)
#     return prediction

# def main():
#     parser = argparse.ArgumentParser(description='Predict convex or concave using a trained model on an image.')
#     parser.add_argument('image_path', type=str, help='Path to the input image')

#     args = parser.parse_args()
#     image_path = args.image_path

#     prediction = predict_image(image_path)
#     print(f"Prediction: {LABELS[prediction[0]]}")

# if __name__ == "__main__":
#     main()

import pickle
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from skimage.io import imread
from skimage.transform import resize
import numpy as np

app = FastAPI()

MODEL_PATH = './model.p'
LABELS = ['convex', 'concave']

loaded_model = pickle.load(open(MODEL_PATH, 'rb'))

def predict_image(img):
    new_data = []
    img = resize(img, (15, 15))
    flattened_img = img.flatten()
    new_data.append(flattened_img)
    new_data = np.asarray(new_data)
    prediction = loaded_model.predict(new_data)
    return LABELS[prediction[0]]

@app.post("/predict")
def predict(file: UploadFile = File(...)):
    try:
        # contents = file.file.read()
        img = imread(file.file)
        prediction = predict_image(img)
        return JSONResponse(content={"Prediction": str(prediction)})
    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)




