import numpy as np
import torch
from torchvision import transforms
from torchvision.models.vgg import vgg11
from sklearn.preprocessing import LabelEncoder

from PIL import Image

resize = transforms.Resize([224, 224])

def preprocess_img(image):
    '''
	This method processes the image by cropping and reshaping to 224x224.
	'''
    image = transforms.functional.resized_crop(\
                image, i=20, j=15, h=300, w=300, size=(224,224))
    return image

def image_loader(image):
    '''
	This method converts the image into a PyTorch tensor that the CNN can accept.
	'''
    image = transforms.functional.to_tensor(image)
    image = (image-image.mean())/image.std()
    image = image.unsqueeze(0)
    return image


class Predictor:
    def __init__(self):
        # ======== YOUR CODE ========= #
        self.model = vgg11(num_classes=26)
        # load model
        self.model.load_state_dict(torch.load('model/results/vgg_final.pth'))

        self.label_encoder = LabelEncoder()
        self.label_encoder.fit(['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W',\
         'X', 'Y', 'Z'])

    def predict(self, request):
        '''
		This method reads the file uploaded from the Flask application POST request,
		and performs a prediction using the MNIST model.
		'''

        #read in image, preprocess, convert to tensor
        f = request.files['image']
        image = Image.open(f)
        image = preprocess_img(image)
        image = image_loader(image)

        #feed image tensor to CNN and return output
        model_output = self.model(image)
        prediction = torch.argmax(model_output)

        output = self.label_encoder.inverse_transform([prediction])

        return output[0]
