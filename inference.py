import torch
from torch import nn
from torch import sigmoid
import segmentation_models_pytorch as smp
from utils import big_image_predict


# device = torch.device('cpu')

class Inference_model:
    def __init__(self, path_to_model = 'weights.pth', device=None):
        if device == 'cuda':
            self.device = torch.device("cuda:" + str(torch.cuda.device_count() - 1) if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device('cpu')
        self.model = smp.Unet(encoder_name='efficientnet-b3', encoder_weights='imagenet', in_channels=1, classes=2,  activation='sigmoid').to(self.device)
        self.model.load_state_dict(torch.load(path_to_model))
        
    def predict(self, img):
        img, st, asb = big_image_predict(self.model, 
                                  img, 
                                  crop_size=(1024, 1024),#(img_tr_stones_shape[0] // 2, img_tr_stones_shape[1] // 2),
                                  inp_size=(1024, 1024),
                                  device=self.device
                                 )
        return asb

