import os
import argparse
import cv2
from PIL import Image
import numpy as np
import torch
import torch.optim as optim
import requests
from torchvision import transforms, models

# device checking
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
all_s_weights= {'conv1_1': 1.,
                'conv2_1': 0.75,
                'conv3_1': 0.2,
                'conv4_1': 0.2,
                'conv5_1': 0.2}
c_weight = 1
s_weight = 1e4

class StyleTransfer(object):
    def __init__(self, output_dir_path):
        self.feature_extractor = None
        self.c_features = None
        self.s_features = None
        self.s_grams = dict()
        self.output_dir_path = output_dir_path
        self.num_for_saving = 200
        self.max_epoch = 2000

    def saving_image(self, input_tensor, out_path):
        new_img = input_tensor.to('cpu').clone().detach()
        new_img = new_img.numpy().squeeze()  # discard the batch dimension
        new_img = new_img.transpose(1,2, 0)
        new_img = new_img * np.array((0.229, 0.224, 0.225)) + np.array((0.485, 0.456, 0.406))
        new_img = new_img.clip(0, 1) * 255
        
        # RGB to BGR
        new_img = new_img[..., ::-1]
        cv2.imwrite(out_path, new_img.astype(int))

    def load_Model(self):
        # load VGG19 and discard fully connected layers
        self.feature_extractor = models.vgg19(pretrained=True).features
        
        # freeze all weights in VGG19.
        for para in self.feature_extractor.parameters():
            para.requires_grad_(False)
        self.feature_extractor = self.feature_extractor.to(device)

    def get_features(self, image):
        features = dict()
        
        # pre-define all desired layers
        layers = {'0': 'conv1_1',
                  '5': 'conv2_1',
                  '10': 'conv3_1',
                  '19': 'conv4_1',
                  '28': 'conv5_1',
                  '21': 'conv4_2'  # content representation
                  }
        
        # extract features
        x = image
        for name, layer in self.feature_extractor._modules.items():
            x = layer(x)
            if name in layers:
                features[layers[name]] = x
        
        return features

    def get_gram(self, input_tensor):
        # get the batch_size, depth, height, and width of the input tensor
        _, c, h, w = input_tensor.size()
        
        # reshape the tensor
        input_tensor = input_tensor.view(c, h * w)
        
        # calculate the gram matrix
        return torch.mm(input_tensor, torch.transpose(input_tensor, 0, 1))

    def style_transformation(self, t_img):
        optimizer = optim.Adam([t_img], lr=0.003)
        
        for epoch in range(1, self.max_epoch + 1):
            t_features = self.get_features(t_img)
            c_loss = torch.mean((t_features['conv4_2'] - self.c_features['conv4_2']) ** 2)
            
            # style loss
            s_loss = 0
            for layer in all_s_weights:
                t_feature = t_features[layer]
                _, c, h, w = t_feature.shape
                
                # calculate the target gram matrix
                t_gram = self.get_gram(t_feature)
                # get the style representation from the style image
                s_gram = self.s_grams[layer]
                # calculate layer-wise style loss
                layer_style_loss = all_s_weights[layer] * torch.mean((t_gram - s_gram) ** 2)
                # add to the style loss
                s_loss += layer_style_loss / (c * h * w)
            
            # calcualte the total loss
            total_loss = c_weight * c_loss + s_weight * s_loss
            
            # update your target image
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # save images
            if epoch % self.num_for_saving == 0:
                print('Epoch ', epoch)
                print('Total loss: ', total_loss.item())
                print('Saving Image at epoch ', epoch)
                print('=========')
                full_path = os.path.join(self.output_dir_path, str(epoch) + '.png')
                self.saving_image(t_img, full_path)

    def image_synthesis(self, c_img, s_img, t_img):
        self.load_Model()
        
        self.c_features = self.get_features(c_img)
        self.s_features = self.get_features(s_img)

        # calculate the gram matrices for each layer of the style representation
        for layer in self.s_features:
            self.s_grams[layer] = self.get_gram(self.s_features[layer])

        # style transformation computation
        self.style_transformation(t_img)


def load_img(input_path, max_img_size=400, ref_shape=None):
    img = Image.open(input_path).convert('RGB')
    mean_vec = [0.485, 0.456, 0.406]
    std_vec = [0.229, 0.224, 0.225]


    # resize the input image
    img_size = max_img_size if max(img.size) > max_img_size else max(img.size)

    if ref_shape is not None:
        img_size = ref_shape

    # image transfermation function
    img = transforms.Compose([
            transforms.Resize(img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean_vec,std=std_vec) 
          ])(img)
    # add the batch dimension
    img = img.unsqueeze(0)

    return img


def main():
    # get arguments
    parser = argparse.ArgumentParser()
    parser.add_argument('--content_img_path', type=str, default='', help='path to a content image')
    parser.add_argument('--style_img_path', type=str, default='', help='path to a style image')
    parser.add_argument('--output_img_dir_path', type=str, default='', help='path to an output directory')
    args = parser.parse_args()
    c_img_path = os.path.abspath(args.content_img_path)
    s_img_path = os.path.abspath(args.style_img_path)
    output_dir_path = os.path.abspath(args.output_img_dir_path)

    if not os.path.exists(output_dir_path):
        os.mkdir(output_dir_path)

    c_img = load_img(c_img_path).to(device)
    s_img = load_img(s_img_path, ref_shape=c_img.shape[-2:]).to(device)
    # define a target image
    t_img = c_img.clone().requires_grad_(True).to(device)

    # generate the target image
    StyleTransfer(output_dir_path).image_synthesis(c_img, s_img, t_img)


if __name__ == "__main__":
    main()
