# Style-Transfer

This repo is a simple implementaion of performing image style transformation in PyTorch. This is an exercise in the Udacity course, Intro to Deep Learning with PyTorch. Speficially, I take a pre-trained VGG-19 model as a feature extractor and use it to extract content and style features from a content and style image, respectively. The transferred image is crafted in a way minimizing the content and style loss.

# Usage
This script has been tested on anaconda3 and PyTorch 0.4.0.

Transferred Image Generation:
```
python style_transfer.py --content_img_path=<path to a content image> 
                         --style_img_path=<path to a style image> 
                         --output_img_dir_path=<path to an output image directory>
```

# Transferred Image
Content Image: space needle, Style Image: kahlo <br/>
![Image description](/transferred_image/space_needle_and_kahlo.png)
