import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

img_path = r"../Images/cube.jpg"   # any image
img = Image.open(img_path).convert("RGB")

transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225])
])

x = transform(img).unsqueeze(0).to(device)

def show_feature_map(fmap, title, num_channels=6):
    fmap = fmap[0]  # remove batch
    fig, axes = plt.subplots(1, num_channels, figsize=(15,3))
    for i in range(num_channels):
        axes[i].imshow(fmap[i], cmap="gray")
        axes[i].axis("off")
    fig.suptitle(title)
    plt.show()

vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT)
vgg = vgg.to(device)
vgg.eval()

vgg_features = {}

def vgg_hook(name):
    def hook(module, input, output):
        vgg_features[name] = output.detach().cpu()
    return hook

# Early and deep layers
vgg.features[0].register_forward_hook(vgg_hook("vgg_conv1"))
vgg.features[28].register_forward_hook(vgg_hook("vgg_conv5"))

with torch.no_grad():
    _ = vgg(x)

show_feature_map(vgg_features["vgg_conv1"], "VGG conv1")
show_feature_map(vgg_features["vgg_conv5"], "VGG conv5")
