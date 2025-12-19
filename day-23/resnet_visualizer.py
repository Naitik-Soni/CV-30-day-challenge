import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load pretrained ResNet-18
resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
resnet = resnet.to(device)
resnet.eval()

img_path = r"../Images/cube.jpg"   # any image
img = Image.open(img_path).convert("RGB")

transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225])
])

x = transform(img).unsqueeze(0).to(device)

feature_maps = {}

def hook_fn(name):
    def hook(module, input, output):
        feature_maps[name] = output.detach().cpu()
    return hook

resnet.conv1.register_forward_hook(hook_fn("conv1"))
resnet.layer1.register_forward_hook(hook_fn("layer1"))
resnet.layer2.register_forward_hook(hook_fn("layer2"))
resnet.layer3.register_forward_hook(hook_fn("layer3"))
resnet.layer4.register_forward_hook(hook_fn("layer4"))

# Forward pass
with torch.no_grad():
    _ = resnet(x)

def show_feature_map(fmap, title, num_channels=6):
    fmap = fmap[0]  # remove batch
    fig, axes = plt.subplots(1, num_channels, figsize=(15,3))
    for i in range(num_channels):
        axes[i].imshow(fmap[i], cmap="gray")
        axes[i].axis("off")
    fig.suptitle(title)
    plt.show()

show_feature_map(feature_maps["conv1"], "ResNet conv1")
show_feature_map(feature_maps["layer1"], "ResNet layer1")
show_feature_map(feature_maps["layer4"], "ResNet layer4")
