import torch
import torchvision.models as models
import torchvision.transforms as T
from PIL import Image
import matplotlib.pyplot as plt
import numpy as np

device = "cuda" if torch.cuda.is_available() else "cpu"

img = Image.open(r"../Images/cube.jpg").convert("RGB")

transform = T.Compose([
    T.Resize((224,224)),
    T.ToTensor(),
    T.Normalize(mean=[0.485,0.456,0.406],
                std=[0.229,0.224,0.225])
])

x = transform(img).unsqueeze(0).to(device)
target = torch.tensor([0]).to(device)  # dummy target


resnet = models.resnet18(weights=models.ResNet18_Weights.DEFAULT).to(device)
vgg = models.vgg16(weights=models.VGG16_Weights.DEFAULT).to(device)

resnet.train()
vgg.train()


def capture_gradients(model):
    grad_dict = {}

    def hook(name):
        def fn(module, grad_input, grad_output):
            grad = grad_output[0]
            grad_dict[name] = grad.abs().mean().item()
        return fn

    for name, layer in model.named_modules():
        if isinstance(layer, torch.nn.Conv2d):
            layer.register_full_backward_hook(hook(name))

    return grad_dict

resnet_grads = capture_gradients(resnet)

out = resnet(x)
loss = torch.nn.CrossEntropyLoss()(out, target)

resnet.zero_grad()
loss.backward()

vgg_grads = capture_gradients(vgg)

out = vgg(x)
loss = torch.nn.CrossEntropyLoss()(out, target)

vgg.zero_grad()
loss.backward()

def plot_gradients(grad_dict, title):
    layers = list(grad_dict.keys())
    grads = list(grad_dict.values())

    plt.figure(figsize=(10,4))
    plt.plot(grads)
    plt.xticks(range(len(layers)), layers, rotation=90)
    plt.ylabel("Mean |Gradient|")
    plt.title(title)
    plt.grid(True)
    plt.show()

plot_gradients(vgg_grads, "VGG Gradient Flow")
plot_gradients(resnet_grads, "ResNet Gradient Flow")
