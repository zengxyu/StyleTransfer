import torch
from torch import nn, optim
from torch.nn import functional as f
from torchvision import models
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt

# Check if gpu available
use_cuda = torch.cuda.is_available()
# use_cuda = False
device = torch.device("cuda:0" if use_cuda else "cpu")
print(device)

# Three images
content_path = "data/content/content_cwr.jpg"
style_path = "data/style/style_circle_tree2.jpg"
size = 256 if use_cuda else 128

loader_transform = transforms.Compose([
    transforms.Scale(size),
    transforms.CenterCrop(size),
    transforms.ToTensor()])
unloader_transform = transforms.ToPILImage()


def read_img(path):
    img = Image.open(path)
    img = loader_transform(img)
    img = img.unsqueeze(0)
    return img.to(device, torch.float)


def show_img(img, title):
    img = img.cpu().clone()
    img = img.squeeze(0)
    img = unloader_transform(img)
    plt.imshow(img)
    plt.title(title)
    plt.pause(0.0001)


content_img = read_img(content_path)
style_img = read_img(style_path)
input_img = content_img.clone()


# plt.figure()
# show_img(content_img, 'content image')


class ContentLoss(nn.Module):
    def __init__(self, target):
        super(ContentLoss, self).__init__()
        self.target = target

    def forward(self, x):
        self.loss = f.mse_loss(x, self.target)
        return x


# Define style loss
class StyleLoss(nn.Module):
    def __init__(self, target):
        super(StyleLoss, self).__init__()
        self.target = target.detach()

    def forward(self, x):
        g = gram_matrix(x)
        self.loss = f.mse_loss(g, self.target)
        return x


def gram_matrix(input):
    a, b, c, d = input.size()
    # a = batch size
    # b = the number of feature maps
    # (c,d) = dimensions of a feature map

    # resize input
    features = input.view(a * b, c * d)
    g = torch.mm(features, features.t())
    # Normalize g
    return g.div(a * b * c * d)


class Normalization(nn.Module):
    def __init__(self, mean, std):
        super(Normalization, self).__init__()
        # .view the mean and std to make them [C x 1 x 1] so that they can
        # directly work with image Tensor of shape [B x C x H x W].
        # B is batch size. C is number of channels. H is height and W is width.
        self.mean = torch.tensor(mean).view(-1, 1, 1)
        self.std = torch.tensor(std).view(-1, 1, 1)

    def forward(self, x):
        # normalize img
        return (x - self.mean) / self.std


# Create model
cnn_normalization_mean = torch.tensor([0.485, 0.456, 0.406]).to(device)
cnn_normalization_std = torch.tensor([0.229, 0.224, 0.225]).to(device)
normalization = Normalization(cnn_normalization_mean, cnn_normalization_std).to(device)

model = nn.Sequential(normalization)

cnn = models.vgg19(pretrained=True).features.to(device).eval()

style_layers = [1, 3, 5, 9, 13]
content_layers = [4]

content_loss_module = []
style_loss_module = []
i = 0
for layer in cnn.children():
    if isinstance(layer, nn.Conv2d):
        i += 1
        model.add_module('conv_{}'.format(i), layer)
        if i in content_layers:
            # add a content loss layer
            content_target = model(content_img).detach()
            c = ContentLoss(content_target)
            model.add_module('c_loss_{}'.format(i), c)
            content_loss_module.append(c)
        if i in content_layers:
            # add a style loss layer
            style_target = model(style_img).detach()
            s = StyleLoss(gram_matrix(style_target))
            model.add_module('s_loss_{}'.format(i), s)
            style_loss_module.append(s)
        if i == max(style_layers):
            break
    elif isinstance(layer, nn.ReLU):
        layer = nn.ReLU(inplace=False)
        model.add_module('relu_{}'.format(i), layer)
    elif isinstance(layer, nn.MaxPool2d):
        model.add_module('pool_{}'.format(i), layer)
    elif isinstance(layer, nn.BatchNorm2d):
        model.add_module('bn_{}'.format(i), layer)
    else:
        raise RuntimeError('Unrecognised layer:{}'.format(layer.__class__.__name__))
print(model)


def run_training(input_img):
    # Start train
    optimizer = optim.LBFGS([input_img.requires_grad_()])
    num_steps = 300
    style_weight = 1000000000
    content_weight = 1

    run = [0]
    while run[0] < num_steps:
        def closure():
            run[0] += 1
            input_img.data.clamp_(0, 1)
            optimizer.zero_grad()
            model(input_img)
            style_score = 0
            content_score = 0
            for sl in style_loss_module:
                style_score += sl.loss
            for cl in content_loss_module:
                content_score += cl.loss
            style_score = style_score * style_weight
            content_score = content_score * content_weight
            loss = style_score + content_score
            loss.backward()
            if run[0] % 50 == 0:
                print("run {}:".format(run))
                print('Style Loss : {:4f} Content Loss: {:4f}'.format(
                    style_score.item(), content_score.item()))
            return loss

        optimizer.step(closure)
    return input_img.data.clamp_(0, 1)

run_training(input_img)

plt.figure()
show_img(input_img, 'Final output')
