from torchvision.models.vgg import vgg16
from torchsummary import summary

vgg = vgg16(pretrained=True)
print(summary(vgg),(1,256,256))

class demo(nn.Module):
	def __init__(self):
		super(demo, self).__init__()
		