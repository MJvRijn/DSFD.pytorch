import torch
import torch.nn as nn 
import torch.nn.functional as F

from data.config import cfg
from models.factory import build_net
from layers.modules import MultiBoxLoss

class DetectionLoss(nn.Module):
  def __init__(self):
    super().__init__()

    # Load detector
    self.net = build_net('train', cfg.NUM_CLASSES, 'vgg')
    self.net.load_state_dict(torch.load('../models/dsfd_exp1.pth'))
    self.net.eval()

    self.criterion = MultiBoxLoss(cfg)

    # Freeze detector weights
    for p in self.net.parameters():
      p.requires_grad = False

  def forward(self, faces, positions):
    batch_size = faces.size(0)
    losses = torch.zeros(size=(batch_size,)).cuda()

    for i in range(batch_size):
      position = positions[i]

      # Prepare image
      facecontext_width = round((position[2] - position[0]).item() * cfg.resize_width * 2)
      facecontext_height = round((position[3] - position[1]).item() * cfg.resize_height * 2)
      facecontext_x = round(position[0].item() * cfg.resize_width - 0.25 * facecontext_width)
      facecontext_y = round(position[1].item() * cfg.resize_height - 0.05 * facecontext_height)

      face = F.interpolate(faces[i:i+1], size=(facecontext_height, facecontext_width), mode='bilinear', align_corners=False)

      pad_left = max(0, facecontext_x)
      pad_right = max(0, cfg.resize_width - (facecontext_x + facecontext_width))
      pad_top = max(0, facecontext_y)
      pad_bottom = max(0, cfg.resize_height - (facecontext_y + facecontext_height))

      face = F.pad(face, (pad_left, pad_right, pad_top, pad_bottom), mode='replicate')

      # Chop overflows
      cxmin = 0 if facecontext_x >= 0 else -facecontext_x
      cymin = 0 if facecontext_y >= 0 else -facecontext_y
      face = face[:, :, cxmin:, cymin:]
      face = face[:, :, :cfg.resize_width+1, :cfg.resize_height+1]
     
      # Calculate loss
      out = self.net(face)
      target = torch.cat((position, torch.ones(1))).unsqueeze(0).unsqueeze(0)

      loss_l_pal1, loss_c_pal1 = self.criterion(out[:3], target)
      loss_l_pa12, loss_c_pal2 = self.criterion(out[3:], target)

      losses[i] = loss_l_pal1 + loss_c_pal1 + loss_l_pa12 + loss_c_pal2

    return losses.mean()

# from PIL import Image
# from torchvision import transforms
# from matplotlib import pyplot as plt
# import matplotlib.patches as patches
# # position = torch.Tensor([0.28, 0.50, 0.34, 0.58])
# # fig,ax = plt.subplots(1)
# # ax.imshow((face[0].cpu() + torch.Tensor(cfg.img_mean)).permute(1,2,0))
# # rect = patches.Rectangle((position[0]*640,position[1]*640),(position[2]-position[0])*640,(position[3]-position[1])*640,linewidth=1,edgecolor='r',facecolor='none')
# # ax.add_patch(rect)
# # plt.show()

# d = DetectionLoss().cuda()
# i = Image.open('/home/matthew/proj/data/WIDERFACE/WIDER_test/images/13--Interview/13_Interview_Interview_2_People_Visible_13_24.jpg')

# transform = transforms.ToTensor()

# i = transform(i).unsqueeze(0)
# i -= torch.Tensor(cfg.img_mean)
# i = i.cuda()

# p = torch.Tensor([[0.25, 0.5, 0.35, 0.58]])
# l = d(i, p)
# print(l)

