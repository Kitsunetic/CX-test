import os
import pickle
import random

import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import rawpy
import imageio
import matplotlib.pyplot as plt
import imageio
import skimage.transform
from torchvision import transforms
from tqdm import tqdm

from ContextualLoss import Contextual_Loss, Distance_Type

black_lv = 512
white_lv = 16383
n_epochs = 20
shift_sign = -1

rgb_mean = np.array([0.14428656, 0.11273184, 0.10870461, 0.13277838])


class MeanShiftRaw(nn.Conv2d):
  def __init__(self, rgb_range, rgb_mean, rgb_std=[1., 1., 1., 1.], sign=-1):
    super(MeanShiftRaw, self).__init__(1, 4, kernel_size=1, )
    std = torch.Tensor(rgb_std)
    self.weight.data = torch.eye(4).view(4, 4, 1, 1) / std.view(4, 1, 1, 1)
    self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
    for p in self.parameters():
      p.requires_grad = False

class RAW2RGB(torch.utils.data.Dataset):
  def __init__(self, dataset_path: str, black_lv=512, white_lv=16383):
    self.black_lv = black_lv
    self.white_lv = white_lv
    
    self.train_path = os.path.join(dataset_path, 'train')
    self.test_path = os.path.join(dataset_path, 'test')
    self.train_list = list(filter(lambda f: f.endswith('.ARW'), os.listdir(self.train_path)))
    self.test_list = list(filter(lambda f: f.endswith('.JPG'), os.listdir(self.test_path)))

    with open(os.path.join(dataset_path, 'wb.txt'), 'r') as f:
      self.white_balance = []
      lines = f.read().split('\n')[2::2]
      for line in lines:
        wb = list(map(float, line.split()))
        self.white_balance.append(wb)
  
  def __getitem__(self, idx: int):
    with rawpy.imread(self.train_list[idx]) as raw:
      train = np.array(raw.raw_image_visible, dtype=np.float32)
      train = (train-black_lv) / (white_lv-black_lv)
    
    test = imageio.imread(self.test_list[idx])
    
    transform = transforms.ToTensor()
    train = transform(train)
    test = transform(test)
    wb = torch.Tensor(self.white_balance[idx])
    return train, test, wb, idx

  def __len__(self):
    return min(len(self.train_list),
               len(self.test_list),
               len(self.white_balance))

def main():
  device = torch.device('cpu')

  dataset = RAW2RGB('./datasets')
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

  model = nn.Sequential(
    nn.Conv2d(1, 4, 4, stride=1, padding=2),
    nn.Conv2d(4, 3, 2, stride=2, padding=0),
  ).to(device)

  optimizer = torch.optim.Adam(model.parameters())
  loss = nn.L1Loss().to(device)
  
  history = {'cost': []}
  for epoch in range(1, n_epochs+1):
    with tqdm(total=len(dataset), ncols=128, position=0, miniters=1) as t:
      for step, (train, test, wb, idx) in enumerate(dataloader):
        # to device
        train = train.to(device)
        test = test.to(device)
        wb = wb.to(device)
        
        # adjust wb
        train[0, 0::2, 0::2] -= wb[0]
        train[0, 1::2, 0::2] -= wb[1]
        train[0, 1::2, 1::2] -= wb[2]
        train[0, 0::2, 1::2] -= wb[3]
        
        # forward
        result = model(train)
        cost = loss(result, test)
        
        # backward
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        # disadjust wb
        """
        4->3 채널로 변경되었으므로 몇 번 wb를 적용해야 할 지 모르겠다.
        
        Task 1.
          wb가 RGBG이므로 wb[0, 1, 2], wb[0, 2, 3] 으로 적용해본다.
          
        Task 2.
          target 이미지의 평균값으로 normalize를 해준다.
        """
        
        # update tqdm
        cost_val = cost.item()
        history['cost'].append(cost_val)
        t.set_postfix_str('%.04f'%cost_val)
        t.update()
    
  # show result
  # TODO

if __name__ == "__main__":
  main()
