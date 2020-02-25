import os
import pickle
import random

import imageio
import matplotlib.pyplot as plt
import numpy as np
import rawpy
import skimage.transform
import torch
import torch.nn as nn
from PIL import Image
from torchvision import transforms
from tqdm import tqdm

from ContextualLoss import Contextual_Loss, Distance_Type

black_lv = 512
white_lv = 16383
n_epochs = 200
patch_size = 1024

class RAW2RGB(torch.utils.data.Dataset):
  def __init__(self, dataset_path: str, black_lv=512, white_lv=16383):
    self.black_lv = black_lv
    self.white_lv = white_lv
    
    self.train_path = os.path.join(dataset_path, 'train')
    self.test_path = os.path.join(dataset_path, 'test')
    self.train_list, self.test_list = [], []
    for train, test in zip(os.listdir(self.train_path), os.listdir(self.test_path)):
      self.train_list.append(os.path.join(self.train_path, train))
      self.test_list.append(os.path.join(self.test_path, test))

    with open(os.path.join(dataset_path, 'wb.txt'), 'r') as f:
      self.white_balance = []
      lines = f.read().split('\n')[2::2]
      for line in lines:
        wb = list(map(float, line.split()))
        self.white_balance.append(wb)
  
  def __getitem__(self, idx: int):
    # load images
    with rawpy.imread(self.train_list[idx]) as raw:
      train = np.array(raw.raw_image_visible, dtype=np.float32)[8:-8]
    test = imageio.imread(self.test_list[idx])
    #test = Image.open(self.test_list[idx])
    
    # make image patch
    h, w = min(train.shape[0], test.shape[0]), min(train.shape[1], test.shape[1])
    dh, dw = random.randint(0, h-patch_size), random.randint(0, w-patch_size)
    train = train[dh:dh+patch_size, dw:dw+patch_size]
    test = test[dh:dh+patch_size, dw:dw+patch_size]
    test = Image.fromarray(test)
    
    # white balance
    train = (train-black_lv) / (white_lv-black_lv)
    
    # to tensor transformation
    transform_train = transforms.ToTensor()
    transform_test = transforms.Compose([
      transforms.Resize((test.height//2, test.width//2)),
      transforms.ToTensor()
    ])
    
    # 이미지 스케일 up이 없기 때문에 test의 사이즈가 2배가 됨
    train = transform_train(train)
    test = transform_test(test)
    wb = torch.Tensor(self.white_balance[idx])
    return train, test, wb, idx

  def __len__(self):
    return min(len(self.train_list),
               len(self.test_list),
               len(self.white_balance))

def main():
  os.makedirs('./result', exist_ok=True)
  
  device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

  dataset = RAW2RGB('./datasets')
  dataloader = torch.utils.data.DataLoader(dataset, batch_size=3, shuffle=True)

  model = nn.Sequential(
    nn.Conv2d(1, 4, 4, stride=1, padding=2),
    nn.Conv2d(4, 3, 2, stride=2, padding=0),
  ).to(device)

  optimizer = torch.optim.Adam(model.parameters())
  
  #loss = nn.L1Loss().to(device)
  layers = {
    "conv_1_1": 1.0,
    "conv_3_2": 1.0
  }
  loss = Contextual_Loss(layers, max_1d_size=64).to(device)
  
  history = {'cost': []}
  for epoch in range(1, n_epochs+1):
    with tqdm(total=len(dataloader), desc='[%04d/%04d]'%(epoch, n_epochs), ncols=128, position=0, miniters=1) as t:
      for step, (train, test, wb, idx) in enumerate(dataloader):
        # to device
        train = train.to(device)
        test = test.to(device)
        wb = wb.to(device) # dataloader에서 (1, 4) -> (1, 1, 4)로 변환됨
        
        # adjust wb
        train[:, 0::2, 0::2] -= wb[0][0]
        train[:, 1::2, 0::2] -= wb[0][1]
        train[:, 1::2, 1::2] -= wb[0][2]
        train[:, 0::2, 1::2] -= wb[0][3]
        
        # forward
        result = model(train)
        
        # disadjust wb
        """
        4->3 채널로 변경되었으므로 몇 번 wb를 적용해야 할 지 모르겠다.
        
        Task 1.
          wb가 RGBG이므로 wb[0, 1, 2], wb[0, 2, 3], wb[0, 1+3/2, 2] 으로 적용해본다.
          
        Task 2.
          target 이미지의 평균값으로 normalize를 해준다.
        """
        result[:, 0, :, :] += wb[0][0]
        result[:, 1, :, :] += wb[0][1]
        result[:, 2, :, :] += wb[0][2]
        #result[:, 0, :, :] += wb[0][0]
        #result[:, 1, :, :] += wb[0][2]
        #result[:, 2, :, :] += wb[0][3]
        #result[:, 0, :, :] += wb[0][0]
        #result[:, 1, :, :] += (wb[0][1]+wb[0][3])/2
        #result[:, 2, :, :] += wb[0][2]
        
        # crop result -> raw파일의 w/h가 16pixel씩 크므로, 1/2사이즈 된 result에서 상하좌우 4픽셀씩 잘라낸다
        #result = result[:, :, 4:-4, 4:-4]
        
        # calculate loss
        #print(result.shape, test.shape)
        cost = loss(result, test)
        
        # backward
        optimizer.zero_grad()
        cost.backward()
        optimizer.step()
        
        # update tqdm
        cost_val = cost.item()
        history['cost'].append(cost_val)
        t.set_postfix_str('cost %.04f'%cost_val)
        t.update()
    
      # save result
      if epoch % 1 == 0:
        test_image = transforms.ToPILImage()(test[0].cpu())
        test_image.save('./result/test_%d.png'%epoch)
        test_image.close()
        
        result_image = transforms.ToPILImage()(result[0].cpu())
        result_image.save('./result/result_%d.png'%epoch)
        result_image.close()
        
        with open('./result/history_%d.pkl'%epoch, 'wb') as f:
          pickle.dump(history, f)

if __name__ == "__main__":
  main()
