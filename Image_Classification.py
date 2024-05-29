# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
# """
from torch.utils.data import Dataset
from pathlib import Path
from PIL import Image
from torchvision import transforms
from torch.utils.data import DataLoader
from torch import nn
import zipfile
import torch
from tqdm.auto import tqdm


# ���J�v����� , �����Y zip �ɮ�
with zipfile.ZipFile("one_piece_mini.zip", "r") as zip_file:
  zip_file.extractall("one_piece_mini")


# �N�ɮץ]�˦�Dataset
class ImageDataset(Dataset):
  def __init__(self, root, train, transform=None):
    
    # �ھڰV�m�δ��ն��]�m�Ϲ��ڥؿ�
    if train:
      image_root = Path(root) / "train"
    else:
      image_root = Path(root) / "test"  # �P�_Ū�����O�V�m����ƩάO���ն����
      
     # Ū�����O�W��
    with open(Path(root) / "classnames.txt", "r") as f:
      lines = f.readlines()
      self.classes = [line.strip() for line in lines]  # �h����������
      
    # ���o�Ҧ��Ϲ���󪺸��|
    self.paths = [i for i in image_root.rglob("*") if i.is_file()]  
    self.transform = transform  # �Ϲ��ഫ�ާ@

  def __getitem__(self, index):
    # �ھگ���Ū���Ϲ�
    img = Image.open(self.paths[index]).convert("RGB")  
    class_name = self.paths[index].parent.name  # ���o�Ϲ����ݪ����O�W��
    class_idx = self.classes.index(class_name)  # ���o���O����

    # �p�G���ഫ�ާ@�A�h�����ഫ
    if self.transform:
      return self.transform(img), class_idx
    else:
      return img, class_idx  


  def __len__(self):
    return len(self.paths)  # ��^�ƾڶ����j�p

  
# �Τ@�v���j�p,�ƾڼW�j,�ഫ��tensor�榡
train_transforms = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.TrivialAugmentWide(),
  transforms.ToTensor()
])

test_transforms = transforms.Compose([
  transforms.Resize((64, 64)),
  transforms.ToTensor()
])
train_dataset = ImageDataset("one_piece_mini", train=True, transform=train_transforms)
train_dataset.classes

# import matplotlib.pyplot as plt
# x, y = train_dataset[100]
# plt.imshow(x.permute(1, 2, 0))  ���ܺ������v�����


# �إ߰V�m�M���� Dataset
train_dataset = ImageDataset(root="one_piece_mini",
                train=True,
                transform=train_transforms
)

test_dataset = ImageDataset(root="one_piece_mini",
                train=False,
                transform=test_transforms
)
# len(train_dataset), len(test_dataset)
# x, y = test_dataset[0]
# x.shape, y


# �إ� DataLoader
BATCH_SIZE = 8

train_dataloader = DataLoader(dataset=train_dataset,
                batch_size=BATCH_SIZE,
                shuffle=True
)

test_dataloader = DataLoader(dataset=test_dataset,
                batch_size=BATCH_SIZE,
                shuffle=False
)
# len(train_dataloader), len(test_dataloader)


# ���o�@����
x_first_batch, y_first_batch = next(iter(train_dataloader))


# x_first_batch[0].shape, y_first_batch[0]


# �إ߼ҫ�

class ImageClassificationModel(nn.Module):
  def __init__(self, input_shape, output_shape):
    super().__init__()
    self.conv_block_1 = nn.Sequential(     
      nn.Conv2d(in_channels=input_shape,
          out_channels=8,
          kernel_size=(3, 3),
          stride=1,
          padding=1
      ),
      nn.ReLU(),
      nn.Conv2d(in_channels=8,
          out_channels=8,
          kernel_size=(3, 3),
          stride=1,
          padding=1
      ),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=(2, 2),
            stride=2,
            padding=0
      )
    )

    self.conv_block_2 = nn.Sequential(   
      nn.Conv2d(in_channels=8,
          out_channels=16,
          kernel_size=(3, 3),
          stride=1,
          padding=1
      ),
      nn.ReLU(),
      nn.Conv2d(in_channels=16,
          out_channels=16,
          kernel_size=(3, 3),
          stride=1,
          padding=1
      ),
      nn.ReLU(),
      nn.MaxPool2d(kernel_size=(2, 2),
            stride=2,
            padding=0
      )
    )

    self.classifier = nn.Sequential(
      nn.Flatten(start_dim=1, end_dim=-1),
      nn.Linear(in_features=16*16*16, out_features=output_shape)
    )
  def forward(self, x):
    x = self.conv_block_1(x)
    x = self.conv_block_2(x)
    x = self.classifier(x)
    return x

model = ImageClassificationModel(3, len(train_dataset.classes))


# �w�q�l����Ƥγ̨Τƨ��
cost_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(params=model.parameters(), lr=0.001)


# �p��ǽT�v
def accuracy_fn(y_pred, y_true):
  correct_num = (y_pred==y_true).sum()
  acc = correct_num / len(y_true) * 100
  return acc


# �إ߰V�m�y�{
def train_step(dataloader, model, cost_fn, optimizer, accuracy_fn):
  train_cost = 0
  train_acc = 0
  for batch, (x, y) in enumerate(dataloader):
   
    model.train() # �]�m�ҫ����V�m�Ҧ�

    y_pred = model(x)

    cost = cost_fn(y_pred, y) # �p��l��

    train_cost += cost
    train_acc += accuracy_fn(y_pred.argmax(dim=1), y) # �p��ǽT�v
     
    optimizer.zero_grad()  # ����k�s

    cost.backward()  # �p����

    optimizer.step()  # ��s�Ѽ�

  train_cost /= len(train_dataloader)
  train_acc /= len(train_dataloader)

  print(f"\nTrain Cost: {train_cost:.4f}, Train Acc: {train_acc:.2f}")


# ���ըB�J
def test_step(dataloader, model, cost_fn, accuracy_fn):
  test_cost = 0
  test_acc = 0
  model.eval()  # �]�m�ҫ��������Ҧ�
  with torch.inference_mode():
    for x, y in dataloader:
    
      test_pred = model(x) 

      test_cost += cost_fn(test_pred, y)   # �p��l��
      test_acc += accuracy_fn(test_pred.argmax(dim=1), y)  # �p��ǽT�v

    test_cost /= len(test_dataloader)
    test_acc /= len(test_dataloader)

  print(f"Test Cost: {test_cost:.4f}, Test Acc: {test_acc:.2f} \n")
  

# �}�l�V�m
epochs = 10

for epoch in tqdm(range(epochs)):
  print(f"Epoch: {epoch}\n-------")
  train_step(train_dataloader, model, cost_fn, optimizer, accuracy_fn)

  test_step(test_dataloader, model, cost_fn, accuracy_fn)

