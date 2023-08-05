# Retinal_Blood_Vessels_Segmentation

## U-net architecture
```
class conv_block(nn.Module):
  def __init__(self, in_c, out_c):
    super().__init__()
    self.conv1 = nn.Conv2d(in_c, out_c, kernel_size=3, padding='same')
    self.bn1 = nn.BatchNorm2d(out_c)
    self.conv2 = nn.Conv2d(out_c, out_c, kernel_size=3, padding='same')
    self.bn2 = nn.BatchNorm2d(out_c)

  def forward(self, inputs):
    x = self.conv1(inputs)
    x = self.bn1(x)
    x = F.leaky_relu(x)
    x = self.conv2(x)
    x = self.bn2(x)
    x = F.leaky_relu(x)
    return x

class encoder_block(nn.Module):
  def __init__(self, in_c, out_c):
    super().__init__()
    self.conv_block = conv_block(in_c, out_c)

  def forward(self, inputs):
    x = self.conv_block(inputs)
    p = F.max_pool2d(x, 2)
    #print(f"Encoder x shape:{x.shape}, p shape: {p.shape}")
    return x, p

class decoder_block(nn.Module):
  def __init__(self, in_c, out_c):
    super().__init__()
    self.up_conv = nn.ConvTranspose2d(in_c, out_c, kernel_size=2, stride=2, padding=0, output_padding=0)
    self.conv_block = conv_block(out_c+out_c, out_c)

  def forward(self, inputs, skip):
    x = self.up_conv(inputs)
    d2 = skip.shape[2] - x.shape[2]
    d3 = skip.shape[3] - x.shape[3]
    x = torch.cat([x, skip[:,:,d2//2:d2//2+x.shape[2],d3//2:d3//2+x.shape[3]]], axis=1)
    x = self.conv_block(x)
    return x
```

```
class build_unet(nn.Module):
  def __init__(self):
    super().__init__()
    self.e1 = encoder_block(3, 64)
    self.e2 = encoder_block(64, 128)
    self.e3 = encoder_block(128, 256)
    self.e4 = encoder_block(256, 512)

    self.b = conv_block(512, 1024)

    self.d1 = decoder_block(1024, 512)
    self.d2 = decoder_block(512, 256)
    self.d3 = decoder_block(256, 128)
    self.d4 = decoder_block(128, 64)

    self.outputs = nn.Conv2d(64, 1, kernel_size=1, padding='same')

  def forward(self, inputs):

    s1, p1 = self.e1(inputs)
    s2, p2 = self.e2(p1)
    s3, p3 = self.e3(p2)
    s4, p4 = self.e4(p3)
    b = self.b(p4)
    
    d1 = self.d1(b, s4)
    d2 = self.d2(d1, s3)
    d3 = self.d3(d2, s2)
    d4 = self.d4(d3, s1)

    outputs = self.outputs(d4)
    outputs = F.sigmoid(outputs)
    return outputs
```
## Loss function
```
def dice_coef(y_true, y_pred):
  smooth = 1e-15
  y_true = torch.flatten(y_true)
  y_pred = torch.flatten(y_pred)
  intersection = (y_true * y_pred).sum()
  return (2. * intersection + smooth) / (y_true.sum() + y_pred.sum() + smooth)

def dice_loss(y_true, y_pred):
  return 1.0 - dice_coef(y_true, y_pred)
```

## Training
The training took 100 epochs. I used Google Colab in order to train our model (Nvidia T4). The training took no longer than 10 minutes.
<img src="https://github.com/jedrzej-put/Retinal_Blood_Vessels_Segmentation/blob/main/plots/metrics.jpg" width="800" height="800"  title="xD">

## Results
<img src="https://github.com/jedrzej-put/Retinal_Blood_Vessels_Segmentation/blob/main/plots/results.jpg" width="800" height="1000"  title="xD">

### Average metrics for the test set
- Recall: 0.76
- Precision: 0.86



