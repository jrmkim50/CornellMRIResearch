from Hang.utils_u_groupnorm_pytorchLightning import unetConv3d, unetUp3d, upsampleConv, concatConvUp
import pytorch_lightning as pl
import torch.nn as nn
import torch.optim as optim
import torch
from torch.utils.data import Dataset, DataLoader
import pdb

class unet3d(pl.LightningModule):
    def __init__(self,
                 learning_rate,
                 decay_factor = 0.2,
        feature_scale = 2, 
        n_classes = 6, 
        is_deconv = False, 
        in_channels = 6, 
        is_groupnorm = True,
        is_hpool = True,
        weight = 0
    ):
        super(unet3d, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_groupnorm = is_groupnorm
        self.feature_scale = feature_scale
        self.weight = weight
        self.learning_rate = learning_rate
        self.decay_factor = decay_factor

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv3d(self.in_channels, filters[0], self.is_groupnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size = (2,2,2), padding = (1,1,1))
        self.conv2 = unetConv3d(filters[0], filters[1], self.is_groupnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size = (2,2,2), padding = (1,1,1))
        self.conv3 = unetConv3d(filters[1], filters[2], self.is_groupnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size = (2,2,2), padding = (1,1,1))
        self.conv4 = unetConv3d(filters[2], filters[3], self.is_groupnorm)
        self.maxpool4 = nn.MaxPool3d(kernel_size = (2,2,2), padding = (1,1,1))
        
        self.center = unetConv3d(filters[3], filters[4], self.is_groupnorm)
#         self.center = unetConv3d(filters[2], filters[3], self.is_groupnorm)
        
        # upsampling
        self.up_concat4 = unetUp3d(filters[4]+filters[3], filters[3], self.is_deconv) #remove this addition if is_deconv = true
        self.up_concat3 = unetUp3d(filters[3]+filters[2], filters[2], self.is_deconv)
        self.up_concat2 = unetUp3d(filters[2]+filters[1], filters[1], self.is_deconv)
        self.up_concat1 = unetUp3d(filters[1]+filters[0], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1) 
        
        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

#         center = self.center(maxpool4)
#         center = self.center(maxpool3)

        up4 = self.up_concat4(conv4, self.center(maxpool4))
        up3 = self.up_concat3(conv3, up4)
#         up3 = self.up_concat3(conv3, center)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final
    
    def training_step(self, batch, batch_idx):
        x, y, mask = batch
        y_hat = self(x)
        criterion1 = nn.L1Loss()
        loss = criterion1(y_hat, y) + self.weight * criterion1(mask * y_hat, mask * y)
        tensorboard_logs = {'loss': loss }
        return {'loss': loss, 'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        y_hat = self(x)
        criterion1 = nn.L1Loss()
        loss = criterion1(y_hat, y) + self.weight * criterion1(mask * y_hat, mask * y)
        tensorboard_logs = {'val_loss': loss.cpu()}
        return {'val_loss': loss.cpu(), 'log': tensorboard_logs}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss.cpu()}
        return {'avg_val_loss': avg_loss.cpu(), 'log': tensorboard_logs}
    
    def test_step(self, batch, batch_idx):
        x, y, mask = batch
        y_hat = self(x)
        criterion1 = nn.L1Loss()
        return {'test_loss': criterion1(y_hat, y) + self.weight * criterion1(mask * y_hat, mask * y)}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=self.learning_rate)
#         scheduler = optim.lr_scheduler.CyclicLR(optimizer,base_lr=self.base_lr,max_lr=self.learning_rate,cycle_momentum=False)
        scheduler = {
            'scheduler': optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=self.decay_factor, patience=6),
            'monitor': 'avg_val_loss', # Default: val_loss
            'interval': 'epoch',
            'frequency': 1
        }
        return [optimizer], [scheduler]
    

    
class unet3dpp(pl.LightningModule):
    def __init__(self, 
        feature_scale = 2, 
        n_classes = 6, 
        is_deconv = False, 
        in_channels = 6, 
        is_groupnorm = True,
        is_hpool = True,
        weight = 0
    ):
        super(unet3dpp, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_groupnorm = is_groupnorm
        self.feature_scale = feature_scale
        self.weight = weight

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.pool = nn.MaxPool3d(kernel_size = (2,2,2), padding = (1,1,1))
        self.conv0_0 = unetConv3d(self.in_channels, filters[0], self.is_groupnorm)
        self.conv1_0 = unetConv3d(filters[0], filters[1], self.is_groupnorm)
        self.conv2_0 = unetConv3d(filters[1], filters[2], self.is_groupnorm)
        self.conv3_0 = unetConv3d(filters[2], filters[3], self.is_groupnorm)
        
        self.conv4_0 = unetConv3d(filters[3], filters[4], self.is_groupnorm)
        
        # upsampling
        self.conv0_1 = concatConvUp(filters[1]+filters[0], filters[0], self.is_deconv)
        self.conv1_1 = concatConvUp(filters[2]+filters[1], filters[1], self.is_deconv)
        self.conv2_1 = concatConvUp(filters[3]+filters[2], filters[2], self.is_deconv) 
        
        self.conv0_2 = concatConvUp(filters[1]+filters[0]+filters[0], filters[0], self.is_deconv)
        self.conv1_2 = concatConvUp(filters[2]+filters[1]+filters[1], filters[1], self.is_deconv)
        
        self.conv0_3 = concatConvUp(filters[1]+filters[0]+filters[0]+filters[0], filters[0], self.is_deconv)

        self.conv3_1 = concatConvUp(filters[4]+filters[3], filters[3], self.is_deconv) 
        self.conv2_2 = concatConvUp(filters[3]+filters[2]+filters[2], filters[2], self.is_deconv)
        self.conv1_3 = concatConvUp(filters[2]+filters[1]+filters[1]+filters[1], filters[1], self.is_deconv)
        self.conv0_4 = concatConvUp(filters[1]+filters[0]+filters[0]+filters[0]+filters[0], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)

    def forward(self, inputs):
        x0_0 = self.conv0_0(inputs)
        x1_0 = self.conv1_0(self.pool(x0_0))
        x0_1 = self.conv0_1(x0_0, None, None, None, x1_0)

        x2_0 = self.conv2_0(self.pool(x1_0))
        x1_1 = self.conv1_1(x1_0, None, None, None, x2_0)
        x0_2 = self.conv0_2(x0_0, x0_1, None, None, x1_1)

        x3_0 = self.conv3_0(self.pool(x2_0))
        x2_1 = self.conv2_1(x2_0, None, None, None, x3_0)
        x1_2 = self.conv1_2(x1_0, x1_1, None, None, x2_1)
        x0_3 = self.conv0_3(x0_0, x0_1, x0_2, None, x1_2)
        
        x4_0 = self.conv4_0(self.pool(x3_0))
        x3_1 = self.conv3_1(x3_0, None, None, None, x4_0)
        x2_2 = self.conv2_2(x2_0, x2_1, None, None, x3_1)
        x1_3 = self.conv1_3(x1_0, x1_1, x1_2, None, x2_2)
        x0_4 = self.conv0_4(x0_0, x0_1, x0_2, x0_3, x1_3)
        
        final = self.final(x0_4)

        return final
    
    def training_step(self, batch, batch_idx):
        x, y, mask = batch        
        y_hat = self(x)
        criterion1 = nn.L1Loss()
#         y_hat[y_hat != y_hat] = 0
        loss = criterion1(y_hat, y) + self.weight * criterion1(mask * y_hat, mask * y)
        tensorboard_logs = {'loss': loss }
        return {'loss': loss, 'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_idx):
        x, y, mask = batch
        y_hat = self(x)
        criterion1 = nn.L1Loss()
        loss = criterion1(y_hat, y) + self.weight * criterion1(mask * y_hat, mask * y)
        tensorboard_logs = {'val_loss': loss}
        return {'val_loss': loss, 'log': tensorboard_logs}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss.cpu()}
        return {'avg_val_loss': avg_loss.cpu(), 'log': tensorboard_logs}
    
    def test_step(self, batch, batch_idx):
        x, y, mask = batch
        y_hat = self(x)
        criterion1 = nn.L1Loss()
        return {'test_loss': criterion1(y_hat, y) + self.weight * criterion1(mask * y_hat, mask * y)}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=0.01)
#         scheduler = optim.lr_scheduler.CyclicLR(optimizer,base_lr=3e-10,max_lr=3e-2,cycle_momentum=False)
        scheduler = optim.lr_scheduler.StepLR(optimizer, 40, gamma = 0.2)
        return [optimizer], [scheduler]
    
class unet3dDin(pl.LightningModule):
    def __init__(self, 
        feature_scale = 2, 
        n_classes = 6, 
        is_deconv = False, 
        in_channels = 6, 
        is_groupnorm = True,
        is_hpool = True
    ):
        super(unet3dDin, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_groupnorm = is_groupnorm
        self.feature_scale = feature_scale

        filters = [64, 128, 256, 512, 1024]
        filters = [int(x / self.feature_scale) for x in filters]

        # downsampling
        self.conv1 = unetConv3d(self.in_channels, filters[0], self.is_groupnorm)
        self.maxpool1 = nn.MaxPool3d(kernel_size = (2,2,2), padding = (1,1,1))
        self.conv2 = unetConv3d(filters[0], filters[1], self.is_groupnorm)
        self.maxpool2 = nn.MaxPool3d(kernel_size = (2,2,2), padding = (1,1,1))
        self.conv3 = unetConv3d(filters[1], filters[2], self.is_groupnorm)
        self.maxpool3 = nn.MaxPool3d(kernel_size = (2,2,2), padding = (1,1,1))
        self.conv4 = unetConv3d(filters[2], filters[3], self.is_groupnorm)
        self.maxpool4 = nn.MaxPool3d(kernel_size = (2,2,2), padding = (1,1,1))
        
        self.center = unetConv3d(filters[3], filters[4], self.is_groupnorm)
        
        # upsampling
        self.up_concat4 = unetUp3d(filters[4]+filters[3], filters[3], self.is_deconv) #remove this addition if is_deconv = true
        self.up_concat3 = unetUp3d(filters[3]+filters[2], filters[2], self.is_deconv)
        self.up_concat2 = unetUp3d(filters[2]+filters[1], filters[1], self.is_deconv)
        self.up_concat1 = unetUp3d(filters[1]+filters[0], filters[0], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[0], n_classes, 1)

    def forward(self, inputs):
        
        conv1 = self.conv1(inputs)
        maxpool1 = self.maxpool1(conv1)

        conv2 = self.conv2(maxpool1)
        maxpool2 = self.maxpool2(conv2)

        conv3 = self.conv3(maxpool2)
        maxpool3 = self.maxpool3(conv3)

        conv4 = self.conv4(maxpool3)
        maxpool4 = self.maxpool4(conv4)

        center = self.center(maxpool4)

        up4 = self.up_concat4(conv4, center)
        up3 = self.up_concat3(conv3, up4)
        up2 = self.up_concat2(conv2, up3)
        up1 = self.up_concat1(conv1, up2)

        final = self.final(up1)

        return final, center
    
    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat, _ = self(x)
        criterion1 = nn.L1Loss()
        loss = criterion1(y_hat, y)
        tensorboard_logs = {'loss': loss }
        return {'loss': loss, 'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat, _ = self(x)
        criterion1 = nn.L1Loss()
        loss = criterion1(y_hat, y)
        tensorboard_logs = {'val_loss': loss.cpu()}
        return {'val_loss': loss.cpu(), 'log': tensorboard_logs}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss.cpu()}
        return {'avg_val_loss': avg_loss.cpu(), 'log': tensorboard_logs}
    
    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        y_hat, _ = self(x)
        criterion1 = nn.L1Loss()
        return {'test_loss': criterion1(y_hat, y)}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.CyclicLR(optimizer,base_lr=3e-10,max_lr=3e-2,cycle_momentum=False)
        return [optimizer], [scheduler]
    
class upsampleBranch(pl.LightningModule):
    def __init__(self, encoder_model,
        feature_scale = 2, 
        n_classes = 6, 
        is_deconv = False, 
        in_channels = 6, 
        is_groupnorm = True,
        is_hpool = True,
    ):
        super(upsampleBranch, self).__init__()
        self.is_deconv = is_deconv
        self.in_channels = in_channels
        self.is_groupnorm = is_groupnorm
        self.feature_scale = feature_scale
        self.model = encoder_model

        filters = [1024, 512, 256, 128, 64]
        filters = [int(x / self.feature_scale) for x in filters]
        
        image_sizes = [(17,17,5), (33,33,9), (65,65,17), (128,128,32)]
        
        # upsampling
        self.up_concat4 = upsampleConv(filters[0], filters[1], image_sizes[0], self.is_deconv)
        self.up_concat3 = upsampleConv(filters[1], filters[2], image_sizes[1], self.is_deconv)
        self.up_concat2 = upsampleConv(filters[2], filters[3], image_sizes[2], self.is_deconv)
        self.up_concat1 = upsampleConv(filters[3], filters[4], image_sizes[3], self.is_deconv)

        # final conv (without any concat)
        self.final = nn.Conv3d(filters[4], n_classes, 1)

    def forward(self, center):
        
        up4 = self.up_concat4(center)
        up3 = self.up_concat3(up4)
        up2 = self.up_concat2(up3)
        up1 = self.up_concat1(up2)

        final = self.final(up1)

        return final
    
    def training_step(self, batch, batch_idx):
        x, y, _ = batch
        _, center = self.model(x)
        y_hat = self(center)
        criterion1 = nn.L1Loss()
        loss = criterion1(y_hat, y)
        tensorboard_logs = {'loss': loss }
        return {'loss': loss, 'log': tensorboard_logs}
    
    def validation_step(self, batch, batch_idx):
        x, y, _ = batch
        _, center = self.model(x)
        y_hat = self(center)
        criterion1 = nn.L1Loss()
        loss = criterion1(y_hat, y)
        tensorboard_logs = {'val_loss': loss.cpu()}
        return {'val_loss': loss.cpu(), 'log': tensorboard_logs}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_val_loss': avg_loss.cpu()}
        return {'avg_val_loss': avg_loss.cpu(), 'log': tensorboard_logs}
    
    def test_step(self, batch, batch_idx):
        x, y, _ = batch
        _, center = self.model(x)
        y_hat = self(center)
        criterion1 = nn.L1Loss()
        return {'test_loss': criterion1(y_hat, y)}

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x['test_loss'] for x in outputs]).mean()
        tensorboard_logs = {'avg_test_loss': avg_loss}
        return {'avg_test_loss': avg_loss, 'log': tensorboard_logs}

    def configure_optimizers(self):
        optimizer = optim.AdamW(self.parameters(), lr=0.001)
        scheduler = optim.lr_scheduler.CyclicLR(optimizer,base_lr=3e-10,max_lr=3e-2,cycle_momentum=False)
        return [optimizer], [scheduler]