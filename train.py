import torch.nn.functional as F
import warnings
import numpy as np
from torch import optim, nn
import torchvision
from torch.utils.data import DataLoader
from data.dataloader import TrainDataloader,TestDataloader
from data.metrics import ssim, psnr
from data.utils import Logger
from networks import Generator
from networks import CR
import torch
from data import utils
from torchvision import transforms as tf
from PIL import Image
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings('ignore')
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

batchsize=1
lr=0.0001
n_epochs=100
epoch=1
decay_epoch=50

# 数据增强
transforms_train = [
    tf.ToTensor(),  # range [0, 255] -> [0.0,1.0]
    tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # range [0.0,1.0] -> [-1.0,1.0]
]

transforms_val = [
    tf.ToTensor(),  # range [0, 255] -> [0.0,1.0]
    tf.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # range [0.0,1.0] -> [-1.0,1.0]
]

checkpoints_path="checkpoints/" # train模型保存路径

haze_path = "datasets/indoor/train/haze/"  # 有雾图像路径
clear_path = "datasets/indoor/train/clear/"  # 清晰图像路径

haze_eval_path = "datasets/indoor/test/haze/"  # 有雾图像路径
clear_eval_path = "datasets/indoor/test/clear/"  # 清晰图像路径

if __name__ == '__main__':

    train_sets = TrainDataloader(haze_path, clear_path, transform=transforms_train, unaligned=False)

    val_sets = TestDataloader(haze_eval_path, clear_eval_path, transform=transforms_val)

    train_loader = DataLoader(dataset=train_sets, batch_size=batchsize, shuffle=True, num_workers=0)

    val_loader = DataLoader(dataset=val_sets, batch_size=batchsize, shuffle=False, num_workers=0)

    logger_train = Logger(n_epochs, len(train_loader))
    logger_val   = Logger(n_epochs, len(val_loader))

    G = Generator.Generator().to(device)


    print('The models are initialized successfully!')

    G.train()

    pytorch_total_params = sum(p.numel() for p in G.parameters() if p.requires_grad)
    print("Total_params: ==> {}".format(pytorch_total_params))


    opt_G = optim.Adam(G.parameters(),lr=lr,betas=(0.9,0.999))


    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(opt_G, lr_lambda=utils.LambdaLR(n_epochs, epoch,decay_epoch).step)


    loss_l1 = nn.L1Loss().to(device)  # l1 loss

    loss_contrast = CR.ContrastLoss().to(device)  # contrast loss base on VGG19

    # loss_perceptual=VGG16PerceptualLoss.PerceptualLoss().to(device)  # Perceptual loss base on VGG16


    max_ssim = 0
    max_psnr = 0
    all_ssims=[]
    all_psnrs=[]

################################################ Training ####################################
    for epoch in range(1, n_epochs + 1):

        ssims = []  #每轮清空
        psnrs = []  #每轮清空

        for a, batch in enumerate(train_loader):

            haze  = batch[0].to(device)
            clear = batch[1].to(device)

################################# Training for Generator ###################################

            dehazed = G(haze)

            loss1 = loss_l1(dehazed, clear).to(device)

            loss2 = loss_contrast(dehazed, clear,haze).to(device)

            loss_G = loss1 + loss2 * 0.1

            opt_G.zero_grad()
            loss_G.backward()
            opt_G.step()

            #  pip install visdom -i https://pypi.tuna.tsinghua.edu.cn/simple , python -m visdom.server
            logger_train.log_train({'Loss_G': loss_G,                   # Total generator loss
},                 # Total discriminator loss , multi-discriminator

                                    images={'dehazed': dehazed, 'haze': haze, 'clear': clear})

        # 保存最新的生成器模型
        torch.save(G.state_dict(), checkpoints_path+"G_latest" + ".pth")
        # torch.save(D.state_dict(), checkpoints_path+"D_latest" + ".pth")

        print('Save the latest checkpoints successfully!')

        lr_scheduler_G.step()

################################################ Validating ##########################################

        with torch.no_grad():

            G.eval()

            torch.cuda.empty_cache()

            print("epoch:{}---> Metrics are being evaluated！".format(epoch))

            for a, batch_val in enumerate(val_loader):

                haze_val  = batch_val[0].to(device)
                clear_val = batch_val[1].to(device)

                res_val = G(haze_val)

                psnr1 = psnr(res_val, clear_val)
                ssim1 = ssim(res_val, clear_val).item()

                psnrs.append(psnr1)
                ssims.append(ssim1)

                # pip install visdom , python -m visdom.server
                logger_val.log_val({'PSNR': psnr1,
                                    'SSIM': ssim1},
                                   images={'res_val': res_val,'val': clear_val}) #函数内部进行反归一化处理

            ssim_eval=np.mean(ssims)
            psnr_eval=np.mean(psnrs)

            all_ssims.append(ssim_eval)
            all_psnrs.append(psnr_eval)

            if psnr_eval > max_psnr:

                max_psnr = max(max_psnr, psnr_eval)
                max_ssim = ssim_eval

                torch.save(G.state_dict(), checkpoints_path + "G_Best_PSNR" + ".pth")
                print("Max_psnr：{}, SSIM: {}".format(max_psnr, max_ssim))

            else:

                print("Max_psnr：{}, SSIM: {}".format(max_psnr,max_ssim))
            #
            # if ssim_eval > max_ssim:
            #
            #     max_ssim = max(max_ssim, ssim_eval)
            #
            #     torch.save(G.state_dict(), checkpoints_path + "G_Best_SSIM" + ".pth")
            #
            # else:
            #
            #     print(" The ssim for this epoch are not the best ,max_ssim：{} ".format(max_ssim))

            if epoch % 10 == 0:

                for m in range(len(all_psnrs)):

                    print("epoch {}---> psnr:{}, ssim:{}".format(m + 1, all_psnrs[m], all_ssims[m]))

        epoch += 1









