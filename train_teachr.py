import argparse
import os
from os.path import join
from dataset.data import myDataloader
import scipy.misc
from network.loss import *
from network.SSIM import SSIM
import random
import re
from scheduler import CyclicCosineDecayLR
import imageio
from network.Teachr import *
import numpy as np

parser = argparse.ArgumentParser(description="PyTorch Train")
parser.add_argument("--batchSize", type=int, default=8, help="Training batch size")
parser.add_argument("--start_training_step", type=int, default=2, help="Training step")
parser.add_argument("--nEpochs", type=int, default=30, help="Number of epochs to train")
parser.add_argument("--lrG", type=float, default=1e-4, help="Learning rate, default=1e-4")
parser.add_argument("--lrD", type=float, default=1e-4, help="Learning rate, default=1e-4")
parser.add_argument("--step", type=int, default=10, help="Change the learning rate for every 30 epochs")
parser.add_argument("--start-epoch", type=int, default=1, help="Start epoch from 1")
parser.add_argument("--lr_decay", type=float, default=0.1, help="Decay scale of learning rate, default=0.5")
parser.add_argument("--resume", default="", type=str, help="Path to checkpoint (default: none)")
parser.add_argument("--scale", default=4, type=int, help="Scale factor, Default: 4")
parser.add_argument("--lambda_db", type=float, default=0.5, help="Weight of deblurring loss, default=0.5")
parser.add_argument("--gated", type=bool, default=True, help="Activated gate module")
parser.add_argument("--isTest", type=bool, default=False, help="Test or not")
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def mkdir_steptraing():
    root_folder = os.path.abspath('.')
    models_folder = join(root_folder, 'models/modelW')
    step1_folder, step2_folder, step3_folder = join(models_folder,'1'), join(models_folder,'2'), join(models_folder, '3')
    isexists = os.path.exists(step1_folder) and os.path.exists(step2_folder) and os.path.exists(step3_folder)
    if not isexists:
        os.makedirs(step1_folder)
        os.makedirs(step2_folder)
        os.makedirs(step3_folder)
        print("===> Step training models store in models/1 & /2 & /3.")


def mkdir_model(path):
    root_folder = os.path.abspath('.')
    models_folder = join(root_folder, path)
    isexists = os.path.exists(models_folder)
    if not isexists:
        os.makedirs(models_folder)

        print("===> Step training models store in models/1 & /2 & /3.")


def is_hdf5_file(filename):
    return any(filename.endswith(extension) for extension in [".h5"])


def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)


def which_trainingstep_epoch(resume):
    trainingstep = "".join(re.findall(r"\d", resume)[0])
    start_epoch = "".join(re.findall(r"\d", resume)[1:])
    return int(trainingstep), int(start_epoch)

class trainer_2:
    def __init__(self, train_gen, step, numD=4):
        super(trainer_2, self).__init__()

        self.numd = numD
        self.step = step
        self.trainloader = train_gen
        self.modelG = endeFUINT2_1().to(device)
        self.optimizer_G = torch.optim.Adam(filter(lambda p: p.requires_grad, self.modelG.parameters()), lr=1e-4,
                                            betas=(0.9, 0.999))
        self.scheduler_lr = CyclicCosineDecayLR(self.optimizer_G, init_interval=20, min_lr=5e-8, restart_interval=20, restart_lr=1e-4)

        self.criterion = nn.L1Loss().to(device)
        self.VGGLoss = VGGLoss().to(device)
        self.ssim = SSIM().to(device)

        self.weight = [1./4, 1./2, 1]

    def opt_G1(self, Jx, fake):
        self.optimizer_G.zero_grad()

        g_loss_MSE0 = self.criterion(fake, Jx.detach())

        loss = g_loss_MSE0
        loss.backward()

        self.optimizer_G.step()

        return g_loss_MSE0, g_loss_MSE0, g_loss_MSE0

    def adjust_learning_rate(self, epoch):
        lrG = 1e-5 * (0.1 ** ((epoch+1) // 60))
        print(lrG)
        for param_group in self.optimizer_G.param_groups:
            param_group['lr'] = lrG

    def checkpoint(self, epoch):
        path = "models/Teacher/".format(10)
        mkdir_model(path)
        model_out_path = path + "/model_FULL_{}.pkl".format(epoch)
        torch.save(self.modelG, model_out_path)

        print("===>Checkpoint saved to {}".format(model_out_path))

    def train(self, epoch, train_gen):
        path = './log/log14.txt'
        with open(path, 'a') as f:
            f.write("===++++++++++++++++++++++++++++++++++++++===========")

        self.trainloader = train_gen

        self.scheduler_lr.step(epoch=epoch-1)

        print(self.optimizer_G.param_groups[0]["lr"])

        epoch_loss1 = 0
        epoch_loss2 = 0
        epoch_loss3 = 0
        for iteration, (Ix, Jx) in enumerate(self.trainloader):
            Ix = Ix.to(device)
            Jx = Jx.to(device)

            fake, _ = self.modelG(Jx)

            loss1, loss2, loss3 = self.opt_G1(Jx, fake)

            epoch_loss1 += float(loss1)
            epoch_loss2 += float(loss2)
            epoch_loss3 += float(loss3)

            if iteration % 100 == 0:
                print(
                    "===> Epoch[{}]({}/{}): Loss{:.4f};".format(epoch, iteration, len(trainloader), loss2.cpu()))

                Ix_cc = fake# modelD(Detail_I) #+ Ix[:, 6:9, :, :] modelD(Detail_I)#
                Ix_cc = Ix_cc.clamp(0, 1)
                Ix_cc = Ix_cc[0].permute(1, 2, 0).detach().cpu().numpy()


                Ix = Ix.clamp(0, 1)
                Ix = Ix[0].permute(1, 2, 0).detach().cpu().numpy()
                Ix_cc = np.hstack([Ix, Ix_cc])

                Jx = Jx.clamp(0, 1)
                Jx = Jx[0].permute(1, 2, 0).detach().cpu().numpy()
                Ix_cc = np.hstack([Ix_cc, Jx])

                #print(Ix_cc.shape)
                imageio.imsave('./results' + '/' + str((epoch - 1) * 100 + iteration / 100) + '.png', np.uint8(Ix_cc*255))

                print("MSE:{:4f},MSSIM:{:4f},VGG:{:4f},VGG:{:4f}".format(loss1, loss2, loss3, loss1))

        print("===>Epoch{} Complete: Avg loss is :Loss1:{:4f},Loss2:{:4f},Loss3:{:4f},  ".format(epoch, epoch_loss1 / len(trainloader), epoch_loss2 / len(trainloader), epoch_loss3 / len(trainloader)))

        path = './log/log13.txt'
        with open(path, 'a') as f:
            f.write("===>Epoch{} Complete: Avg loss is :Loss1:{:4f},Loss2:{:4f},Loss3:{:4f},  ".format(epoch, epoch_loss1 / len(trainloader), epoch_loss2 / len(trainloader), epoch_loss3 / len(trainloader)))


opt = parser.parse_args()
opt.seed = random.randint(1, 10000)
torch.manual_seed(opt.seed)
torch.cuda.manual_seed(opt.seed)
trainloader = myDataloader().getLoader()

for i in range(1, 2):
    print("===> Loading model and criterion")

    trainModel = trainer_2(trainloader, step=i, numD=1)

    for epoch in range(1, 21):
        print("Step {}:-------------------------------".format(i))
        trainModel.checkpoint(epoch)
        trainModel.train(epoch, trainloader)
        trainModel.checkpoint(epoch)