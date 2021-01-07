from __future__ import print_function
import argparse
import os
import time
import torch
import numpy as np
import scipy.misc
from PIL import Image

import cv2
from skimage.measure import compare_psnr as psnr
from skimage.measure import compare_ssim as ski_ssim
import matplotlib.pyplot as plt
import imageio

def norm_ip(img, min, max):
    img.clamp_(min=min, max=max)
    img.add_(-min).div_(max - min)

    return img


def norm_range(t, range):
    if range is not None:
        norm_ip(t, range[0], range[1])
    else:
        norm_ip(t, t.min(), t.max())
    return norm_ip(t, t.min(), t.max())


os.environ['CUDA_VISIBLE_DEVICES'] = '0'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def test(Ix, model, dense=True):
    med_time = []

    with torch.no_grad():
        Ix = Ix.to(device)

        start_time = time.perf_counter()  # -------------------------begin to deal with an image's time

        f5, Ix_cc, conf = model(Ix)
        '''
        tensor = norm_range(Ix_cc[0].cpu(), None)
        ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).numpy()
        '''
        # modify
        # tensor = norm_range(torch.squeeze(Ix_cc), None)
        # ndarr = tensor.mul(255).clamp(0, 255).byte().permute(1, 2, 0).cpu().numpy()

    # Ix_cc = Ix_cc.clamp(0, 1)
    # Ix_cc = np.uint8(255 * Ix_cc[0].permute(1, 2, 0).cpu().numpy())
    # torch.cuda.synchronize()  # wait for CPU & GPU time syn

    # evalation_time = time.perf_counter() - start_time  # ---------finish an image
    # med_time.append(evalation_time)

    return Ix_cc


def normImge(image, num=1.):
    if len(image.shape) > 2:
        for i in range(3):
            img = image[:, :, i]
            max = np.max(img)
            min = np.min(img)
            image[:, :, i] = (img - min) / (max - min + 1e-8)
    else:
        max = np.max(image)
        min = np.min(image)
        image = (image - min) / (max - min + 1e-8) * num
    return image


# models/modelOut/5/GFNMS2_2_epoch_14.pklmodels/modelOut/10/GFNMS2_2_epoch_11.pkl
def print_network(net):
    if isinstance(net, list):
        net = net[0]
    num_params = 0
    for param in net.parameters():
        num_params += param.numel()
    print(net)
    print('Total number of parameters: %d' % num_params)

import torch.nn.functional as F
def test_student(model):
    im_path = '/home/hm/hm/PycharmProjects/data/ITS/SOTS (Synthetic Objective Testing Set)/SOTS/indoor/hazy/'
    gt_path = '/home/hm/hm/PycharmProjects/data/ITS/SOTS (Synthetic Objective Testing Set)/SOTS/indoor/gt/'

    ave_psnr = 0.0
    ave_ssim = 0.0

    for path, subdirs, files in os.walk(im_path):
        sorted(files)
        for i in range(len(files)):
            nameA = files[i]
            hazyName = im_path + nameA
            print(hazyName)
            # print(hazyName)
            Ix = np.array(Image.open(hazyName).convert('RGB')) / 255.
            # Jx = np.array(Image.open(gt_path + nameA.split('_')[0] + '_im0.png').convert('RGB'))
            Jx = np.array(Image.open(gt_path + nameA.split('_')[0] + '.png').convert('RGB'))
            # Jx = np.array(Image.open(gt_path + nameA).convert('RGB'))

            H = Ix.shape[1]
            W = Ix.shape[0]
            if H < 1200:
                img_H = H // 4 * 4
                img_W = W // 4 * 4
            else:
                img_H = H // 4 * 4
                img_W = W // 4 * 4

            Ixin = cv2.resize(Ix, (img_H, img_W))

            Ixin = Ixin.transpose([2, 0, 1])
            Ixin = torch.from_numpy(Ixin).float()

            testI = Ixin.unsqueeze(0)
            model = model.to(device)
            testI = testI.to(device)

            with torch.no_grad():
                Ix_cc = model(testI)[0]

            res = Ix_cc[0].data.cpu().numpy()
            res[res > 1] = 1
            res[res < 0] = 0
            res *= 255
            res = res.astype(np.uint8)
            out = res.transpose((1, 2, 0))

            out = cv2.resize(out, (H, W)).astype(np.uint8)

            PSNR = psnr(out, Jx.astype(np.uint8), data_range=255)
            SSIM = ski_ssim(out, Jx.astype(np.uint8), data_range=255, multichannel=True)
            ave_psnr += PSNR
            ave_ssim += SSIM

            imageio.imsave('./res/' + nameA,
                               np.hstack([out.astype(np.uint8)]))

            print(nameA, PSNR, SSIM)

        print(ave_psnr / len(files), ave_ssim / len(files))


def test_real(model):
    im_path = '//home/hm/hm/PycharmProjects/data/haze/real/'

    for path, subdirs, files in os.walk(im_path):
        for i in range(len(files)):
            nameA = files[i]
            hazyName = im_path + nameA
            print(hazyName)
            Ix = np.array(Image.open(hazyName).convert('RGB')) / 255.

            H = Ix.shape[1]
            W = Ix.shape[0]

            if max(W, H) > 1200:
                img_H = H // 16 * 4
                img_W = W // 16 * 4
            elif max(W, H) > 400:
                img_H = H // 4 * 4
                img_W = W // 4 * 4
            else:
                img_H = H // 4 * 4
                img_W = W // 4 * 4

            Ixin = cv2.resize(Ix, (img_H, img_W))

            Ixin = Ixin.transpose([2, 0, 1])
            Ixin = torch.from_numpy(np.ascontiguousarray(Ixin)).float()

            testI = torch.cat([Ixin], 0).unsqueeze(0)
            model = model.to(device)
            testI = testI.to(device)

            with torch.no_grad():
                Ix_cc = model(testI)[0][:, 0:3, :,:]
            res = Ix_cc[0].data.cpu().numpy()
            res[res > 1] = 1
            res[res < 0] = 0
            res *= 255
            res = res.astype(np.uint8)
            out = res.transpose((1, 2, 0))

            out = cv2.resize(out, (H, W))

            path = './result_ITS/'
            # path = im_path + '/result/'
            isexists = os.path.exists(path)
            if not isexists:
                os.makedirs(path)

            # scipy.misc.imsave('//home/hm/disk/hm/PycharmProjects/HazeRD/UMDNReal/' + '/' + nameA, out.astype(np.uint8))
            # scipy.misc.imsave(path + '/' + nameA, np.hstack([out.astype(np.uint8), (Ix*255).astype(np.uint8)]))
            imageio.imsave(path + '/' + nameA, out.round().astype(np.uint8))

def test_clear2hazy(model):
    im_path = '//home/hm/hm/PycharmProjects/data/real/Flickr2K/Flickr2K_LR_bicubic/X2/'

    for path, subdirs, files in os.walk(im_path):
        sorted(files)
        for i in range(1):
            nameA = files[7]
            hazyName = im_path + '000004x2.png'
            print(hazyName)
            Ix = np.array(Image.open(hazyName).convert('RGB')) / 255.

            H = Ix.shape[1]
            W = Ix.shape[0]

            if max(W, H) > 1200:
                img_H = H // 16 * 4
                img_W = W // 16 * 4
            elif max(W, H) > 400:
                img_H = H // 4 * 4
                img_W = W // 4 * 4
            else:
                img_H = H // 4 * 4
                img_W = W // 4 * 4

            Ixin = cv2.resize(Ix, (img_H, img_W))

            Ixin = Ixin.transpose([2, 0, 1])
            Ixin = torch.from_numpy(np.ascontiguousarray(Ixin)).float()


            testI = torch.cat([Ixin], 0).unsqueeze(0)
            model = model.to(device)
            testI = testI.to(device)


            for t in range(10):
                trans = torch.ones((testI.shape[0], 1, testI.shape[2], testI.shape[3])).cuda() * (t / 10)
                for j in range(10):
                    Degree = torch.ones((testI.shape[0], 3, testI.shape[2], testI.shape[3])).cuda() * (j / 10)

                    with torch.no_grad():
                        Ix_cc, _ = model(torch.cat((testI, Degree, trans), 1))
                        Ix_cc = F.tanh(Ix_cc)

                    res = Ix_cc[0].data.cpu().numpy()
                    res[res > 1] = 1
                    res[res < 0] = 0
                    res *= 255
                    res = res.astype(np.uint8)
                    out = res.transpose((1, 2, 0))

                    out = cv2.resize(out, (H, W))

                    path = './result_ITS/'
                    # path = im_path + '/result/'
                    isexists = os.path.exists(path)
                    if not isexists:
                        os.makedirs(path)

                    imageio.imsave(path + '/' + str(t)+ '_' + str(j)+'.png', out.round().astype(np.uint8))

def test_teacher(model):
    im_path = '/home/hm/hm/PycharmProjects/data/ITS/SOTS (Synthetic Objective Testing Set)/SOTS/indoor/hazy/'
    gt_path = '/home/hm/hm/PycharmProjects/data/ITS/SOTS (Synthetic Objective Testing Set)/SOTS/indoor/gt/'

    ave_psnr = 0.0
    ave_ssim = 0.0

    for path, subdirs, files in os.walk(im_path):
        sorted(files)
        for i in range(len(files)):
            nameA = files[i]
            hazyName = im_path + nameA

            Jx = np.array(Image.open(gt_path + nameA.split('_')[0] + '.png').convert('RGB')) / 255.

            Ixin = Jx.transpose([2, 0, 1])
            Ixin = torch.from_numpy(Ixin).float()

            testI = Ixin.unsqueeze(0)
            model = model.to(device)
            testI = testI.to(device)

            with torch.no_grad():
                Ix_cc =model(testI)[0]

            res = Ix_cc[0].clamp(0, 1)
            res = res.data.cpu().numpy()

            res *= 255
            res = res.astype(np.uint8)
            out = res.transpose((1, 2, 0))

            PSNR = psnr(out, Jx * 255., data_range=255)
            SSIM = ski_ssim(out, Jx * 255., data_range=255, multichannel=True)
            ave_psnr += PSNR
            ave_ssim += SSIM

            print(nameA, PSNR, SSIM)

        print(ave_psnr / len(files), ave_ssim / len(files))


if __name__ =="__main__":
    # teacherModel = torch.load('./models/Teacher/model_FULL_16.pkl')
    # print_network(teacherModel)
    # test_teacher(teacherModel)

    studentModel = torch.load('./models/Student//model_SmallFollow_60.pkl')
    # print_network(studentModel)
    test_student(studentModel)
