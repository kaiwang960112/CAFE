import os
import time
import copy
import argparse
import numpy as np
import torch
import torch.nn as nn
from torchvision.utils import save_image
from utils import get_loops, get_dataset, get_network, get_eval_pool, evaluate_synset, get_daparam, \
    match_loss, get_time, TensorDataset, epoch, DiffAugment, ParamDiffAug

import os
import logging
import random
import torch.nn as nn


def build_logger(work_dir, cfgname):
    assert cfgname is not None
    log_file = cfgname + '.log'
    log_path = os.path.join(work_dir, log_file)

    logger = logging.getLogger(cfgname)
    logger.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
    formatter = logging.Formatter('%(asctime)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    handler1 = logging.FileHandler(log_path)
    handler1.setFormatter(formatter)
    logger.addHandler(handler1)

    handler2 = logging.StreamHandler()
    handler2.setFormatter(formatter)
    logger.addHandler(handler2)
    logger.propagate = False

    return logger


def adjust_learning_rate(optimizer, epoch, init_lr):
    """Decay the learning rate based on schedule"""
    lr = init_lr
    for milestone in [1200, 1600, 1800]:
        lr *= 0.5 if epoch >= milestone else 1.
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def criterion_middle(real_feature, syn_feature):
    MSE_Loss = nn.MSELoss(reduction='sum')
    shape_real = real_feature.shape
    real_feature = torch.mean(real_feature.view(10, shape_real[0] // 10, *shape_real[1:]), dim=1)

    shape_syn = syn_feature.shape
    syn_feature = torch.mean(syn_feature.view(10, shape_syn[0] // 10, *shape_syn[1:]), dim=1)

    return MSE_Loss(real_feature, syn_feature)


def main():
    parser = argparse.ArgumentParser(description='Parameter Processing')
    parser.add_argument('--method', type=str, default='DC', help='DC/DSA')
    parser.add_argument('--dataset', type=str, default='CIFAR10', help='dataset')
    parser.add_argument('--model', type=str, default='ConvNet', help='model')
    parser.add_argument('--ipc', type=int, default=1, help='image(s) per class')
    parser.add_argument('--eval_mode', type=str, default='S',
                        help='eval_mode')  # S: the same to training model, M: multi architectures,  W: net width, D: net depth, A: activation function, P: pooling layer, N: normalization layer,
    parser.add_argument('--num_exp', type=int, default=1, help='the number of experiments')
    parser.add_argument('--num_eval', type=int, default=20, help='the number of evaluating randomly initialized models')
    parser.add_argument('--epoch_eval_train', type=int, default=100, help='epochs to train a model with synthetic data')
    parser.add_argument('--Iteration', type=int, default=2000, help='training iterations')
    parser.add_argument('--lr_img', type=float, default=0.1, help='learning rate for updating synthetic images')
    parser.add_argument('--lr_net', type=float, default=0.01, help='learning rate for updating network parameters')
    parser.add_argument('--batch_real', type=int, default=256, help='batch size for real data')
    parser.add_argument('--batch_train', type=int, default=256, help='batch size for training networks')
    parser.add_argument('--init', type=str, default='noise',
                        help='noise/real: initialize synthetic images from random noise or randomly sampled real images.')
    parser.add_argument('--dsa_strategy', type=str, default='None', help='differentiable Siamese augmentation strategy')
    parser.add_argument('--data_path', type=str, default='data', help='dataset path')
    parser.add_argument('--save_path', type=str, default='oi_cifar10_ipc10_watcher_5_v3', help='path to save results')
    parser.add_argument('--dis_metric', type=str, default='ours', help='distance metric')
    parser.add_argument('--fourth_weight', type=float, default=0.1, help='batch size for training networks')
    parser.add_argument('--third_weight', type=float, default=0.1, help='batch size for training networks')
    parser.add_argument('--second_weight', type=float, default=1.0, help='batch size for training networks')
    parser.add_argument('--first_weight', type=float, default=1.0, help='batch size for training networks')
    parser.add_argument('--inner_weight', type=float, default=0.01, help='batch size for training networks')
    parser.add_argument('--lambda_1', type=float, default=0.04, help='break outlooper')
    parser.add_argument('--lambda_2', type=float, default=0.03, help='break innerlooper')
    parser.add_argument('--gpu_id', type=str, default='0', help='dataset path')

    args = parser.parse_args()
    logger = build_logger('.', cfgname=str(args.lambda_1) + "_" + str(args.lambda_2) + "_" + str(
        args.inner_weight) + '_' + str(args.fourth_weight) + '_' + str(args.third_weight) + '_' + str(
        args.second_weight) + '_' + str(args.first_weight) + 'oi_cifar10_dsa_ipc50_watcher_5_v3')
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu_id
    args.outer_loop, args.inner_loop = get_loops(args.ipc)
    # import pdb; pdb.set_trace()
    args.save_path = str(args.lambda_1) + "_" + str(args.lambda_2) + "_" + 'oi_cifar10_ipc10_watcher_5_v3'
    
    args.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    args.dsa_param = ParamDiffAug()
    args.dsa = True if args.method == 'DSA' else False

    if not os.path.exists(args.data_path):
        os.mkdir(args.data_path)

    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    eval_it_pool = np.arange(0, args.Iteration + 1, 100).tolist() if args.eval_mode == 'S' else [
        args.Iteration]  # The list of iterations when we evaluate models and record results.
    channel, im_size, num_classes, class_names, mean, std, dst_train, dst_test, testloader = get_dataset(args.dataset,
                                                                                                         args.data_path)
    model_eval_pool = get_eval_pool(args.eval_mode, args.model, args.model)

    accs_all_exps = dict()  # record performances of all experiments
    for key in model_eval_pool:
        accs_all_exps[key] = []

    data_save = []

    for exp in range(args.num_exp):
        logger.info('================== Exp %d ==================' % exp)
        logger.info('Hyper-parameters: \n', args.__dict__)
        print('Evaluation model pool: ', model_eval_pool)

        ''' organize the real dataset '''
        images_all = []
        labels_all = []
        indices_class = [[] for c in range(num_classes)]

        images_all = [torch.unsqueeze(dst_train[i][0], dim=0) for i in range(len(dst_train))]
        labels_all = [dst_train[i][1] for i in range(len(dst_train))]
        for i, lab in enumerate(labels_all):
            indices_class[lab].append(i)
        images_all = torch.cat(images_all, dim=0)
        labels_all = torch.tensor(labels_all, dtype=torch.long, device=args.device)

        for c in range(num_classes):
            logger.info('class c = %d: %d real images' % (c, len(indices_class[c])))

        def get_images(c, n):  # get random n images from class c
            # import pdb; pdb.set_trace()
            idx_shuffle = np.random.permutation(indices_class[c])[:n]
            return images_all[idx_shuffle].to(args.device)

        for ch in range(channel):
            logger.info('real images channel %d, mean = %.4f, std = %.4f' % (
                ch, torch.mean(images_all[:, ch]), torch.std(images_all[:, ch])))

        ''' initialize the synthetic data '''
        image_syn = torch.randn(size=(num_classes * args.ipc, channel, im_size[0], im_size[1]), dtype=torch.float,
                                requires_grad=True, device=args.device)
        label_syn = torch.tensor([np.ones(args.ipc) * i for i in range(num_classes)], dtype=torch.int,
                                 requires_grad=False, device=args.device).view(-1)  # [0,0,0, 1,1,1, ..., 9,9,9]

        if args.init == 'real':
            logger.info('initialize synthetic data from random real images')
            for c in range(num_classes):
                image_syn.data[c * args.ipc:(c + 1) * args.ipc] = get_images(c, args.ipc).detach().data
        else:
            logger.info('initialize synthetic data from random noise')

        ''' training '''
        optimizer_img = torch.optim.SGD([image_syn, ], lr=args.lr_img, momentum=0.5)  # optimizer_img for synthetic data
        optimizer_img.zero_grad()
        criterion = nn.CrossEntropyLoss().to(args.device)
        criterion_sum = nn.CrossEntropyLoss(reduction='sum').to(args.device)
        logger.info('%s training begins' % get_time())

        for it in range(args.Iteration + 1):
            adjust_learning_rate(optimizer_img, it, args.lr_img)

            ''' Evaluate synthetic data '''
            if it in eval_it_pool:
                for model_eval in model_eval_pool:
                    logger.info(
                        '-------------------------\nEvaluation\nmodel_train = %s, model_eval = %s, iteration = %d' % (
                            args.model, model_eval, it))
                    if args.dsa:
                        args.epoch_eval_train = 1000
                        args.dc_aug_param = None
                        logger.info('DSA augmentation strategy: \n' + args.dsa_strategy)
                        logger.info('DSA augmentation parameters: \n' + str(args.dsa_param.__dict__))
                    else:
                        # This augmentation parameter set is only for DC method. It will be muted when args.dsa is True.
                        args.dc_aug_param = get_daparam(args.dataset, args.model, model_eval,
                                                        args.ipc)  
                        logger.info('DC augmentation parameters: \n' + str(args.dc_aug_param))

                    if args.dsa or args.dc_aug_param['strategy'] != 'none':
                        args.epoch_eval_train = 1000  # Training with data augmentation needs more epochs.
                    else:
                        args.epoch_eval_train = 600

                    accs = []
                    for it_eval in range(args.num_eval):
                        # get a random model
                        net_eval = get_network(model_eval, channel, num_classes, im_size).to(
                            args.device)  
                        # avoid any unaware modification
                        image_syn_eval, label_syn_eval = copy.deepcopy(image_syn.detach()), copy.deepcopy(
                            label_syn.detach())  
                        _, acc_train, acc_test = evaluate_synset(it_eval, net_eval, image_syn_eval, label_syn_eval,
                                                                 testloader, args)
                        accs.append(acc_test)
                    logger.info('Evaluate %d random %s, mean = %.4f std = %.4f\n-------------------------' % (
                        len(accs), model_eval, np.mean(accs), np.std(accs)))
                    # record the final results
                    if it == args.Iteration:  
                        accs_all_exps[model_eval] += accs

                ''' visualize and save '''
                save_name = os.path.join(args.save_path, 'vis_%s_%s_%s_%dipc_exp%d_iter%d.png' % (
                    args.method, args.dataset, args.model, args.ipc, exp, it))
                image_syn_vis = copy.deepcopy(image_syn.detach().cpu())
                for ch in range(channel):
                    image_syn_vis[:, ch] = image_syn_vis[:, ch] * std[ch] + mean[ch]
                image_syn_vis[image_syn_vis < 0] = 0.0
                image_syn_vis[image_syn_vis > 1] = 1.0
                # Trying normalize = True/False may get better visual effects.
                save_image(image_syn_vis, save_name,
                           nrow=args.ipc)  

            ''' Train synthetic data '''
            # get a random model
            net = get_network(args.model, channel, num_classes, im_size).to(args.device)  
            net.train()
            net_parameters = list(net.parameters())
            optimizer_net = torch.optim.SGD(net.parameters(), lr=args.lr_net)  # optimizer_img for synthetic data
            optimizer_net.zero_grad()
            loss_avg = 0
            loss_kai = 0
            loss_middle_item = 0
            args.dc_aug_param = None  # Mute the DC augmentation when training synthetic data.

            # for ol in range(args.outer_loop):
            acc_watcher = list()
            pop_cnt = 0
            acc_test = 0.0
            while True:
                syn_centers = []
                real_feature_concat = []
                real_feature_concat_mm = []
                real_label_concat = []
                img_real_gather = []
                img_syn_gather = []
                lab_real_gather = []
                lab_syn_gather = []

                loss = torch.tensor(0.0).to(args.device)
                for c in range(num_classes):
                    img_real = get_images(c, args.batch_real)
                    lab_real = torch.ones((img_real.shape[0],), device=args.device, dtype=torch.long) * c
                    img_syn = image_syn[c * args.ipc:(c + 1) * args.ipc].reshape(
                        (args.ipc, channel, im_size[0], im_size[1]))
                    lab_syn = torch.ones((args.ipc,), device=args.device, dtype=torch.long) * c

                    if args.dsa:
                        seed = int(time.time() * 1000) % 100000
                        img_real = DiffAugment(img_real, args.dsa_strategy, seed=seed, param=args.dsa_param)
                        img_syn = DiffAugment(img_syn, args.dsa_strategy, seed=seed, param=args.dsa_param)
                    img_real_gather.append(img_real)
                    lab_real_gather.append(lab_real)
                    img_syn_gather.append(img_syn)
                    lab_syn_gather.append(lab_syn)

                img_real_gather = torch.stack(img_real_gather, dim=0).reshape(args.batch_real * 10, 3, 32, 32)
                img_syn_gather = torch.stack(img_syn_gather, dim=0).reshape(args.ipc * 10, 3, 32, 32)
                lab_real_gather = torch.stack(lab_real_gather, dim=0).reshape(args.batch_real * 10)
                lab_syn_gather = torch.stack(lab_syn_gather, dim=0).reshape(args.ipc * 10)

                ####forward#####
                output_real, real_features = net(
                    img_real_gather)
                output_syn, syn_features = net(
                    img_syn_gather)

                loss_middle = args.fourth_weight * criterion_middle(real_features[-1], syn_features[-1]) + args.third_weight * criterion_middle(real_features[-2], syn_features[-2]) + args.second_weight * criterion_middle(real_features[-3], syn_features[-3]) + args.first_weight * criterion_middle(real_features[-4], syn_features[-4])
                loss_real = criterion(output_real, lab_real_gather)
                loss += loss_middle
                loss += loss_real

                last_real_feature = torch.mean(real_features[0].view(10, int(real_features[0].shape[0] / num_classes), real_features[0].shape[1]), dim=1)
                last_syn_feature = torch.mean(syn_features[0].view(10, int(syn_features[0].shape[0] / num_classes), syn_features[0].shape[1]), dim=1)
                output = torch.mm(real_features[0], last_syn_feature.t())
                last_real_feature = torch.mean(
                    last_real_feature.unsqueeze(0).reshape(10, int(last_real_feature.shape[0] / num_classes),
                                                      last_real_feature.shape[1]), dim=1)
                loss_output = criterion_middle(last_syn_feature, last_real_feature) + args.inner_weight * criterion_sum(output, lab_real_gather)
                loss += loss_output

                loss.backward()
                optimizer_img.step()
                optimizer_img.zero_grad()
                loss_avg += loss.item()
                loss_kai += loss_output.item()
                loss_middle_item += loss_middle.item()
                ############ for outloop testing ############

                for c in range(num_classes):
                    img_real_test = get_images(c, 128)
                    lab_real_test = torch.ones((img_real_test.shape[0],), device=args.device, dtype=torch.long) * c
                    prob, _ = net(img_real_test)
                    acc_test += (lab_real_test == prob.max(dim=1)[1]).float().mean()
                acc_test /= num_classes
                acc_watcher.append(acc_test.detach().cpu())
                pop_cnt += 1
                if len(acc_watcher) == 10:
                    if max(acc_watcher) - min(acc_watcher) < args.lambda_1:
                        acc_watcher = list()
                        pop_cnt = 0
                        acc_test = 0.0
                        break
                    else:
                        acc_watcher.pop(0)

                ''' update network '''
                image_syn_train, label_syn_train = copy.deepcopy(image_syn.detach()), copy.deepcopy(
                    label_syn.detach())  # avoid any unaware modification
                dst_syn_train = TensorDataset(image_syn_train, label_syn_train)
                trainloader = torch.utils.data.DataLoader(dst_syn_train, batch_size=args.batch_train, shuffle=True,
                                                          num_workers=0)
                acc_inner_watcher = list()
                acc_syn_inner_watcher = list()
                pop_inner_cnt = 0
                acc_inner_test = 0
                # for il in range(args.inner_loop):
                while (1):
                    inner_loss, inner_acc = epoch('train', trainloader, net, optimizer_net, criterion, args,
                                                  aug=True if args.dsa else False)
                    acc_syn_inner_watcher.append(inner_acc)
                    for c in range(num_classes):
                        img_real_test = get_images(c, 128)
                        lab_real_test = torch.ones((img_real_test.shape[0],), device=args.device, dtype=torch.long) * c
                        prob, _ = net(img_real_test)
                        acc_inner_test += (lab_real_test == prob.max(dim=1)[1]).float().mean()
                    acc_inner_test /= num_classes
                    acc_inner_watcher.append(acc_inner_test.detach().cpu())
                    pop_inner_cnt += 1
                    if len(acc_inner_watcher) == 10:
                        if max(acc_inner_watcher) - min(acc_inner_watcher) > args.lambda_2:
                            acc_inner_watcher = list()
                            acc_syn_inner_watcher = list()
                            pop_inner_cnt = 0
                            acc_inner_test = 0
                            break
                        else:
                            acc_inner_watcher.pop(0)

                    epoch('test', trainloader, net, optimizer_net, criterion, args, aug=True if args.dsa else False)

            loss_avg /= (num_classes * args.outer_loop)

            if it % 10 == 0:
                logger.info('%s iter = %04d, loss = %.4f, loss_kai = %.4f, loss_middle = %.4f' % (
                    get_time(), it, loss_avg, loss_kai, loss_middle_item))

            if it == args.Iteration:  # only record the final results
                data_save.append([copy.deepcopy(image_syn.detach().cpu()), copy.deepcopy(label_syn.detach().cpu())])
                torch.save({'data': data_save, 'accs_all_exps': accs_all_exps, }, os.path.join(args.save_path,
                                                                                               'res_%s_%s_%s_%dipc.pt' % (
                                                                                                   args.method,
                                                                                                   args.dataset,
                                                                                                   args.model,
                                                                                                   args.ipc)))

    logger.info('\n==================== Final Results ====================\n')
    for key in model_eval_pool:
        accs = accs_all_exps[key]
        logger.info('Run %d experiments, train on %s, evaluate %d random %s, mean  = %.2f%%  std = %.2f%%' % (
            args.num_exp, args.model, len(accs), key, np.mean(accs) * 100, np.std(accs) * 100))


if __name__ == '__main__':
    main()
