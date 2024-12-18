import torch
import numpy as np
import random
import os
from util.test import evaluation
from model.model import PretrainedFeatureExtractor, ED, Discriminator
import logging
from argparse import ArgumentParser
from dataset.dataset import OODDataSet
from itertools import cycle
import tqdm


def parse_args():
    
    parser = ArgumentParser(description='Pytorch implemention of Boosting Fine-Grained Visual Anomaly Detection with Coarse-Knowledge-Aware Adversarial Learning')

    parser.add_argument('--dataset', type=str, default='mvtec', help='training dataset')
    
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    
    parser.add_argument('--lr', type=float, default=0.005, help='learning tate')

    parser.add_argument('--epochs', type=int, default=200, help='training epoch')
    
    parser.add_argument('--img_size', type=int, default=32, help='img size')
    
    parser.add_argument('--normal', type=str, default='carpet', help='normal class')
    
    parser.add_argument('--seed', type=int, default=0, help='seed')
    
    parser.add_argument('--labeled_anomaly_class_num', type=int, default=0, help='labeled anomaly class num')
    
    parser.add_argument('--labeled_anomaly_class', type=int, default=1, help='labeled anomaly class')
    
    parser.add_argument('--labeled_anomaly_ratio', type=float, default=0.0, help='labeled anomaly ratio')
    
    parser.add_argument('--log_dir', type=str, default='./log/', help='log dir')
    
    parser.add_argument('--model', type=str, default='resnet18', choices=['resnet18', 'resnet34', 'resnet50', 'wide_resnet50_2', 'wide_resnet101_2', 'resnet152'])
    
    parser.add_argument('--eval_epoch', type=int, default=1, help='every eval_epoch to eval')
    
    parser.add_argument('--layer', nargs='+', type=int, default=[2], help='choose pretrain resnet layer to reconstruct')
    
    parser.add_argument('--d_lr', type=float, default=1e-05, help='discriminator learning rate')
    
    parser.add_argument('--adv_conf', type=float, default=0.02, help='adversial loss conf')
    
    parser.add_argument('--topk', type=int, default=100, help='calculate topk values')
    
    return parser.parse_args()


def get_logger(filename, verbosity=1, name=None):
    level_dict = {0: logging.DEBUG, 1: logging.INFO, 2: logging.WARNING}
    formatter = logging.Formatter(
        "[%(asctime)s][%(filename)s][line:%(lineno)d][%(levelname)s] %(message)s"
    )
    logger = logging.getLogger(name)
    logger.setLevel(level_dict[verbosity])

    fh = logging.FileHandler(filename, "w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    return logger


def setup_seed(seed):
    if seed == -1:
        seed = random.randint(0, 1000)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    return seed

def get_res_str(metrics):
    score_res_str = ""
    for key, value in metrics.items():
        for item, v in value.items():
            score_res_str += "{}_{}: {:.6f} ".format(key, item, v) 
    return score_res_str

def loss_function(a, b):
    cos_loss = torch.nn.CosineSimilarity()
    loss = 0
    for item in range(len(a)):
        loss += torch.mean(1-cos_loss(a[item].view(b[item].shape[0], -1),
                                      b[item].view(b[item].shape[0], -1)))
    return loss

def train(args):
    log_dir = os.path.join(args.log_dir, "lan{:.2f}_acn{}".format(args.labeled_anomaly_ratio,  args.labeled_anomaly_class_num), args.dataset)
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger_filename = os.path.join(log_dir, 'n_{}_a_{}_s_{}'.format(args.normal, args.labeled_anomaly_class, args.seed) + '.txt')
    logger = get_logger(logger_filename)
    logger.info("log file: {}".format(logger_filename))
    logger.info("class: {}".format(args.normal))
    
    print_args(logger, args)
    epochs = args.epochs
    batch_size = args.batch_size
        
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    logger.info("device: {}".format(device))

    dataset = OODDataSet(root='/data2/fqq', dataset=args.dataset, image_size=args.img_size, category=args.normal,
                         labeled_anomaly_ratio=args.labeled_anomaly_ratio,
                         labeled_anomaly_class_num=args.labeled_anomaly_class_num,
                         labeled_anomaly_class=args.labeled_anomaly_class)
    train_dataloader, valid_dataloader, anomaly_dataloader, test_dataloader = dataset.get_data_loader(batch_size=batch_size)

    pfe = PretrainedFeatureExtractor(args.model, layers=args.layer, image_size=args.img_size).to(device)
    for param in pfe.parameters():
        param.requires_grad_(False)
    pfe.eval()
    ae = ED(backbone=args.model, input_channels=pfe.output_channels).to(device)
    discriminator = Discriminator(input_sizes=pfe.output_sizes, input_channels=pfe.output_channels, expansion=pfe.expansion).to(device)
    ae_optimizer = torch.optim.Adam(ae.parameters(), lr=args.lr, betas=(0.5, 0.999))
    discriminator_optimizer = torch.optim.Adam(discriminator.parameters(), lr=args.d_lr, betas=(0.5, 0.999))
    
    alpha = 0.5
    true_label = 0
    fake_label = 1
    for epoch in range(1, epochs+1):
        ae.train()
        discriminator.train()
        dis_loss_list = []
        recon_loss_list = []
        adv_loss_list = []
        ae_loss_list = []
        
        for normal, anomaly in tqdm.tqdm(zip(train_dataloader, cycle(anomaly_dataloader))):
            normal_img = normal[0].to(device)
            if anomaly is not None:
                anomaly_img = anomaly[0].to(device)
            elif args.dataset in ['mvtec', 'visa', 'btad'] and len(normal) == 4:
                anomaly_img = normal[1].to(device)
            else:
                anomaly_img = normal_img[:0]
            anomaly_size = anomaly_img.size(0)
            normal_inputs = pfe(normal_img)
            normal_outputs = ae(normal_inputs)
            
            if anomaly_size > 0: 
                anomaly_inputs = pfe(anomaly_img)
                anomaly_outputs = ae(anomaly_inputs)
                
                outputs = [torch.cat([n_o, a_o]) for n_o, a_o in zip(normal_outputs, anomaly_outputs)]
            else:
                outputs = normal_outputs
                
                
            dis_loss = torch.tensor(0.0).to(device)
            adv_loss = torch.tensor(0.0).to(device)
            
            normal_inputs_detach = [i.detach() for i in normal_inputs]
            if anomaly_size > 0:
                anomaly_inputs_detach = [i.detach() for i in anomaly_inputs]
                
            outputs_detach = [o.detach() for o in outputs]
            if anomaly_size > 0:
                dis_loss = discriminator.calculate_loss(normal_inputs_detach, true_label) + alpha * discriminator.calculate_loss(anomaly_inputs_detach, fake_label) + alpha * discriminator.calculate_loss(outputs_detach, fake_label)
                discriminator_optimizer.zero_grad()
                dis_loss.backward()
                torch.nn.utils.clip_grad_norm_(discriminator.parameters(), 1.0)
                discriminator_optimizer.step()
                    
                adv_loss = discriminator.calculate_loss(outputs, true_label)
            
            recon_loss = loss_function(normal_inputs, normal_outputs)
            ae_loss = recon_loss + args.adv_conf  * adv_loss
            ae_optimizer.zero_grad()
            ae_loss.backward()
            ae_optimizer.step()

            dis_loss_list.append(dis_loss.item())
            ae_loss_list.append(ae_loss.item())
            recon_loss_list.append(recon_loss.item())
            adv_loss_list.append(adv_loss.item())

        logger.info("epoch [{}/{}], dis_loss: {:.6f}, recon_loss:{:.6f}, adv_loss:{:.6f}, ae_loss: {:.6f}".format(epoch, epochs, np.mean(dis_loss_list),
                                                                                                                                 np.mean(recon_loss_list), np.mean(adv_loss_list), np.mean(ae_loss_list),
                                                                                                                                 ))
        if (epoch) % args.eval_epoch == 0:
            if valid_dataloader is not None:
                valid_metrics = evaluation(pfe, ae, valid_dataloader, device, args)
                valid_info = get_res_str(valid_metrics)
                logger.info("Valid: {}".format(valid_info))
                
            metrics = evaluation(pfe, ae, test_dataloader, device, args)
            infostr = get_res_str(metrics)
            logger.info("Test: {}".format(infostr))
       
def print_args(logger, args):
    logger.info('--------args----------')
    for k in list(vars(args).keys()):
        logger.info('{}: {}'.format(k, vars(args)[k]))
    logger.info('--------args----------\n')


if __name__ == '__main__':

    args = parse_args()
    args.seed = setup_seed(args.seed)
    train(args)
   
