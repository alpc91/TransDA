import argparse
import os
import datetime
import logging
import time, itertools
import math
import random
import numpy as np
from collections import OrderedDict
import torch
import torch.nn.functional as F

from core.configs import cfg
from core.datasets import build_dataset
from core.models import build_adversarial_discriminator_cls, build_adversarial_discriminator_bin, build_feature_extractor, build_classifier
from core.solver import adjust_learning_rate
from core.models.checkpoint import load_checkpoint
from core.utils.misc import mkdir, AverageMeter, intersectionAndUnionGPU, resize
from core.utils.logger import setup_logger
from core.utils.metric_logger import MetricLogger


def get_share_weight(domain_out, before_softmax,  class_temperature=10.0):
    before_softmax = before_softmax / class_temperature
    after_softmax = F.softmax(before_softmax, dim=1)
    
    entropy = torch.sum(- after_softmax * torch.log(after_softmax + 1e-10), dim=1, keepdim=True)
    entropy_norm = entropy / np.log(after_softmax.size(1))
    weight = entropy_norm - domain_out
    return weight.detach()

def normalize_weight(x):
    min_val = x.min()
    max_val = x.max()
    x = (x - min_val) / (max_val - min_val)
    return x.detach()

def strip_prefix_if_present(state_dict, prefix):
    keys = sorted(state_dict.keys())
    if not all(key.startswith(prefix) for key in keys):
        return state_dict
    stripped_state_dict = OrderedDict()
    for key, value in state_dict.items():
        stripped_state_dict[key.replace(prefix, "")] = value
    return stripped_state_dict

def soft_label_cross_entropy(pred, soft_label, pixel_weights=None):
    N, C, H, W = pred.shape
    loss = -soft_label.float()*torch.log(pred)
    if pixel_weights is None:
        return torch.mean(torch.sum(loss, dim=1))
    return torch.mean(pixel_weights*torch.sum(loss, dim=1))

def train(cfg, local_rank, distributed):
    logger = logging.getLogger("TransDA.trainer")
    logger.info("Start training")
    device = torch.device(cfg.MODEL.DEVICE)
    batch_size = cfg.SOLVER.BATCH_SIZE
    
    feature_extractor = build_feature_extractor(cfg)
    feature_extractor.init_weights()
    feature_extractor.to(device)
    load_checkpoint(feature_extractor, cfg.MODEL.WEIGHTS, strict=False)

    feature_extractor_ema = build_feature_extractor(cfg)
    feature_extractor_ema.init_weights()
    feature_extractor_ema.to(device)
    load_checkpoint(feature_extractor_ema, cfg.MODEL.WEIGHTS, strict=False)
    
    classifier, aux = build_classifier(cfg)
    classifier.init_weights()
    aux.init_weights()
    classifier.to(device)
    aux.to(device)

    classifier_ema, aux_ema = build_classifier(cfg)
    classifier_ema.init_weights()
    aux_ema.init_weights()
    classifier_ema.to(device)
    aux_ema.to(device)

    decay = cfg.decay

    if cfg.SOLVER.DIS == 'binary':
        model_D = build_adversarial_discriminator_bin(cfg)
    else:
        model_D = build_adversarial_discriminator_cls(cfg)
    model_D.to(device)

    model_Dis = build_adversarial_discriminator_bin(cfg)
    model_Dis.to(device)

    if local_rank==0:
        print(feature_extractor)
        print(classifier)
        print(aux)
        print(model_D)
        print(model_Dis)
    

    if distributed:
        batch_size = int(cfg.SOLVER.BATCH_SIZE / torch.distributed.get_world_size())
        classifier = torch.nn.SyncBatchNorm.convert_sync_batchnorm(classifier)
        aux = torch.nn.SyncBatchNorm.convert_sync_batchnorm(aux)
        pg1 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        feature_extractor = torch.nn.parallel.DistributedDataParallel(
            feature_extractor, device_ids=[local_rank], output_device=local_rank, 
            find_unused_parameters=True, process_group=pg1
        )
        pg2 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        classifier = torch.nn.parallel.DistributedDataParallel(
            classifier, device_ids=[local_rank], output_device=local_rank, 
            find_unused_parameters=True, process_group=pg2
        )
        pg3 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        aux = torch.nn.parallel.DistributedDataParallel(
            aux, device_ids=[local_rank], output_device=local_rank, 
            find_unused_parameters=True, process_group=pg3
        )
        pg4 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        model_D = torch.nn.parallel.DistributedDataParallel(
            model_D, device_ids=[local_rank], output_device=local_rank, 
            find_unused_parameters=True, process_group=pg4
        )
        pg5 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        model_Dis = torch.nn.parallel.DistributedDataParallel(
            model_Dis, device_ids=[local_rank], output_device=local_rank, 
            find_unused_parameters=True, process_group=pg5
        ) 
        pg6 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        feature_extractor_ema = torch.nn.parallel.DistributedDataParallel(
            feature_extractor_ema, device_ids=[local_rank], output_device=local_rank, 
            find_unused_parameters=True, process_group=pg6
        )
        pg7 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        classifier_ema = torch.nn.parallel.DistributedDataParallel(
            classifier_ema, device_ids=[local_rank], output_device=local_rank, 
            find_unused_parameters=True, process_group=pg7
        )
        pg8 = torch.distributed.new_group(range(torch.distributed.get_world_size()))
        aux_ema = torch.nn.parallel.DistributedDataParallel(
            aux_ema, device_ids=[local_rank], output_device=local_rank, 
            find_unused_parameters=True, process_group=pg8
        )
        torch.autograd.set_detect_anomaly(True)
        torch.distributed.barrier()
    
    optimizer_fea = torch.optim.AdamW(feature_extractor.parameters(), lr=cfg.SOLVER.BASE_LR, betas=(0.9, 0.999), weight_decay=0.01)
    optimizer_cls = torch.optim.AdamW(itertools.chain(classifier.parameters(),aux.parameters()), lr=cfg.SOLVER.BASE_LR, betas=(0.9, 0.999), weight_decay=0.01)
    optimizer_D = torch.optim.Adam(itertools.chain(model_D.parameters(), model_Dis.parameters()), lr=cfg.SOLVER.BASE_LR_D, betas=(0.9, 0.99))    

    output_dir = cfg.OUTPUT_DIR
    pth_dir = output_dir.replace('results','pth')
    if pth_dir:
        mkdir(pth_dir)

    save_to_disk = local_rank == 0

    iteration = 0

    src_train_data = build_dataset(cfg, mode='train', is_source=True)
    tgt_train_data = build_dataset(cfg, mode='train', is_source=False)

    if distributed:
        src_train_sampler = torch.utils.data.distributed.DistributedSampler(src_train_data)
        tgt_train_sampler = torch.utils.data.distributed.DistributedSampler(tgt_train_data)
    else:
        src_train_sampler = None
        tgt_train_sampler = None
    
    src_train_loader = torch.utils.data.DataLoader(src_train_data, batch_size=batch_size, shuffle=(src_train_sampler is None), num_workers=4, pin_memory=True, sampler=src_train_sampler, drop_last=True)
    tgt_train_loader = torch.utils.data.DataLoader(tgt_train_data, batch_size=batch_size, shuffle=(tgt_train_sampler is None), num_workers=4, pin_memory=True, sampler=tgt_train_sampler, drop_last=True)

    criterion = torch.nn.CrossEntropyLoss(ignore_index=cfg.INPUT.IGNORE_LABEL)
    bce_loss = torch.nn.BCELoss()
    test_stats = []

    max_iters = cfg.SOLVER.MAX_ITER
    logger.info("Start training")
    meters = MetricLogger(delimiter="  ")
    feature_extractor.train()
    feature_extractor_ema.train()
    classifier.train()
    classifier_ema.train()
    aux.train()
    aux_ema.train()
    model_D.train()
    model_Dis.train()
    start_training_time = time.time()
    end = time.time()


    for i, ((src_input, src_label, src_name), (tgt_input, tgt_label, _)) in enumerate(zip(src_train_loader, tgt_train_loader)):
        data_time = time.time() - end
        current_lr = adjust_learning_rate(cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR, iteration, 1500, 1e-6, max_iters)
        current_lr_D = adjust_learning_rate(cfg.SOLVER.LR_METHOD, cfg.SOLVER.BASE_LR_D, iteration, 0, 0, max_iters, power=cfg.SOLVER.LR_POWER)
        for index in range(len(optimizer_fea.param_groups)):
            optimizer_fea.param_groups[index]['lr'] = current_lr
        for index in range(len(optimizer_cls.param_groups)):
            optimizer_cls.param_groups[index]['lr'] = current_lr
        for index in range(len(optimizer_D.param_groups)):
            optimizer_D.param_groups[index]['lr'] = current_lr_D
            

        src_input = src_input.cuda(non_blocking=True)
        src_label = src_label.cuda(non_blocking=True).long()
        tgt_input = tgt_input.cuda(non_blocking=True)
        tgt_label = tgt_label.cuda(non_blocking=True).long()
                    
        src_size = src_input.shape[-2:]
        tgt_size = tgt_input.shape[-2:]

        optimizer_fea.zero_grad()
        optimizer_cls.zero_grad()

        src_fea = feature_extractor(src_input)
        src_pred = classifier(src_fea)
        src_aux = aux(src_fea) 
        src_pred = resize(
            input=src_pred,
            size=src_size,
            mode='bilinear',
            align_corners=False)
        src_aux = resize(
            input=src_aux,
            size=src_size,
            mode='bilinear',
            align_corners=False)
        loss_seg_src = 1.0*criterion(src_pred,src_label)
        loss_aux_src = 0.4*criterion(src_aux,src_label)

        with torch.no_grad():
            tgt_fea_ema = feature_extractor_ema(tgt_input)
            tgt_pred_ema = classifier_ema(tgt_fea_ema)
            tgt_pred_ema = resize(
                input=tgt_pred_ema,
                size=tgt_size,
                mode='bilinear',
                align_corners=False)

            if cfg.SOLVER.DIS == 'class':
                # generate soft labels
                src_soft_label = F.softmax(src_pred, dim=1).detach()
                src_soft_label[src_soft_label>0.9] = 0.9

                tgt_soft_label = F.softmax(tgt_pred_ema, dim=1).detach()
                tgt_soft_label[tgt_soft_label>0.9] = 0.9
        

        src_fea_D = src_fea[-2]
        src_Dis_pred = model_Dis(src_fea_D.detach(), src_size)
        source_share_weight = get_share_weight(src_Dis_pred, src_pred, class_temperature=10.0)
        source_share_weight = normalize_weight(source_share_weight)
        src_D_pred = model_D(src_fea_D, src_size)
        if cfg.SOLVER.DIS == 'binary':
            loss_adv_src = 0.001*soft_label_cross_entropy((torch.ones_like(src_D_pred)-src_D_pred).clamp(min=1e-7, max=1.0), torch.ones_like(src_D_pred),source_share_weight)
        else:
            loss_adv_src = 0.001*soft_label_cross_entropy(F.softmax(src_D_pred, dim=1).clamp(min=1e-7, max=1.0), torch.cat((src_soft_label,torch.zeros_like(src_soft_label)), dim=1),source_share_weight)

        (loss_seg_src+loss_aux_src+loss_adv_src).backward()

        if cfg.warm_up == False:
            tgt_fea = feature_extractor(tgt_input)
            tgt_pred = classifier(tgt_fea)  
            tgt_aux = aux(tgt_fea) 
            tgt_pred = resize(
                input=tgt_pred,
                size=tgt_size,
                mode='bilinear',
                align_corners=False)
            tgt_aux = resize(
                input=tgt_aux,
                size=tgt_size,
                mode='bilinear',
                align_corners=False) 
            loss_seg_tgt = 1.0*criterion(tgt_pred,tgt_label)
            loss_aux_tgt = 0.4*criterion(tgt_aux,tgt_label)

            (loss_seg_tgt+loss_aux_tgt).backward()

            meters.update(loss_seg_tgt=loss_seg_tgt.item())
            meters.update(loss_aux_tgt=loss_aux_tgt.item())

        optimizer_fea.step()
        optimizer_cls.step()
        meters.update(loss_seg_src=loss_seg_src.item())
        meters.update(loss_aux_src=loss_aux_src.item())
        meters.update(loss_adv_src=loss_adv_src.item())

        optimizer_D.zero_grad()

        src_fea_D = src_fea[-2]
        src_Dis_pred = model_Dis(src_fea_D.detach(), src_size)
        loss_Dis_src = 0.5*bce_loss(src_Dis_pred, torch.ones_like(src_Dis_pred))
        loss_Dis_src.backward()

        tgt_fea_D = tgt_fea_ema[-2]
        tgt_Dis_pred = model_Dis(tgt_fea_D.detach(), tgt_size)
        loss_Dis_tgt = 0.5*bce_loss(tgt_Dis_pred, torch.zeros_like(tgt_Dis_pred))
        loss_Dis_tgt.backward()

        source_share_weight = get_share_weight(src_Dis_pred, src_pred, class_temperature=10.0)
        source_share_weight = normalize_weight(source_share_weight)
        target_share_weight = -get_share_weight(tgt_Dis_pred, tgt_pred_ema, class_temperature=1.0)
        target_share_weight = normalize_weight(target_share_weight)

        src_D_pred = model_D(src_fea_D.detach(), src_size)
        if cfg.SOLVER.DIS == 'binary':
            loss_D_src = 0.5*soft_label_cross_entropy(src_D_pred.clamp(min=1e-7, max=1.0), torch.ones_like(src_D_pred), source_share_weight)
        else:
            loss_D_src = 0.5*soft_label_cross_entropy(F.softmax(src_D_pred, dim=1).clamp(min=1e-7, max=1.0), torch.cat((torch.zeros_like(src_soft_label),src_soft_label), dim=1), source_share_weight)
        loss_D_src.backward()

        tgt_D_pred = model_D(tgt_fea_D.detach(), tgt_size)
        if cfg.SOLVER.DIS == 'binary':
            loss_D_tgt = 0.5*soft_label_cross_entropy((torch.ones_like(tgt_D_pred)-tgt_D_pred).clamp(min=1e-7, max=1.0), torch.ones_like(tgt_D_pred), target_share_weight)
        else:
            loss_D_tgt = 0.5*soft_label_cross_entropy(F.softmax(tgt_D_pred, dim=1).clamp(min=1e-7, max=1.0), torch.cat((tgt_soft_label,torch.zeros_like(tgt_soft_label)), dim=1), target_share_weight)
        loss_D_tgt.backward()

        optimizer_D.step()

        meters.update(loss_D_src=loss_D_src.item())
        meters.update(loss_D_tgt=loss_D_tgt.item())
        meters.update(loss_Dis_src=loss_Dis_src.item())
        meters.update(loss_Dis_tgt=loss_Dis_tgt.item())
        
        iteration+=1
        batch_time = time.time() - end
        end = time.time()
        meters.update(time=batch_time, data=data_time)
        eta_seconds = meters.time.global_avg * (cfg.SOLVER.STOP_ITER - iteration)
        eta_string = str(datetime.timedelta(seconds=int(eta_seconds)))
        if iteration % 20 == 0 or iteration == max_iters:
            logger.info(
                meters.delimiter.join(
                    [
                        "eta: {eta}",
                        "iter: {iter}",
                        "{meters}",
                            "lr: {lr:.6f}",
                            "max mem: {memory:.0f}",
                        ]
                    ).format(
                        eta=eta_string,
                        iter=iteration,
                        meters=str(meters),
                        lr=optimizer_fea.param_groups[0]["lr"],
                        memory=torch.cuda.max_memory_allocated() / 1024.0 / 1024.0,
                    )
                )
        with torch.no_grad():
            for param_q, param_k in zip(feature_extractor.parameters(), feature_extractor_ema.parameters()):
                param_k.data = param_k.data * (1-decay) + param_q.data * decay
            for param_q, param_k in zip(classifier.parameters(), classifier_ema.parameters()):
                param_k.data = param_k.data * (1-decay) + param_q.data * decay
            for param_q, param_k in zip(aux.parameters(), aux_ema.parameters()):
                param_k.data = param_k.data * (1-decay) + param_q.data * decay
        
        if iteration == cfg.SOLVER.STOP_ITER:
            rec = run_test(cfg, (feature_extractor_ema, classifier_ema), local_rank, distributed)
            rec['iters']=iteration
            test_stats.append(rec)

        if (iteration == cfg.SOLVER.STOP_ITER) and save_to_disk:
            filename = os.path.join(pth_dir, "model_last.pth")
            time.sleep(120)
            torch.save({'iteration': iteration, 
            'feature_extractor': feature_extractor_ema.state_dict(), 
            'classifier':classifier_ema.state_dict(), 
            'aux':aux_ema.state_dict(),
            'model_D': model_D.state_dict(), 
            'model_Dis': model_Dis.state_dict()
            }, filename)
            logger.info('Save in{}.'.format(iteration))

            with open(os.path.join(output_dir, 'test_results.csv'),'w') as handle:
                for i, rec in enumerate(test_stats):
                    if i==0:
                        handle.write(','.join(list(rec.keys()))+'\n')
                    line = [str(rec[key]) for key in rec.keys()]
                    handle.write(','.join(line)+'\n')

        if iteration == cfg.SOLVER.MAX_ITER:
            break
        if iteration == cfg.SOLVER.STOP_ITER:
            break
               
    total_training_time = time.time() - start_training_time
    total_time_str = str(datetime.timedelta(seconds=total_training_time))
    logger.info(
        "Total training time: {} ({:.4f} s / it)".format(
            total_time_str, total_training_time / (cfg.SOLVER.STOP_ITER)
        )
    )
      

def run_test(cfg, model, local_rank, distributed):
    logger = logging.getLogger("TransDA.tester")
    if local_rank==0:
        logger.info('>>>>>>>>>>>>>>>> Start Testing >>>>>>>>>>>>>>>>')
    
    batch_time = AverageMeter()
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    target_meter = AverageMeter()
    feature_extractor, classifier = model

    test_data = build_dataset(cfg, mode='test', is_source=False)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False, num_workers=4, pin_memory=True, sampler=None)
 
    feature_extractor.eval()
    classifier.eval()
    end = time.time()
    with torch.no_grad():
        for i, (x, y, _) in enumerate(test_loader):
            x = x.cuda(non_blocking=True)
            y = y.cuda(non_blocking=True).long()

            size = y.shape[-2:]
            pred = classifier(feature_extractor(x))
            pred = resize(
                input=pred,
                size=y.shape[-2:],
                mode='bilinear',
                align_corners=False)
            
            output = pred.max(1)[1]
            intersection, union, target = intersectionAndUnionGPU(output, y, cfg.MODEL.NUM_CLASSES, cfg.INPUT.IGNORE_LABEL)
            intersection, union, target = intersection.cpu().numpy(), union.cpu().numpy(), target.cpu().numpy()
            intersection_meter.update(intersection), union_meter.update(union), target_meter.update(target)

            accuracy = sum(intersection_meter.val) / (sum(target_meter.val) + 1e-10)
            batch_time.update(time.time() - end)
            end = time.time()
    
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    accuracy_class = intersection_meter.sum / (target_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    mAcc = np.mean(accuracy_class)
    allAcc = sum(intersection_meter.sum) / (sum(target_meter.sum) + 1e-10)
    rec = {'mIoU':mIoU}
    if local_rank==0:
        logger.info('Val result: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(mIoU, mAcc, allAcc))
        for i in range(cfg.MODEL.NUM_CLASSES):
            rec[str(i)] = iou_class[i]
            logger.info('Class_{} Result: iou/accuracy {:.4f}/{:.4f}.'.format(i, iou_class[i], accuracy_class[i]))
    feature_extractor.train()
    classifier.train()
    return rec

def main():
    parser = argparse.ArgumentParser(description="PyTorch Semantic Segmentation Training")
    parser.add_argument("-cfg",
        "--config-file",
        default="",
        metavar="FILE",
        help="path to config file",
        type=str,
    )
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument(
        "--skip-test",
        dest="skip_test",
        help="Do not test the final model",
        action="store_true",
    )
    parser.add_argument(
        "opts",
        help="Modify config options using the command-line",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

    num_gpus = int(os.environ["WORLD_SIZE"]) if "WORLD_SIZE" in os.environ else 1
    args.distributed = num_gpus > 1

    if args.distributed:
        torch.cuda.set_device(args.local_rank)
        torch.distributed.init_process_group(
            backend="nccl", init_method="env://"
        )

    cfg.merge_from_file(args.config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()

    output_dir = cfg.OUTPUT_DIR
    if output_dir:
        mkdir(output_dir)

    logger = setup_logger("TransDA", output_dir, args.local_rank)
    logger.info("Using {} GPUs".format(num_gpus))
    logger.info(args)

    logger.info("Loaded configuration file {}".format(args.config_file))
    with open(args.config_file, "r") as cf:
        config_str = "\n" + cf.read()
        logger.info(config_str)
    logger.info("Running with config:\n{}".format(cfg))

    # np.random.seed(cfg.SOLVER.SEED)
    # random.seed(cfg.SOLVER.SEED) 
    # torch.manual_seed(cfg.SOLVER.SEED)
    # torch.cuda.manual_seed_all(cfg.SOLVER.SEED)
    # torch.cuda.set_rng_state(torch.cuda.get_rng_state())
    # torch.cuda.set_rng_state_all(torch.cuda.get_rng_state_all())
    # torch.backends.cudnn.deterministic = True

    train(cfg, args.local_rank, args.distributed)


if __name__ == "__main__":
    main()
