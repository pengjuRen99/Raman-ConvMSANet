import argparse
from pathlib import Path
import os
import torch
from torch.utils.tensorboard import SummaryWriter
from lib.BloodSpectra import BloodSpectra
import model.ConvMSANet as ConvMSANet
import time
import utils.misc as misc
from timm.utils import accuracy
import json
import datetime
from utils.misc import EarlyStopping
from utils.misc import NativeScalerWithGradNormCount as NativeScaler
from thop import profile

from sklearn.metrics import classification_report, precision_score, recall_score
import numpy as np

import xlwt
import shutil
from utils.SplitDataset import cross_validation

def get_args_parser():
    parser = argparse.ArgumentParser('Raman ConvMSANet training', add_help=False)
    parser.add_argument('--batch_size', default=64, type=int, 
                        help='Batch size')
    parser.add_argument('--epochs', default=80, type=int)

    # Model parameters
    parser.add_argument('--model', default='convmsa_transmission', type=str, metavar='MODEL',
                        help='Name of reflection or transmission model to train')

    parser.add_argument('--spectra_size', default=960, type=int,
                        help='spectras input size')

    # Optimizer parameters
    parser.add_argument('--weight_decay', type=float, default=0.05,         # 0.05
                        help='weight decay (default: 0.05)')

    parser.add_argument('--lr', type=float, default=3e-4, metavar='LR',     # 3e-4     1e-3
                        help='learning rate (absolute lr)')
    

    # Dataset parameters
    parser.add_argument('--data_path', default='split_animal_data/Transmissive_blood_dataset', type=str,
                        help='dataset path')

    parser.add_argument('--output_dir', default='output_dir/transmissive_blood_ConvMSANet',
                        help='path where to save, empty for no saving')
    parser.add_argument('--log_dir', default='output_dir/transmissive_blood_ConvMSANet',
                        help='path where to tensorboard log')
    
    parser.add_argument('--device', default='cuda:0',
                        help='device to use for training / testing')
    
    parser.add_argument('--num_workers', default=10, type=int)
    parser.add_argument('--num_classes', default=42, type=int, help='number of the classification types')

    return parser


def train_one_epoch(model, loss_function, data_loader, optimizer, scheduler_lr, device, epoch, args):
    model.train(True)
    metric_logger = misc.MetricLogger(delimiter='  ')
    metric_logger.add_meter('lr', misc.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 5

    optimizer.zero_grad()
    
    for data_iter_step, (samples, targets) in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        samples = samples.to(device)
        targets = targets.to(device)
        outputs = model(samples)
        loss = loss_function(outputs, targets)
        loss_value = loss.item()
        optimizer.zero_grad()
        loss.mean().backward()
        optimizer.step()
        metric_logger.update(loss=loss_value)
        metric_logger.update(lr=optimizer.param_groups[0]['lr'])
        acc1, acc5 = accuracy(outputs, targets, topk=(1, 5))
        batch_size = samples.shape[0]
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)

    scheduler_lr.step()

    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


@torch.no_grad()
def evaluate(data_loader, model, device, criterion, header = 'Valid:'):
    
    metric_logger = misc.MetricLogger(delimiter='  ')

    model.eval()

    # test stage 
    test_target = np.array([])
    test_output = np.array([])

    for batch in metric_logger.log_every(data_loader, 2, header):
        images = batch[0]
        target = batch[-1]
        images = images.to(device)
        target = target.to(device)

        output = model(images)
        loss = criterion(output, target)

        # test stage
        test_output = np.append(test_output, output.cpu().argmax(dim=-1).detach().numpy())
        test_target = np.append(test_target, target.cpu().numpy())

        batch_size = images.shape[0]
        acc1, acc5 = accuracy(output, target, topk=(1, 5))
        precision = precision_score(target.cpu(), output.cpu().argmax(dim=-1), average='weighted', zero_division=1)
        recall = recall_score(target.cpu(), output.cpu().argmax(dim=-1), average='weighted', zero_division=1)
        F1_score = 2 * (precision * recall) / (precision + recall)
        
        metric_logger.update(loss=loss.item())
        metric_logger.meters['acc1'].update(acc1.item(), n=batch_size)
        metric_logger.meters['acc5'].update(acc5.item(), n=batch_size)
        metric_logger.meters['precision'].update(precision, n=batch_size)
        metric_logger.meters['recall'].update(recall, n=batch_size)
        metric_logger.meters['f1_score'].update(F1_score, n=batch_size)


    if header == 'Valid:':
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'.format(
            top1=metric_logger.acc1,
            top5=metric_logger.acc5,
            losses=metric_logger.loss
        ))

    if header == 'Test:':
        print('Classification_report', classification_report(test_target, test_output, digits=4, zero_division=1))
        print('* Acc@1 {top1.global_avg:.3f} Acc@5 {top5.global_avg:.3f} loss {losses.global_avg:.3f}'.format(
            top1=metric_logger.acc1,
            top5=metric_logger.acc5,
            losses=metric_logger.loss
        ))
        print('* precision {precision.global_avg:.3f} recall {recall.global_avg:.3f} F1_score {f1_score.global_avg:.3f}'.format(
            precision=metric_logger.precision,
            recall=metric_logger.recall,
            f1_score=metric_logger.f1_score,
        ))
        
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}


def main(args, i, worksheet):
    
    print('job dir: {}'.format(os.path.dirname(os.path.realpath(__file__))))
    print("{}".format(args).replace(', ', ',\n'))

    device = torch.device(args.device)
    print('training device: {}'.format(device))

    # tensorboard
    if args.log_dir is not None:
        os.makedirs(args.log_dir, exist_ok=True)
        log_writer = SummaryWriter(log_dir=args.log_dir)
    else:
        log_writer = None

    # remove last fold files
    if os.path.exists(args.data_path + '/test'):
        shutil.rmtree(args.data_path + '/test')
    os.mkdir(args.data_path + '/test')
    if os.path.exists(args.data_path + '/train'):
        shutil.rmtree(args.data_path + '/train')
    os.mkdir(args.data_path + '/train')
    if os.path.exists(args.data_path + '/valid'):
        shutil.rmtree(args.data_path + '/valid')
    os.mkdir(args.data_path + '/valid')

    cross_validation(i=i)
    dataset_train = BloodSpectra(spectra_path=args.data_path, json_name='class_Transmission.json', mode='train')
    dataset_valid = BloodSpectra(spectra_path=args.data_path, json_name='class_Transmission.json', mode='valid')
    dataset_test = BloodSpectra(spectra_path=args.data_path, json_name='class_Transmission.json', mode='test')

    
    data_loader_train = torch.utils.data.DataLoader(
        dataset_train, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True
    )
    data_loader_valid = torch.utils.data.DataLoader(
        dataset_valid, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=True
    )
    
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test, batch_size=args.batch_size, num_workers=args.num_workers, shuffle=False
    )

    model = ConvMSANet.__dict__[args.model](num_classes=args.num_classes)

    model.to(device)
    model_without_ddp = model
    n_parameters = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print("Model = %s" % str(model_without_ddp))
    print('number of params (M): %.2f' % (n_parameters / 1.e6))
    input1 = torch.randn(1, 1, 960).to(device)
    flops, params = profile(model, inputs=(input1, ))
    print('FLOPs = ' + str(flops/1000**3) + 'G')
    print('Params = ' + str(params/1000**2) + 'M')


    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=args.weight_decay)
    ExpLR = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.98)
    loss_function = torch.nn.CrossEntropyLoss()
    loss_scaler = NativeScaler()
    # early_stop = EarlyStopping(monitor='acc1', mode='max', patience=10)

    print('Start training: ')
    start_time = time.time()
    max_accuracy = 0.0
    for epoch in range(0, args.epochs):
        # train stage
        train_stats = train_one_epoch(model=model, loss_function=loss_function, data_loader=data_loader_train, optimizer=optimizer, scheduler_lr=ExpLR, device=device, epoch=epoch, args=args)
        # evaluation stage
        val_stats = evaluate(data_loader=data_loader_valid, model=model, device=device, criterion=loss_function, header='Valid:')
        print(f"Accuracy of the network on the {len(dataset_valid)} valid spectra: {val_stats['acc1']:.2f}%")

        if max_accuracy < val_stats['acc1']:
            max_accuracy = val_stats['acc1']
            misc.save_best_model(args=args, epoch=epoch, model=model, model_without_ddp=model_without_ddp, optimizer=optimizer, loss_scaler=loss_scaler,)
        print(f'Max accuracy: {max_accuracy:.2f}%')

        if log_writer is not None:
            log_writer.add_scalar('train_acc1', train_stats['acc1'], epoch)
            log_writer.add_scalar('train_acc5', train_stats['acc5'], epoch)
            log_writer.add_scalar('train_loss', train_stats['loss'], epoch)
            log_writer.add_scalar('train_lr', train_stats['lr'], epoch)
            log_writer.add_scalar('val_acc1', val_stats['acc1'], epoch)
            log_writer.add_scalar('val_acc5', val_stats['acc5'], epoch)
            log_writer.add_scalar('val_loss', val_stats['loss'], epoch)

        log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                        **{f'val_{k}': v for k, v in val_stats.items()},
                        'epoch': epoch,
                        'n_parameters': n_parameters}
        
        with open(os.path.join(args.output_dir, "Transmission_train_log.txt"), mode="a", encoding="utf-8") as f:
                f.write(json.dumps(log_stats) + "\n")
        
        # 提前停止训练
        # if early_stop(val_stats):
        #     break

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

    # 测试阶段
    model.load_state_dict(torch.load(args.output_dir + '/best_checkpoint.pth', map_location=device)['model'], strict=True)    # 加载最佳验证集权重
    test_stats = evaluate(data_loader_test, model, device, loss_function, header='Test:')
    print(f"Accuracy of the network on the {len(dataset_test)} test spectra: {test_stats['acc1']:.2f}%")
    # plot_cm(test_target, test_output)

    # 测试结果写入excel
    if worksheet is not None:
        worksheet.write(i+2, 0, label=i+1)
        worksheet.write(i+2, 1, label='{:.4f}'.format(test_stats['loss']))
        worksheet.write(i+2, 2, label='{:.4f}'.format(test_stats['acc1']))
        worksheet.write(i+2, 3, label='{:.4f}'.format(test_stats['acc5']))
        worksheet.write(i+2, 4, label='{:.4f}'.format(test_stats['precision']))
        worksheet.write(i+2, 5, label='{:.4f}'.format(test_stats['recall']))
        worksheet.write(i+2, 6, label='{:.4f}'.format(test_stats['f1_score']))
        worksheet.write(i+2, 7, label='{:.4f}'.format(params/1000**2))
        worksheet.write(i+2, 8, label='{:.4f}'.format(flops/1000**3))

if __name__ == '__main__':
    args = get_args_parser().parse_args()
    if args.output_dir:
        Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # 进行十折交叉验证
    # 创建excel
    
    
    workbook = xlwt.Workbook(encoding='utf-8')
    worksheet = workbook.add_sheet('Transmission_Vit')
    worksheet.write(0, 0, label='Transmission_ConvMSANet_10CrossValidation_result')
    worksheet.write(1, 0, label='test_index')
    worksheet.write(1, 1, label='test_loss')
    worksheet.write(1, 2, label='test_acc1')
    worksheet.write(1, 3, label='test_acc5')
    worksheet.write(1, 4, label='test_precision')
    worksheet.write(1, 5, label='test_recall')
    worksheet.write(1, 6, label='test_f1-Score')
    worksheet.write(1, 7, label='Parameters')
    worksheet.write(1, 8, label='FLOPs')

    for i in range(0, 10):
        main(args, i, worksheet)
    workbook.save(args.output_dir + '/Transmission_ConvMSANet_10CrossValidation_result.xls')
    

    # main(args, 0, None)

