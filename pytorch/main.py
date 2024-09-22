import os
import sys
sys.path.insert(1, os.path.join(sys.path[0], '../utils'))
import numpy as np
import argparse
import time
import logging
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torch.utils.data
 
from config import (sample_rate, classes_num, mel_bins, fmin, fmax, window_size, 
    hop_size, window, pad_mode, center, ref, amin, top_db)
from losses import get_loss_func
from pytorch_utils import move_data_to_device, do_mixup, count_parameters
from utilities import (create_folder, get_filename, create_logging, StatisticsContainer, Mixup)
from data_generator import Dataset, TrainSampler, EvaluateSampler, collate_fn
#from models import Cnn14, Cnn14_nnAudio, Cnn14_nnAudio_gf, Cnn14_nnAudio_norm, Cnn6, Cnn10, Cnn10_fusion, Cnn10_no_specaug, Cnn10_gf, Cnn10_gf_no_specaug, Transfer_Cnn10, Cnn10_origin, ResNet22, ResNet38, ResNet54
#from model import Cnn10, ResNet54, Cnn14, Cnn10_fusion, Cnn10_fusion_hf, Cnn10_fusion_lf, Cnn10_fusion_gf, Cnn10_lofar_demon, #Cnn10_gf_no_specaug, Cnn10_origin, Transfer_Cnn10, Transfer_Cnn10_gf, Cnn10_gf_DecisionLevelMax, Cnn10_gf
from models import Cnn6, Cnn10, Cnn14, ResNet22, ResNet38, ResNet54, Cnn10_gf
from evaluate import Evaluator

import wandb

def train(args):

    # Arugments & parameters
    workspace = args.workspace
    holdout_fold = args.holdout_fold
    model_type = args.model_type
    pretrained_checkpoint_path = args.pretrained_checkpoint_path
    freeze_base = args.freeze_base
    set_indexes = args.set_indexes
    loss_type = args.loss_type
    augmentation = args.augmentation
    learning_rate = args.learning_rate
    batch_size = args.batch_size
    resume_iteration = args.resume_iteration
    stop_iteration = args.stop_iteration
    device = 'cuda' if (args.cuda and torch.cuda.is_available()) else 'cpu'
    filename = args.filename
    num_workers = 8
    use_wandb = False if args.wandb=='none' else args.wandb
    feature_name = args.feature_name
    learnable = args.learnable

    loss_func = get_loss_func(loss_type)
    pretrain = True if pretrained_checkpoint_path else False
    
    #hdf5_path = os.path.join(workspace, 'data', 'waveform.h5')
    hdf5_path = args.hdf5_path

    checkpoints_dir = os.path.join(workspace, 'checkpoints', filename, 
        'holdout_fold={}'.format(holdout_fold), model_type, 'feature={}'.format(feature_name), 
        'learnable={}'.format(learnable), 
         'batch_size={}'.format(batch_size))
    create_folder(checkpoints_dir)

    statistics_path = os.path.join(workspace, 'statistics', filename, 
        'holdout_fold={}'.format(holdout_fold), model_type, 'feature={}'.format(feature_name), 
        'learnable={}'.format(learnable), 
        'batch_size={}'.format(batch_size), 
        'statistics.pickle')
    create_folder(os.path.dirname(statistics_path))
    
    logs_dir = os.path.join(workspace, 'logs', filename, 
        'holdout_fold={}'.format(holdout_fold), model_type, 'feature={}'.format(feature_name), 
        'learnable={}'.format(learnable), 
        'batch_size={}'.format(batch_size))
    create_logging(logs_dir, 'w')
    logging.info(args)

    if 'cuda' in device:
        logging.info('Using GPU.')
    else:
        logging.info('Using CPU. Set --cuda flag to use GPU.')
    
    # wandb
    if use_wandb:
        default_config = dict(
            model_type = model_type,
            holdout_fold = holdout_fold,
            batch_size = batch_size,
            loss_type = loss_type
        )

        wandb.init(project=args.wandb, config=args)
        #wandb.init(dir='/root/autodl-fs/non-synced')
    
    
    # Model
    Model = eval(model_type)
    model = Model(sample_rate, window_size, hop_size, mel_bins, fmin, fmax, classes_num, feature_name, learnable)
    
    params_num = count_parameters(model)
    logging.info('Parameters num: {}'.format(params_num))
    
    # Statistics
    statistics_container = StatisticsContainer(statistics_path)

    if pretrain:
        logging.info('Load pretrained model from {}'.format(pretrained_checkpoint_path))
        model.load_from_pretrain(pretrained_checkpoint_path)

    if resume_iteration:
        resume_checkpoint_path = os.path.join(checkpoints_dir, '{}_iterations.pth'.format(resume_iteration))
        logging.info('Load resume model from {}'.format(resume_checkpoint_path))
        resume_checkpoint = torch.load(resume_checkpoint_path)
        model.load_state_dict(resume_checkpoint['model'])
        statistics_container.load_state_dict(resume_iteration)
        iteration = resume_checkpoint['iteration']
    else:
        iteration = 0

    # Parallel
    print('GPU number: {}'.format(torch.cuda.device_count()))
    model = torch.nn.DataParallel(model)

    dataset = Dataset()

    # Data generator
    train_sampler = TrainSampler(
        hdf5_path=hdf5_path, 
        holdout_fold=holdout_fold, 
        batch_size=batch_size * 2 if 'mixup' in augmentation else batch_size,
        set_indexes=set_indexes)

    validate_sampler = EvaluateSampler(
        hdf5_path=hdf5_path, 
        holdout_fold=holdout_fold, 
        batch_size=batch_size,
        set_indexes=set_indexes)

    # Data loader
    train_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=train_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    validate_loader = torch.utils.data.DataLoader(dataset=dataset, 
        batch_sampler=validate_sampler, collate_fn=collate_fn, 
        num_workers=num_workers, pin_memory=True)

    if 'cuda' in device:
        model.to(device)

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999),
        eps=1e-08, weight_decay=0., amsgrad=True)

    
    if 'mixup' in augmentation:
        mixup_augmenter = Mixup(mixup_alpha=1.)
     
    # Evaluator
    evaluator = Evaluator(model=model)
    
    train_bgn_time = time.time()
    
    # Train on mini batches
    for batch_data_dict in train_loader:

        # import crash
        # asdf
        
        # Evaluate
        if iteration % 200 == 0:
            #if resume_iteration > 0 and iteration == resume_iteration:
            #    pass
            #else:
            logging.info('------------------------------------')
            logging.info('Iteration: {}'.format(iteration))

            train_fin_time = time.time()

            statistics = evaluator.evaluate(validate_loader)
            print(statistics)
            logging.info('Validate accuracy: {:.3f}'.format(np.mean(statistics['average_precision'])))

            statistics_container.append(iteration, statistics, 'validate')
            statistics_container.dump()
            if use_wandb:
                wandb.log({'Validate accuracy':np.mean(statistics['average_precision'])})#,'auc':np.mean(statistics['auc'])})#,
                           #'1-AP':statistics['average_precision'][0],'1-auc':statistics['auc'][0],
                           #'2-AP':statistics['average_precision'][1],'2-auc':statistics['auc'][1],
                           #'3-AP':statistics['average_precision'][2],'3-auc':statistics['auc'][2],
                           #'4-AP':statistics['average_precision'][3],'4-auc':statistics['auc'][3]})

            train_time = train_fin_time - train_bgn_time
            validate_time = time.time() - train_fin_time

            logging.info(
                'Train time: {:.3f} s, validate time: {:.3f} s'
                ''.format(train_time, validate_time))

            train_bgn_time = time.time()
            
            del statistics
            torch.cuda.empty_cache()

        # Save model 
        if iteration % 2000 == 0 and iteration > 0:
            checkpoint = {
                'iteration': iteration, 
                'model': model.module.state_dict()}

            checkpoint_path = os.path.join(
                checkpoints_dir, '{}_iterations.pth'.format(iteration))
                
            torch.save(checkpoint, checkpoint_path)
            logging.info('Model saved to {}'.format(checkpoint_path))
        
        if 'mixup' in augmentation:
            batch_data_dict['mixup_lambda'] = mixup_augmenter.get_lambda(len(batch_data_dict['waveform']))
        
        # Move data to GPU
        for key in batch_data_dict.keys():
            batch_data_dict[key] = move_data_to_device(batch_data_dict[key], device)
        
        # Train
        model.train()

        if 'mixup' in augmentation:
            batch_output_dict = model(batch_data_dict['waveform'], 
                batch_data_dict['mixup_lambda'])
            """{'clipwise_output': (batch_size, classes_num), ...}"""

            batch_target_dict = {'target': do_mixup(batch_data_dict['target'], 
                batch_data_dict['mixup_lambda'])}
            """{'target': (batch_size, classes_num)}"""
        else:
            batch_output_dict = model(batch_data_dict['waveform'], None)
            """{'clipwise_output': (batch_size, classes_num), ...}"""

            batch_target_dict = {'target': batch_data_dict['target']}
            """{'target': (batch_size, classes_num)}"""

        # loss
        loss = loss_func(batch_output_dict, batch_target_dict)
        # print(iteration, loss)
        if use_wandb:
            wandb.log({'loss':loss})

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Stop learning
        if iteration == stop_iteration:
            break 

        iteration += 1
        

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example of parser. ')
    subparsers = parser.add_subparsers(dest='mode')

    # Train
    parser_train = subparsers.add_parser('train')
    parser_train.add_argument('--workspace', type=str, required=True, help='Directory of your workspace.')
    parser_train.add_argument('--hdf5_path', type=str, required=True)
    parser_train.add_argument('--holdout_fold', type=str, choices=['1', '2', '3', '4', '5', '6', '7', '8', '9', '10'], required=True)
    parser_train.add_argument('--feature_name', type=str, required=True)
    parser_train.add_argument('--model_type', type=str, required=True)
    parser_train.add_argument('--pretrained_checkpoint_path', type=str)
    parser_train.add_argument('--freeze_base', action='store_true', default=False)
    parser_train.add_argument('--set_indexes', action='store_true', default=False)
    parser_train.add_argument('--loss_type', type=str, required=True)
    parser_train.add_argument('--augmentation', type=str, choices=['none', 'mixup'], required=True)
    parser_train.add_argument('--learning_rate', type=float, required=True)
    parser_train.add_argument('--batch_size', type=int, required=True)
    parser_train.add_argument('--resume_iteration', type=int)
    parser_train.add_argument('--stop_iteration', type=int, required=True)
    parser_train.add_argument('--cuda', action='store_true', default=False)
    parser_train.add_argument('--wandb', type=str, required=True)
    parser_train.add_argument('--learnable', type=str, required=True)

    # Parse arguments
    args = parser.parse_args()
    args.filename = get_filename(__file__)

    if args.mode == 'train':
        train(args)
        if args.wandb:
            wandb.finish()
    else:
        raise Exception('Error argument!')
    
    for _ in range(5):
        torch.cuda.empty_cache()