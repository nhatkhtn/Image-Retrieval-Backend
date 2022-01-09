import os
import time
import datetime as datetime
import torch
import torch.nn as nn
import torch.optim as optim
import torch.distributed as dist
import numpy as np 
import math 
from os import path 
import torch.nn.functional as F 
from tqdm import tqdm 
from pathlib import Path 
# Model
# from networks.mstcn.mstcn import Network 
from networks.mstcn.mstcn import Network 
from utils.losses import LossComputer
from utils.metrics import IoUComputer, calc_uncertainty 
from utils.integrator import Integrator
from utils.learning import adjust_learning_rate
from loggers.wandb_logger import WandB

# Dataset
from torch.utils.data import DataLoader, ConcatDataset
from datasets.synthetic import SyntheticDataset
from datasets.coco import COCODataset
from datasets.vos import VOSDataset
from datasets.static import StaticTransformDataset

from utils.load_subset import load_sub_davis, load_sub_yv,load_sub_uvo

dist.init_process_group(backend = 'nccl')
local_rank = torch.distributed.get_rank()
world_size = torch.distributed.get_world_size()

def worker_init_fn(worker_id): 
    return np.random.seed(torch.initial_seed()%(2**31) + worker_id + local_rank * 100)

class Trainer(object):
    def __init__(self, cfg, local_rank = 0):
        # Config 
        self.gpu = local_rank
        self.local_rank = local_rank
        self.cfg = cfg
        self.print_log(cfg)
        print("Use GPU {} for training".format(self.gpu))
        torch.cuda.set_device(self.gpu)
        
        # Model
        self.network = Network(cfg = cfg.MODEL).cuda()
        if cfg.DIST.ENABLE:
            # Set seed
            
            self.dist_model = torch.nn.parallel.DistributedDataParallel(
                self.network, 
                device_ids=[self.gpu], broadcast_buffers=False,
                output_device=local_rank,
                find_unused_parameters=True
            )
        else:
            self.dist_model = self.network
        self.network = self.dist_model

        # for name, param in self.network.named_parameters():
        #     if 'pgm' in name:
        #         param.requires_grad = True 
        #     else:
        #         param.requires_grad = False 
        #     print(name, param.requires_grad)

        # Logger
        self.logger = None 
        if self.local_rank == 0:
            long_id = '%s_%s_%s' % (datetime.datetime.now().strftime('%b%d_%H.%M.%S'), cfg.ID, cfg.DESCRIPTION)
            self.logger =  WandB(project_name = cfg.ID, save_path=cfg.LOG.SAVE_PATH, config = cfg, id=long_id)
            self.save_path = path.join(cfg.LOG.SAVE_PATH, long_id, long_id)

        # Losses & Metrics
        self.loss_computer = LossComputer(cfg)
        self.metric_computer = IoUComputer()
        self.integrator = Integrator(self.logger, distributed=True, local_rank=local_rank, world_size=cfg.TRAIN.GPUS)

        # Optimizer & Scheduler
        self.optimizer = optim.AdamW(
            filter(lambda p: p.requires_grad, self.network.parameters()), 
            lr=cfg.OPTIMIZER.LR, 
            weight_decay=cfg.OPTIMIZER.WEIGHT_DECAY
        )

        self.scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, cfg.SCHEDULER.STEPS, cfg.SCHEDULER.GAMMA)

        self.prepare_dataset()
        self.process_pretrained_model()

        if cfg.AMP:
            self.scaler = torch.cuda.amp.GradScaler()

        self.last_time = time.time()
    
    def prepare_dataset(self, max_skip = 5):
        cfg = self.cfg
        self.print_log('Process dataset...')
        train_datasets = []
        for dataset in cfg.DATASETS:
            if dataset.name == 'synthetic':
                curr_dataset = []
                for subdataset in dataset.list:
                    if subdataset.name == 'static':
                        dataset_root = path.expanduser(subdataset.root)
                        fss_dataset = StaticTransformDataset(path.join(dataset_root, 'fss'), method=0)
                        duts_tr_dataset = StaticTransformDataset(path.join(dataset_root, 'DUTS-TR'), method=1)
                        duts_te_dataset = StaticTransformDataset(path.join(dataset_root, 'DUTS-TE'), method=1)
                        ecssd_dataset = StaticTransformDataset(path.join(dataset_root, 'ecssd'), method=1)

                        big_dataset = StaticTransformDataset(path.join(dataset_root, 'BIG_small'), method=1)
                        hrsod_dataset = StaticTransformDataset(path.join(dataset_root, 'HRSOD_small'), method=1)

                        # BIG and HRSOD have higher quality, use them more
                        static_dataset = [fss_dataset, duts_tr_dataset, duts_te_dataset, ecssd_dataset] + [big_dataset, hrsod_dataset]*5

                        curr_dataset.extend(static_dataset * subdataset.num_repeat)

                    if subdataset.name == 'coco':
                        coco_dataset = COCODataset(**subdataset.args.data)
                        curr_dataset.extend([coco_dataset] * subdataset.num_repeat)
                
                curr_dataset = curr_dataset[0] if len(curr_dataset) == 1 else ConcatDataset(curr_dataset)
                synthetic_dataset = SyntheticDataset(dataset = curr_dataset, nimgs = 3)
                train_datasets.extend([synthetic_dataset] * dataset.num_repeat)

            if dataset.name == 'davis17':
                dataset_root = path.expanduser(dataset.root)
                davis_dataset = VOSDataset(path.join(dataset_root, 'JPEGImages', '480p'), 
                        path.join(dataset_root, 'Annotations', '480p'), max_skip, is_bl=False, num_object = dataset.num_object, subset=load_sub_davis())
                
                train_datasets.extend([davis_dataset] * dataset.num_repeat)

            if dataset.name == 'youtubevos':
                dataset_root = path.expanduser(dataset.root)
                youtubevos_dataset = VOSDataset(path.join(dataset_root, 'JPEGImages'), 
                        path.join(dataset_root, 'Annotations'), max_skip//5, is_bl=False, num_object = dataset.num_object, subset=load_sub_yv())
                train_datasets.extend([youtubevos_dataset] * dataset.num_repeat)

            if dataset.name == 'uvo':
                dataset_root = path.expanduser(dataset.root)
                davis_dataset = VOSDataset(path.join(dataset_root, 'JPEGImages', '480p'), 
                        path.join(dataset_root, 'Annotations', '480p'), max_skip, is_bl=False, num_object = dataset.num_object, subset=load_sub_uvo())
                
                train_datasets.extend([davis_dataset] * dataset.num_repeat)

        if len(train_datasets) > 1:
            train_dataset = ConcatDataset(train_datasets)
        elif len(train_datasets) == 1:
            train_dataset = train_datasets[0]
        else:
            self.print_log('No dataset!')
            exit(0)

        # assert cfg.TRAIN.BATCH_SIZE % cfg.TRAIN.NGPUS == 0, "TRAIN.BATCH_SIZE must be divided by TRAIN.NGPUS"
        self.train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset, rank=self.local_rank, shuffle=True)
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=cfg.TRAIN.BATCH_SIZE // cfg.TRAIN.GPUS,
            num_workers=cfg.TRAIN.NUM_WORKERS, 
            worker_init_fn=worker_init_fn,
            drop_last=True, pin_memory=True, 
            sampler=self.train_sampler)

        self.print_log('Done!')

    def process_pretrained_model(self):
        cfg = self.cfg
        self.pretrained_step = 0
        self.step = cfg.TRAIN.START_STEP
        self.epoch = self.epoch = int(np.ceil(self.step / len(self.train_loader)))
            

        if cfg.TRAIN.RESUME and cfg.PRETRAINED:
            self.pretrained_step = self.load_model(cfg.PRETRAINED)
            self.epoch = int(np.ceil(self.step / len(self.train_loader)))
            self.print_log('Load pretrained VOS model from {}.'.format(cfg.PRETRAINED))
            self.print_log('Resume from step {}'.format(self.step))
       
        elif cfg.PRETRAINED:
            self.load_network(cfg.PRETRAINED)
            self.print_log('Load pretrained VOS model from {}.'.format(cfg.PRETRAINED))
        
    def training(self):
        # torch.autograd.set_detect_anomaly(True)
        cfg = self.cfg 
        self.network.module.train_()
        self.report_interval = cfg.LOG.REPORT_INTERVAL
        self.save_im_interval = cfg.LOG.SAVE_IM_INTERVAL 
        self.save_model_interval = cfg.LOG.SAVE_MODEL_INTERVAL

        step = self.step 
        train_sampler = self.train_sampler
        train_loader = self.train_loader
        epoch = self.epoch
        np.random.seed(np.random.randint(2**30-1) + self.local_rank*100)
        increase_skip_fraction = cfg.STRATEGY.increase_skip_fraction
        skip_values = cfg.STRATEGY.skip_values
        total_epoch = math.ceil(cfg.TRAIN.TOTAL_STEPS/len(train_loader))
        increase_skip_epoch = [round(total_epoch*f) for f in increase_skip_fraction]
        self.print_log(f"Skip epoch: {increase_skip_epoch}")
        self.print_log(f"Skip value: {skip_values}")
        
        # TRAINING 

        while step < cfg.TRAIN.TOTAL_STEPS:            
            epoch += 1
            last_time = time.time()
            self.print_log(f'Epoch {epoch}')
            if epoch>=increase_skip_epoch[0]:
                while epoch >= increase_skip_epoch[0]:
                    skip_values = skip_values[1:]
                    cur_skip = skip_values[0]
                    increase_skip_epoch = increase_skip_epoch[1:]
                print('Increasing skip to: ', cur_skip)
                self.prepare_dataset(cur_skip)
                train_sampler = self.train_sampler
                train_loader = self.train_loader
                
            # Crucial for randomness! 
            train_sampler.set_epoch(epoch)
            # Train loop 
            self.train()
            progressbar = tqdm(train_loader) if self.local_rank == 0 else train_loader
            for data in progressbar:
                    if self.pretrained_step > step:
                        step += 1
                        continue 
                    # continue 
                    step += 1
                    losses = self.forward_sample(data, step)
                    
                    if self.local_rank == 0:
                        progressbar.set_description(f"Step {step}, loss: {losses}")
                    
                    if step >= cfg.TRAIN.TOTAL_STEPS:
                        break
                
            # Validate loop 
            # self.val() 

    
    def forward_sample(self, data, step):
        now_lr = adjust_learning_rate(
            optimizer=self.optimizer, 
            base_lr=self.cfg.OPTIMIZER.LR, 
            p=0.9, 
            itr=step, 
            max_itr=self.cfg.TRAIN.TOTAL_STEPS, 
            warm_up_steps=1000, 
        )

        torch.set_grad_enabled(self._is_train)
        with torch.cuda.amp.autocast(enabled=self.cfg.AMP):
            for k, v in data.items():
                if type(v) != list and type(v) != dict and type(v) != int:
                    data[k] = v.cuda(non_blocking=True)

            Fs = data['img'] # B, T, C, H, W 
            Ms = data['mask'] # B, T, H, W
            Ms_oh = F.one_hot(Ms).permute(0, 1, 4, 2, 3).float() # B, T, N, H, W
            num_objs = data['num_objs']
            T = int(data['num_imgs'][0])

            start = 0
            Es, uncertainty_list = self.network(Fs, Ms_oh, start, T)

            if self._do_log or self._is_train:
                pred = torch.argmax(F.softmax(Es, dim=2), dim=2)
                losses = self.loss_computer.compute(Es[:,start:], Ms[:,start:], uncertainty_list, num_objs, step)
                ious, details_iou = self.metric_computer.compute(pred[:, start:], Ms[:, start:], num_objs)
               
                if self._do_log:
                    self.integrator.add_dict(losses)
                    self.integrator.add_tensor("IoU", ious)
                    self.integrator.add_tensor("Num Objects", num_objs.float().mean())
                    if self._is_train:
                        if step % self.save_im_interval == 0 and self.logger:
                            # Log images
                            pass
                            self.logger.log_images(Fs, pred, Ms, uncertainty_list, details_iou, step)

            if self._is_train:
                if step % self.report_interval == 0:
                    if self.logger is not None:
                        print("Distribution perm: ", self.network.module.identity_bank.distributed_p)
                        self.logger.log_metrics('train','lr', now_lr, step)
                        self.logger.log_metrics('train', 'time', (time.time()-self.last_time)/self.report_interval / self.cfg.TRAIN.BATCH_SIZE, step)
                    
                    self.last_time = time.time()
                    self.integrator.finalize('train', step)
                    self.integrator.reset_except_hooks()

                if step % self.save_model_interval == 0 and step != 0:
                    if self.logger is not None:
                        self.save(step)
            
                # Backward pass

        if self._is_train:
            if self.cfg.AMP:
                self.scaler.scale(losses['total_loss']).backward()
                self.scaler.step(self.optimizer)    
                self.scaler.update()
            else:   
                losses['total_loss'].backward() 
                self.optimizer.step()
                self.optimizer.zero_grad(set_to_none=True)
        return losses['total_loss'].detach().cpu()

    def print_log(self, st): 
        if self.local_rank == 0: 
            print(st) 

    def save(self, step):
        if self.save_path is None:
            print("Saving has been disabled.")
            return 
        
        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        model_path = self.save_path + ('_%s.pth' % step)
        torch.save(self.network.module.state_dict(), model_path)
        print('Model saved to %s.' % model_path)
        
        lst_ckpt = list(map(str, sorted(Path(os.path.dirname(self.save_path)).iterdir(), key=os.path.getmtime)))
        if len(lst_ckpt) > 10:
          print("Remove checkpoint %s" % lst_ckpt[0])
          os.remove(lst_ckpt[0])

        self.save_checkpoint(step)
    
    def save_checkpoint(self, step):
        if self.save_path is None:
            print('Saving has been disabled.')
            return

        os.makedirs(os.path.dirname(self.save_path), exist_ok=True)
        checkpoint_path = self.save_path + '_checkpoint.pth'
        checkpoint = { 
            'it': step,
            'network': self.network.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.scheduler.state_dict()}
        torch.save(checkpoint, checkpoint_path)

    def load_model(self, path):
        map_location = 'cuda:%d' % self.local_rank
        checkpoint = torch.load(path, map_location={'cuda:0': map_location})

        # checkpoint = torch.load(path, strict=False) 

        it = checkpoint['it']
        network = checkpoint['network']
        optimizer = checkpoint['optimizer']
        scheduler = checkpoint['scheduler']

        # map_location = 'cuda:%d' % self.local_rank
        print(self.network.module.load_state_dict(network, strict = False))
        self.optimizer.load_state_dict(optimizer)
        self.scheduler.load_state_dict(scheduler)

        print('Model loaded.')

        return it
    
    def load_network(self, path):
        map_location = 'cuda:%d' % self.local_rank
        src_dict = torch.load(path, map_location={'cuda:0': map_location})

        print(self.network.module.load_state_dict(src_dict, strict=False))
        print('Network weight loaded:', path)

    def train(self):
        self._is_train = True
        self._do_log = True
        self.network.eval()
        return self

    def val(self):
        self._is_train = False
        self._do_log = True
        self.network.eval()
        return self
