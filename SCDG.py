import logging
import os
import time
import warnings
import math
import torch
from torch import nn
from torch import optim
import numpy as np
import random
import models
import datasets


class train_utils(object):
    def __init__(self, args, save_dir):
        """
        初始化训练工具类
        :param args: 包含训练参数的字典（例如：批量大小、模型名称等）
        :param save_dir: 保存模型和日志的目录
        """
        self.args = args
        self.save_dir = save_dir

    def setup(self):
        """
        初始化数据集、模型、损失函数和优化器
        """
        args = self.args

        # 检查是否可以使用GPU，并设置设备
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            logging.info('使用 {} 个GPU'.format(self.device_count))
            assert args.batch_size % self.device_count == 0, "批量大小应能被设备数量整除"
        else:
            warnings.warn("没有可用的GPU")
            self.device = torch.device("cpu")
            self.device_count = 1
            logging.info('使用 CPU')

        # 加载数据集
        Dataset = getattr(datasets, args.data_name)  # 动态加载数据集
        self.datasets = {}

        # 如果transfer_task是字符串类型，将其转换为列表
        if isinstance(args.transfer_task[0], str):
            args.transfer_task = eval("".join(args.transfer_task))

        # 将数据集分割成源数据集和目标数据集
        self.datasets['source_train'], self.datasets['source_val'], _, self.datasets['target_val'] = Dataset(
            args.data_dir, args.transfer_task, args.normlizetype).data_split(transfer_learning=True)

        # 为源训练、源验证和目标验证创建数据加载器
        self.dataloaders = {x: torch.utils.data.DataLoader(self.datasets[x], batch_size=args.batch_size,
                                                           shuffle=(True if x.split('_')[1] == 'train' else False),
                                                           num_workers=args.num_workers,
                                                           pin_memory=(True if self.device == 'cuda' else False),
                                                           drop_last=(True if args.last_batch and x.split('_')[
                                                               1] == 'train' else False))
                            for x in ['source_train', 'source_val', 'target_val']}

        # 动态加载模型
        self.model = getattr(models, args.model_name)(args.pretrained)  # 特征提取模型

        # 如果使用瓶颈层，定义瓶颈层和分类器
        if args.bottleneck:
            self.bottleneck_layer = nn.Sequential(nn.Linear(self.model.output_num(), args.bottleneck_num),
                                                  nn.ReLU(inplace=True), nn.Dropout())
            self.classifier_layer = nn.Linear(args.bottleneck_num, Dataset.num_classes)
        else:
            self.classifier_layer = nn.Linear(self.model.output_num(), Dataset.num_classes)

        # 创建一个结合特征提取器、瓶颈层和分类器的模型
        self.model_all = nn.Sequential(self.model, self.bottleneck_layer, self.classifier_layer)

        # 如果启用领域对抗网络，则创建它
        if args.domain_adversarial:
            self.max_iter = len(self.dataloaders['source_train']) * (args.max_epoch - args.middle_epoch)
            if args.bottleneck:
                # getattr(models, 'AdversarialNet_multi'):
                # 动态从 models 模块中加载名为 AdversarialNet_multi 的类。
                # 这个类实现了一个多任务的对抗网络，用于处理多个领域的对抗学习任务。
                self.AdversarialNet = getattr(models, 'AdversarialNet_multi')(
                    in_feature=args.bottleneck_num,
                    output_size=len(args.transfer_task[0]),
                    hidden_size=args.hidden_size,
                    max_iter=self.max_iter,
                    trade_off_adversarial=args.trade_off_adversarial,
                    lam_adversarial=args.lam_adversarial
                )
            else:
                self.AdversarialNet = getattr(models, 'AdversarialNet_multi')(
                    in_feature=self.model.output_num(),
                    output_size=len(args.transfer_task[0]),
                    hidden_size=args.hidden_size,
                    max_iter=self.max_iter,
                    trade_off_adversarial=args.trade_off_adversarial,
                    lam_adversarial=args.lam_adversarial
                )

        # 使用DataParallel支持多GPU训练
        if self.device_count > 1:
            self.model = torch.nn.DataParallel(self.model)
            if args.bottleneck:
                self.bottleneck_layer = torch.nn.DataParallel(self.bottleneck_layer)
            if args.domain_adversarial:
                self.AdversarialNet = torch.nn.DataParallel(self.AdversarialNet)
            self.classifier_layer = torch.nn.DataParallel(self.classifier_layer)

        # 定义参数列表和优化器
        if args.domain_adversarial:
            if args.bottleneck:
                parameter_list = [{"params": self.model.parameters(), "lr": args.lr},
                                  {"params": self.bottleneck_layer.parameters(), "lr": args.lr},
                                  {"params": self.classifier_layer.parameters(), "lr": args.lr},
                                  {"params": self.AdversarialNet.parameters(), "lr": args.lr}]
            else:
                parameter_list = [{"params": self.model.parameters(), "lr": args.lr},
                                  {"params": self.classifier_layer.parameters(), "lr": args.lr},
                                  {"params": self.AdversarialNet.parameters(), "lr": args.lr}]
                # self.model.parameters()：获取主模型（通常是特征提取器，比如 ResNet、VGG 等）的所有参数。
                # self.classifier_layer.parameters()：获取分类器层的所有参数（通常是全连接层，用于最终分类任务）。
                # self.AdversarialNet.parameters()：获取对抗网络的所有参数（如果使用对抗训练，比如领域对齐）。
                # lr: args.lr：为每部分参数设置相同的学习率 args.lr。
        else:
            if args.bottleneck:
                parameter_list = [{"params": self.model.parameters(), "lr": args.lr},
                                  {"params": self.bottleneck_layer.parameters(), "lr": args.lr},
                                  {"params": self.classifier_layer.parameters(), "lr": args.lr}]
            else:
                parameter_list = [{"params": self.model.parameters(), "lr": args.lr},
                                  {"params": self.classifier_layer.parameters(), "lr": args.lr}]

        # 设置优化器（SGD或Adam）
        if args.opt == 'sgd':
            self.optimizer = optim.SGD(parameter_list, lr=args.lr,
                                       momentum=args.momentum, weight_decay=args.weight_decay)
        elif args.opt == 'adam':
            self.optimizer = optim.Adam(parameter_list, lr=args.lr,
                                        weight_decay=args.weight_decay)
        else:
            raise Exception("优化器未实现")

        # 定义学习率调度器（步长衰减、指数衰减等）
        if args.lr_scheduler == 'step':
            steps = [int(step) for step in args.steps.split(',')]
            self.lr_scheduler = optim.lr_scheduler.MultiStepLR(self.optimizer, steps, gamma=args.gamma)
        elif args.lr_scheduler == 'exp':
            self.lr_scheduler = optim.lr_scheduler.ExponentialLR(self.optimizer, args.gamma)
        elif args.lr_scheduler == 'stepLR':
            steps = int(args.steps)
            self.lr_scheduler = optim.lr_scheduler.StepLR(self.optimizer, steps, args.gamma)
        elif args.lr_scheduler == 'fix':
            self.lr_scheduler = None
        else:
            raise Exception("学习率调度器未实现")

        # 初始化开始的epoch，并设置模型到相应的设备
        self.start_epoch = 0
        self.model.to(self.device)
        if args.bottleneck:
            self.bottleneck_layer.to(self.device)
        if args.domain_adversarial:
            self.AdversarialNet.to(self.device)
        self.classifier_layer.to(self.device)
        # 把model(即特征提取模型）、bottleneck_layer(即瓶颈层）、AdversarialNet(即对抗网络）和classifier_layer(即分类器）都移动到设备上
        # 定义损失函数（交叉熵损失函数用于分类）
        self.criterion = nn.CrossEntropyLoss()

    def _clusterInBatch(self,x, y, ds_labels):
        """cluster and order features into same-class group and domain label group"""
        batch_size = y.size()[0]

        with torch.no_grad():
            # 对类标签 y 进行排序
            sorted_y, indices = torch.sort(y)

            # 对域标签 ds_labels 进行排序，确保与类标签保持一致的顺序
            sorted_ds_labels = ds_labels[indices]

            # 根据排序后的 indices 对特征 x 进行排序
            sorted_x = torch.zeros_like(x)
            for idx, order in enumerate(indices):
                sorted_x[idx] = x[order]

            intervals = []
            ex = 0
            # 记录每个类标签的区间
            for idx, val in enumerate(sorted_y):
                if ex == val:
                    continue
                intervals.append(idx)
                ex = val
            intervals.append(batch_size)

            # 返回排序后的特征、类标签、域标签和类别区间
            x = sorted_x
            y = sorted_y
            return x, y, sorted_ds_labels, intervals

    def _shuffleBatch(self,output, proj, intervals):
        """generate shuffled batches"""
        output_2 = torch.zeros_like(output)
        feat_2 = torch.zeros_like(proj)
        output_3 = torch.zeros_like(output)
        feat_3 = torch.zeros_like(proj)
        ex = 0

        for end in intervals:
            shuffle_indices = torch.randperm(end - ex) + ex
            shuffle_indices2 = torch.randperm(end - ex) + ex
            for idx in range(end - ex):
                output_2[idx + ex] = output[shuffle_indices[idx]]
                feat_2[idx + ex] = proj[shuffle_indices[idx]]
                output_3[idx + ex] = output[shuffle_indices2[idx]]
                feat_3[idx + ex] = proj[shuffle_indices2[idx]]
            ex = end
        return output_2, output_3, feat_2, feat_3

    def _selfregLoss(self,
            output, feat, proj, intervals, c_scale, SelfReg_criterion=nn.MSELoss()
    ):
        output_2, output_3, feat_2, feat_3 = self._shuffleBatch(output, proj, intervals)

        lam = np.random.beta(0.5, 0.5)

        # mixup
        output_3 = lam * output_2 + (1 - lam) * output_3
        feat_3 = lam * feat_2 + (1 - lam) * feat_3

        # regularization
       # L_ind_logit = SelfReg_criterion(output, output_2)
        L_hdl_logit = SelfReg_criterion(output, output_3)
       # L_ind_feat = 0.3 * SelfReg_criterion(feat, feat_2)
        L_hdl_feat = 0.3 * SelfReg_criterion(feat, feat_3)

        return c_scale * (
                #lam * (L_ind_logit + L_ind_feat) +
                 (1 - lam) * (L_hdl_logit + L_hdl_feat)
        )

    def train(self):
        """
        训练过程：循环多个epoch，更新模型权重
        :return:
        """
        args = self.args

        step = 0
        best_acc = 0.0
        batch_count = 0
        batch_loss = 0.0
        batch_acc = 0
        step_start = time.time()

        for epoch in range(self.start_epoch, args.max_epoch):
            logging.info('-' * 5 + 'Epoch {}/{}'.format(epoch, args.max_epoch - 1) + '-' * 5)

            # 更新学习率
            if self.lr_scheduler is not None:
                logging.info('当前学习率: {}'.format(self.lr_scheduler.get_lr()))
            else:
                logging.info('当前学习率: {}'.format(args.lr))

            # 每个epoch有训练和验证两个阶段
            for phase in ['source_train', 'source_val', 'target_val']:
                epoch_start = time.time()
                epoch_acc = 0
                epoch_loss = 0.0
                epoch_length = 0

                # 设置模型为训练模式或验证模式
                if phase == 'source_train':
                    self.model.train()
                    if args.bottleneck:
                        self.bottleneck_layer.train()
                    if args.domain_adversarial:
                        self.AdversarialNet.train()
                    self.classifier_layer.train()
                else:
                    self.model.eval()
                    if args.bottleneck:
                        self.bottleneck_layer.eval()
                    if args.domain_adversarial:
                        self.AdversarialNet.eval()
                    self.classifier_layer.eval()

                for batch_idx, (inputs, labels_temp) in enumerate(self.dataloaders[phase]):
                    inputs = inputs.to(self.device)
                    labels_temp = labels_temp.to(self.device)
                    labels = labels_temp.long() % 100  # 用于分类任务的标签,这个是类标签
                    ds_labels = labels_temp.long() // 100  # 用于领域对抗任务的标签，这个是域标签

                    with torch.set_grad_enabled(phase == 'source_train'):
                        # 的作用是根据条件 phase == 'source_train' 来动态地启用或禁用 自动求导计算 (autograd)。它是 PyTorch 的上下文管理器，用于控制是否需要计算梯度。
                        # 正向传播
                        features = self.model(inputs)  # 特征提取
                        if args.bottleneck:
                            features = self.bottleneck_layer(features)  # 通过瓶颈层
                        outputs = self.classifier_layer(features)  # 分类层输出

                        # 计算损失
                        if phase != 'source_train' or epoch < args.middle_epoch:
                            logits = outputs
                            classifier_loss = self.criterion(logits, labels)
                            loss = classifier_loss
                        else:
                            if args.is_selfreg:
                                labels = labels.to(self.device)
                                ds_labels = ds_labels.to(self.device)
                                inputs,labels,ds_labels,intervals = self._clusterInBatch(inputs, labels, ds_labels)
                                feat = self.model(inputs)
                                proj = self.model.projection(feat)
                                if args.bottleneck:
                                   feat1 = self.bottleneck_layer(feat)
                                else:
                                   feat1 = feat

                                outputs = self.classifier_layer(feat1)
                                logits = outputs.narrow(0, 0, labels.size(0))
                                classifier_loss = self.criterion(logits, labels)
                                SelfReg_criterion = nn.MSELoss()
                                selfreg = self._selfregLoss(
                                    logits,
                                    feat,
                                    proj,
                                    intervals,
                                    c_scale=min(classifier_loss.item(), 1.0),
                                    SelfReg_criterion=SelfReg_criterion,
                                )
                            if args.domain_adversarial:
                                adversarial_label = ds_labels.to(self.device)
                                if args.is_selfreg:
                                    adversarial_out = self.AdversarialNet(feat1)
                                    adversarial_loss = self.criterion(adversarial_out, adversarial_label)
                                else:
                                    adversarial_out = self.AdversarialNet(features)
                                    adversarial_loss = self.criterion(adversarial_out, adversarial_label)
                                    selfreg=0
                            else:
                                adversarial_loss = 0
                                selfreg = 0
                            loss = classifier_loss + adversarial_loss+selfreg

                        # 计算准确率
                        pred = logits.argmax(dim=1)
                        correct = torch.eq(pred, labels).float().sum().item()
                        loss_temp = classifier_loss.item() * labels.size(0)  # 计算当前批次的损失总和
                        epoch_loss += loss_temp
                        epoch_acc += correct
                        epoch_length += labels.size(0)

                        # 训练阶段的反向传播和优化
                        if phase == 'source_train':
                            self.optimizer.zero_grad()
                            loss.backward()
                            self.optimizer.step()

                            # 打印训练信息
                            batch_loss += loss_temp
                            batch_acc += correct
                            batch_count += labels.size(0)
                            if step % args.print_step == 0:
                                batch_loss = batch_loss / batch_count
                                batch_acc = batch_acc / batch_count
                                temp_time = time.time()
                                train_time = temp_time - step_start
                                step_start = temp_time
                                batch_time = train_time / args.print_step if step != 0 else train_time
                                sample_per_sec = 1.0 * batch_count / train_time
                                logging.info('Epoch: {} [{}/{}], Train Loss: {:.4f} Train Acc: {:.4f},'
                                             '{:.1f} examples/sec {:.2f} sec/batch'.format(
                                    epoch, batch_idx * len(labels), len(self.dataloaders[phase].dataset),
                                    batch_loss, batch_acc, sample_per_sec, batch_time
                                ))
                                batch_acc = 0
                                batch_loss = 0.0
                                batch_count = 0
                            step += 1

                # 记录当前epoch的结果
                epoch_loss = epoch_loss / epoch_length
                epoch_acc = epoch_acc / epoch_length
                logging.info('Epoch: {} {}-Loss: {:.4f} {}-Acc: {:.4f}, Cost {:.1f} sec'.format(
                    epoch, phase, epoch_loss, phase, epoch_acc, time.time() - epoch_start
                ))

                # 保存基于验证集准确率的最佳模型
                if phase == 'target_val':
                    model_state_dic = self.model_all.state_dict()
                    if (epoch_acc > best_acc or epoch > args.max_epoch - 2) and (epoch > args.middle_epoch - 1):
                        best_acc = epoch_acc
                        logging.info("保存最佳模型，第{}轮，准确率 {:.4f}".format(epoch, epoch_acc))
                        torch.save(model_state_dic,
                                   os.path.join(self.save_dir, '{}-{:.4f}-best_model.pth'.format(epoch, best_acc)))

            # 更新学习率
            if self.lr_scheduler is not None:
                self.lr_scheduler.step()
