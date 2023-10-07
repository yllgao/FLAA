# Copyright 2021 Peng Cheng Laboratory (http://www.szpclab.com/) and FedLab Authors (smilelab.group)

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

#     http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from copy import deepcopy
import torch

from torch import nn
from tqdm import tqdm
from ...core.client.trainer import ClientTrainer, SerialClientTrainer
from ...utils import Logger, SerializationTool
from torch.optim.lr_scheduler import CyclicLR

import time


class SGDClientTrainer(ClientTrainer):  # 单个客户端训练
    """Client backend handler, this class provides data process method to upper layer.

    Args:
        model (torch.nn.Module): PyTorch model.
        cuda (bool, optional): use GPUs or not. Default: ``False``.
        device (str, optional): Assign model/data to the given GPUs. E.g., 'device:0' or 'device:0,1'. Defaults to None.
        logger (Logger, optional): :object of :class:`Logger`.
    """

    def __init__(self,
                 model: torch.nn.Module,
                 cuda: bool = False,
                 device: str = None,
                 logger: Logger = None):
        self.loss_func = nn.CrossEntropyLoss()  # 交叉熵损失函数
        super(SGDClientTrainer, self).__init__(model, cuda, device)

        self._LOGGER = Logger() if logger is None else logger

    @property
    def uplink_package(self):
        """Return a tensor list for uploading to server.

            This attribute will be called by client manager.
            Customize it for new algorithms.
        """
        return [self.model_parameters]

    def setup_dataset(self, dataset):
        self.dataset = dataset

    def setup_optim(self, epochs, batch_size, lr):  # 由unit_test传入参数
        """Set up local optimization configuration.

        Args:
            epochs (int): Local epochs.
            batch_size (int): Local batch size. 
            lr (float): Learning rate.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr)
        self.criterion = torch.nn.CrossEntropyLoss()

    def local_process(self, payload, id):
        model_parameters = payload[0]
        train_loader = self.dataset.get_dataloader(id, self.batch_size)
        self.train(model_parameters, train_loader)

    def train(self, model_parameters, train_loader) -> None:
        """Client trains its local model on local dataset.

        Args:
            model_parameters (torch.Tensor): Serialized model parameters.
        """
        SerializationTool.deserialize_model(
            self._model, model_parameters)  # load parameters
        self._LOGGER.info("Local train procedure is running")
        for ep in range(self.epochs):
            self._model.train()
            for data, target in train_loader:
                if self.cuda:
                    data, target = data.cuda(self.device), target.cuda(self.device)

                outputs = self._model(data)
                loss = self.criterion(outputs, target)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
        self._LOGGER.info("Local train procedure is finished")


class SGDSerialClientTrainer(SerialClientTrainer):  # 多个客户端训练
    """
    Train multiple clients in a single process.

    Customize :meth:`_get_dataloader` or :meth:`_train_alone` for specific algorithm design in clients.

    Args:
        model (torch.nn.Module): Model used in this federation.
        num_clients (int): Number of clients in current trainer.
        cuda (bool): Use GPUs or not. Default: ``False``.
        device (str, optional): Assign model/data to the given GPUs. E.g., 'device:0' or 'device:0,1'. Defaults to None.
        logger (Logger, optional): Object of :class:`Logger`.
        personal (bool, optional): If Ture is passed, SerialModelMaintainer will generate the copy of local parameters list and maintain them respectively. These paremeters are indexed by [0, num-1]. Defaults to False.
    """

    def __init__(self, model, num_clients, cuda=False, device=None, logger=None, personal=False) -> None:
        super().__init__(model, num_clients, cuda, device, personal)

        self._LOGGER = Logger() if logger is None else logger
        self.cache = []
        self.round = 1

    def setup_dataset(self, dataset):
        self.dataset = dataset

    def setup_optim(self, epochs, batch_size, lr):
        """Set up local optimization configuration.

        Args:
            epochs (int): Local epochs.
            batch_size (int): Local batch size. 
            lr (float): Learning rate.
        """
        self.epochs = epochs
        self.batch_size = batch_size
        self.criterion = torch.nn.CrossEntropyLoss()
        self.lr = lr
        self.optimizer = torch.optim.SGD(self._model.parameters(), lr)  # 学习率策略SGD
        self.optimizerRMS = torch.optim.RMSprop(self._model.parameters(),  # 学习率策略RMSprop
                                                lr,
                                                alpha=0.99,
                                                eps=1e-08,
                                                weight_decay=0,
                                                momentum=0,
                                                centered=False)

        # 初始化Adam的超参数
        self.betas = (0.9, 0.999)  # 设置动量参数betas为(0.9, 0.999)，用于计算梯度的一阶和二阶矩估计。
        self.eps = 1e-8  # 设置epsiloneps为1e-8，用于数值稳定性。
        self.weight_decay = 0  # 设置权重衰减weight_decay为0，用于L2正则化。
        self.beta1 = 0.9
        self.beta2 = 0.999
        self.eps = 1e-8

        # 初始化模型参数和优化器状态
        self.optimizerAdam = torch.optim.Adam(self._model.parameters(), lr=0.05, betas=self.betas,
                                              eps=self.eps, weight_decay=self.weight_decay)
        self.optimizerAdam_state = self.optimizerAdam.state_dict()
        self.ExpLR = torch.optim.lr_scheduler.ExponentialLR(self.optimizerAdam, gamma=0.98)
        # 初始化CyclicLR的超参数
        self.base_lr = 0.001  # 设置基础学习率base_lr为0.001，定义学习率在一个周期内的最小值。
        self.max_lr = 0.005  # 设置最大学习率max_lr为0.01，定义学习率在一个周期内的最大值。
        self.step_size = 400  # 设置步长step_size为400，表示每个半周期内的步数。

        self.loss_locals_list = []

        # 设置Adam的超参数
        adam_lr = 0.001
        adam_weight_decay = 0.0001
        adam_beta1 = 0.9
        adam_beta2 = 0.999
        adam_eps = 1e-8
        return [self.loss_locals_list]

        # 设置CyclicLR的超参数
        # cyclic_base_lr = 0.001
        # cyclic_max_lr = 0.01
        # step_size = len(dataloader) * num_epochs

    # params(iterable)：可用于迭代优化的参数或者定义参数组的dicts。
    # lr(float, optional) ：学习率(默认: 1e-3)，更新梯度的时候使用
    # betas(Tuple[float, float], optional)：用于计算梯度的平均和平方的系数(默认: (0.9, 0.999))
    # eps(float, optional)：为了提高数值稳定性而添加到分母的一个项(默认: 1e-8)
    # weight_decay(float, optional)：权重衰减(如L2惩罚)(默认: 0)，针对最后更新参数的时候，给损失函数中的加的一个惩罚参数，更新参数使用

    @property
    def uplink_package(self):
        package = deepcopy(self.cache)
        self.cache = []
        return package

    def local_process(self, payload, id_list):
        model_parameters = payload[0]
        for id in tqdm(id_list, desc=">>> Local training"):
            data_loader = self.dataset.get_dataloader(id, self.batch_size)
            pack = self.train(model_parameters, data_loader)
            self.cache.append(pack)
            self.round += 1

    # def adjust_learning_rate(self, start_lr):  # 学习率周期性下降
    #     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    #     lr = start_lr * (0.1 ** (self.round // 850))  # 学习率调整策略
    #     for param_group in self.optimizer.param_groups:
    #         param_group['lr'] = lr
    #     # 打印当前学习率
    #     print("当前学习率：", self.optimizer.param_groups[0]['lr'])

    def update_learning_rate(self, optimizerAdam):
        cycle = 1 + ((self.round - 1) * 5) // (2 * self.step_size)
        x = abs(((self.round - 1) * 5) / self.step_size - 2 * cycle + 1)
        lr = self.base_lr + (self.max_lr - self.base_lr) * max(0, (1 - x))

        for param_group in optimizerAdam.param_groups:
            param_group['lr'] = lr
        # 打印当前学习率
        print("当前学习率：", self.optimizerAdam.param_groups[0]['lr'])

    def step_decay_learning_rate(self, optimizerAdam, current_step, decay_steps, decay_rate):
        """分段常数衰减学习率函数"""
        lr = optimizerAdam.param_groups[0]['lr']  # 获取当前学习率
        if current_step > 0 and current_step % decay_steps == 0:
            lr *= decay_rate  # 学习率衰减
            for param_group in optimizerAdam.param_groups:
                param_group['lr'] = lr  # 更新优化器中的学习率
        # 打印当前学习率
        print("当前学习率：", self.optimizerAdam.param_groups[0]['lr'])

    def exponential_decay_learning_rate(self, optimizer, current_step, decay_factor):
        """自然指数衰减学习率函数"""
        lr = optimizer.param_groups[0]['lr']  # 获取当前学习率
        lr *= decay_factor ** current_step  # 学习率衰减
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr  # 更新优化器中的学习率
        # 打印当前学习率
        print("当前学习率：", self.optimizerAdam.param_groups[0]['lr'])

    def adjust_learning_rate(self, optimizer, round_, current_loss, prev_loss, threshold):
        # 计算当前损失和前一个损失的比率
        loss_ratio = current_loss / prev_loss
        print(f'loss_ratio:', loss_ratio)

        d = abs(loss_ratio - 1) ** 2  # 变化率-1的平方
        print(f'd:', d)
        if d < 1:
            d = d + 1  # 系数大于1
        a = (1 / d ** (round_ ** 0.5))  # round越大，a越小，变化越小
        print(f'a:', a)
        # for i, param in enumerate(parameters):
        # if grads[i] is None:
        # continue
        # u = grads[i]/grads[i-1]

        if round_ % 100 == 0:
            new_lr = 0.001
            for param_group in optimizer.param_groups:
                param_group['lr'] = new_lr  # 更新优化器中的学习率
        # 如果损失下降幅度超过阈值，降低学习率
        elif round_ % 100 != 0 and threshold > loss_ratio > 0.7:
            new_lr = optimizer.param_groups[0]['lr'] * (1 - (1 / d ** (round_ ** 0.5)))
            if new_lr > 0.0003:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr  # 更新优化器中的学习率
                print(f'Learning rate reduced to {new_lr}')
        # 如果损失下降幅度过小，提升学习率
        elif round_ % 100 != 0 and (loss_ratio > 30 or loss_ratio < 0.3):
            new_lr = optimizer.param_groups[0]['lr'] * (1 + (1 / (d ** (round_ ** 0.5))))
            if new_lr < 0.001:
                for param_group in optimizer.param_groups:
                    param_group['lr'] = new_lr  # 更新优化器中的学习率
                print(f'Learning rate raised to {new_lr}')
        else:
            new_lr = optimizer.param_groups[0]['lr']
            print(f'Learning rate unchanged: {new_lr}')

    def adam(parameters, grads, exp_avg, exp_avg_sq, lr, weight_decay, beta1, beta2, eps):
        for i, param in enumerate(parameters):
            if grads[i] is None:
                continue

            grad = grads[i]
            exp_avg[i] = beta1 * exp_avg[i] + (1 - beta1) * grad
            exp_avg_sq[i] = beta2 * exp_avg_sq[i] + (1 - beta2) * grad ** 2

            bias_correction1 = 1 - beta1 ** (i + 1)
            bias_correction2 = 1 - beta2 ** (i + 1)

            step_size = lr * (bias_correction2.sqrt() / bias_correction1)

            param.data -= step_size * (exp_avg[i] / (exp_avg_sq[i].sqrt() + eps))

    def train(self, model_parameters, train_loader):  # 客户端训练
        """Single round of local training for one client.

        Note:
            Overwrite this method to customize the PyTorch training pipeline.

        Args:
            model_parameters (torch.Tensor): serialized model parameters.
            train_loader (torch.utils.data.DataLoader): :class:`torch.utils.data.DataLoader` for this client.
        """
        global epoch_loss
        self.set_model(model_parameters)
        self._model.train()

        # 初始化模型参数
        parameters = [torch.randn(2, 2, requires_grad=True), torch.randn(3, 3, requires_grad=True)]
        # 设置超参数
        decay_steps = 100  # 衰减步数
        decay_rate = 0.1  # 衰减率
        decay_factor = 0.9  # 衰减因子
        # 初始化梯度和状态
        grads = [torch.randn_like(param) for param in model_parameters]
        exp_avg = [torch.zeros_like(param) for param in parameters]
        exp_avg_sq = [torch.zeros_like(param) for param in parameters]
        epoch_loss = []
        epoch_loss_list = []  # 每迭代一次的损失

        loss_locals = []  # 局部预测损失  对于每一个epoch，初始化worker的损失
        batch_loss = []  # 为了提高计算效率，不会对每个client进行loss统计，统计batch_loss
        # if self.round > 2:
        #     self.adjust_learning_rate(self.optimizerAdam, self.round, self.loss_locals_list[-1],
        #                               self.loss_locals_list[-2],
        #                               threshold=1.3)
        for epoch in range(self.epochs):
            # self.adjust_learning_rate(self.lr)
            for data, target in train_loader:
                if self.cuda:
                    data = data.cuda(self.device)
                    target = target.cuda(self.device)

                output = self.model(data)
                loss = self.criterion(output, target)
                # self.optimizer.zero_grad()  # 将这一轮的梯度清零，防止其影响下一轮的更新
                loss.backward()  # 反向计算出各参数的梯度
                self.optimizerAdam.step()  # 更新网络中的参数
                self.optimizerAdam.zero_grad()  # 将这一轮的梯度清零，防止其影响下一轮的更新
                batch_loss.append(loss.item())  # 客户端中每个batch的loss# 保存损失值
                # 更新学习率
                # self.update_learning_rate(self.optimizerAdam)

            # self.step_decay_learning_rate(self.optimizerAdam, self.round, decay_steps, decay_rate)
            # self.exponential_decay_learning_rate(self.optimizerAdam, self.round, decay_factor)
            # self.ExpLR.step()

            # 更新学习率
            # self.scheduler.step()

            epoch_loss.append(sum(batch_loss) / len(batch_loss))  # 每个epoch的平均loss
        loss_locals = sum(epoch_loss) / len(epoch_loss)  # 每轮每个客户端的本地平均loss
        self.loss_locals_list.append(loss_locals)
        # for i, loss in enumerate(loss_locals_list):
        #     print('Local {} loss: {:.3f}'.format(i, loss))

        # 打印当前学习率
        print("当前学习率：", self.optimizerAdam.param_groups[0]['lr'])
        # self.loss_avg = sum(loss_locals) / len(loss_locals)  # 每轮所有客户端的平均loss
        if self.round % 500 == 0:
            print("round:{},Lr:{:.2E}".format(self.round, self.optimizer.state_dict()['param_groups'][0]['lr']))
        # print('Avg loss {:.3f}'.format(self.loss_avg))
        print('Local loss {:.3f}'.format(loss_locals))
        # print('Epoch loss {:.3f}'.format(epoch_loss))
        return [self.model_parameters]
