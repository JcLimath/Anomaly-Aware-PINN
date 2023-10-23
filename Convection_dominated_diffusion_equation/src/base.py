


from multiprocessing import reduction
from typing import Callable, Dict, Iterable
import torch
import torch.nn as nn
import os
import torch.nn.functional as F

from .utils import AverageMeter, ProgressMeter, timemeter, getLogger
from .loss_zoo import mse_loss
from .config import DEVICE, SAVED_FILENAME, PRE_BEST


class Coach:
    
    def __init__(
        self, model: nn.Module,
        loss_func: Callable, 
        oracle: Callable,
        optimizer: torch.optim.Optimizer, 
        learning_policy: "learning rate policy",
        device: torch.device = DEVICE,
    ):
        self.model = model
        self.device = device
        self.loss_func = loss_func
        self.oracle = oracle
        self.optimizer = optimizer
        self.learning_policy = learning_policy
        self.loss = AverageMeter("Loss")
        self.progress = ProgressMeter(self.loss)

        self._best = float('inf')

   
    def save_best(self, mse: float, path: str, prefix: str = PRE_BEST):
        if mse < self._best:
            self._best = mse
            self.save(path, '_'.join((prefix, SAVED_FILENAME)))
            return 1
        else:
            return 0

    def check_best(
        self, mse: float,
        path: str, epoch: int = 8888
    ):
        logger = getLogger()
        if self.save_best(mse, path):
            logger.debug(f"[Coach] Saving the best nat ({mse:.6f}) model at epoch [{epoch}]")
        
    def save(self, path: str, filename: str = SAVED_FILENAME) -> None:
        torch.save(self.model.state_dict(), os.path.join(path, filename))

    @timemeter("Train/Epoch")
    def train(
        self, 
        trainloader: Iterable,
        boundary: Dict,
        leverage: float,
        *, epoch: int = 8888
    ) -> float:

        def sample_select_rate(t, T_max):
            """

            :param t: now epoch
            :param T: sum epoch
            :return: select rate
            """
            rate_tem = (t+1) / (5 * T_max) + 0.7
            if rate_tem > 0.997:
                rate = 0.997
            else:
                rate = rate_tem
            return rate

        self.progress.step() # reset the meter
        self.model.train()
        bx, by, bg1, bg2 = boundary['x'], boundary['y'], boundary['g1'], boundary['g2']
        for x, y, g1, g2 in trainloader:
            x = x.to(self.device)
            y = y.to(self.device)
            g1 = g1.to(self.device)
            g2 = g2.to(self.device)
            bx = bx.to(self.device)
            by = by.to(self.device)

            def closure():
                target = self.oracle(x, y, g1, g2)
                x.requires_grad_(True)
                y_pred = self.model(x)
                g1_pred = torch.autograd.grad(
                    y_pred, x, 
                    grad_outputs=torch.ones_like(y_pred),
                    create_graph=True
                )[0]
                g2_pred = torch.autograd.grad(
                g1_pred, x, 
                grad_outputs=torch.ones_like(g1_pred),
                create_graph=True
                )[0]
                x.requires_grad_(False)
                pred = self.oracle(x, y_pred, g1_pred, g2_pred)

                loss1 = F.mse_loss(pred, target, reduce=False)
                loss1_number, _ = loss1.shape
                

                #loss_mean = self.loss_func(pred, target, reduction='mean')
                loss_star = loss1
                loss_mean = torch.mean(loss_star)
                loss_d = torch.sqrt(torch.var(loss_star))

                l_control = 3
                vector_up = loss_mean * torch.ones_like(loss_star) + l_control * loss_d * torch.ones_like(loss_star)  # up
                
                relu = nn.ReLU(inplace=True)
                m = torch.sum(relu(loss1 - vector_up)).cpu().detach().numpy()             


                
                if epoch > 1:
                    if m > 0 :
                        select_rate = sample_select_rate(epoch, 150000)
                        select_number = round(select_rate * loss1_number)
                        loss1_tem, _ = torch.topk(loss1, select_number, dim=0, largest=False)
                        loss = torch.mean(loss1_tem)
                    else:
                        loss = loss_mean    #

                else:
                    select_number = round(0.8 * loss1_number)
                    loss1_tem, _ = torch.topk(loss1, select_number, dim=0, largest=False)  #
                    loss = torch.mean(loss1_tem)
                
                
                    


                by_pred = self.model(bx)
                bloss = mse_loss(by_pred, by, reduction="mean")
                loss = loss + bloss * leverage
                self.optimizer.zero_grad()
                loss.backward()
                return loss

            loss = self.optimizer.step(closure)

            self.loss.update(loss.item(), x.size(0), mode="mean")

        self.progress.display(epoch=epoch) 
        self.learning_policy.step() # update the learning rate
        return self.loss.avg
    

    @timemeter("Evaluation")
    def evaluate(self, validloader: Iterable, *, epoch: int = 8888):

        self.progress.step() # reset the meter
        self.model.eval()
        for x, y, g1, g2 in validloader:
            x = x.to(self.device)
            y = y.to(self.device)
            g1 = g1.to(self.device)
            g2 = g2.to(self.device)

            target = self.oracle(x, y, g1, g2)
            x.requires_grad_(True)
            y_pred = self.model(x)
            g1_pred = torch.autograd.grad(
                y_pred, x, 
                grad_outputs=torch.ones_like(y_pred),
                create_graph=True
            )[0]
            g2_pred = torch.autograd.grad(
               g1_pred, x, 
               grad_outputs=torch.ones_like(g1_pred),
               retain_graph=False
            )[0]
            x.requires_grad_(False)
            pred = self.oracle(x, y_pred, g1_pred, g2_pred) 
            loss = self.loss_func(pred, target)
            

            self.loss.update(loss.item(), x.size(0), mode="mean")

        # self.progress.display(epoch=epoch) 

        return self.loss.avg
