from .base_evaluator import BaseEvaluator
from collections import OrderedDict
from torch import nn


class PearsonEvaluator(BaseEvaluator):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt

    def prepare_evaluation(self, phase, dataloader, fn_model_forward, name):
        self.current_phase = phase

    def evaluate_current_batch(self, data, fn_model_forward, name):
        self.target = data["image"].cuda()
        self.out = fn_model_forward(data, "forward")["fake_B"]

    def should_stop_evaluation(self, num_cum_samples):
        return True

    def finish_evaluation(self, dataloader, fn_model_forward, name):
        l1s = OrderedDict()

        #loss = self.lossObj(self.target, self.out)
        # subtract mean
        target_ms = self.target - torch.mean(self.target, dim=(2,3))
        out_ms = self.out - torch.mean(self.out, dim=(2,3))
        loss = torch.sum(target_ms * out_ms, dim=1) / torch.sqrt( torch.sum(target_ms**2) * torch.sum(out_ms**2) )

        pearsons['Pearson%s' % self.current_phase] = loss.item()
        return pearsons
