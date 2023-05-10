from .base_evaluator import BaseEvaluator
from collections import OrderedDict
from torch import nn


class L1Evaluator(BaseEvaluator):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.lossObj = nn.L1Loss()
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

        loss = self.lossObj(self.target, self.out)

        l1s['L1%s' % self.current_phase] = loss.item()
        return l1s
