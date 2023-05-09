from .base_evaluator import BaseEvaluator
from collections import OrderedDict


class L1Evaluator(BaseEvaluator):
    @staticmethod
    def modify_commandline_options(parser, is_train):
        return parser

    def __init__(self, opt):
        super().__init__(opt)
        self.opt = opt

    def prepare_evaluation(self, phase, dataloader, fn_model_forward, name):
        pass

    def evaluate_current_batch(self, data, fn_model_forward, name):
        pass

    def should_stop_evaluation(self, num_cum_samples):
        return True

    def finish_evaluation(self, dataloader, fn_model_forward, name):
        l1s = OrderedDict()
        l1s['L1train'] = 0
        l1s['L1test'] = 1
        return l1s
