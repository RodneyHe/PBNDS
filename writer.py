from pathlib import Path
import torch
import numpy as np

from torch.utils.tensorboard import SummaryWriter

class Writer(object):
    writer = None

    @staticmethod
    def set_writer(results_dir):
        
        if isinstance(results_dir, str):
            results_dir = Path(results_dir)
        
        results_dir.mkdir(exist_ok=True, parents=True)
        results_dir.joinpath("images").mkdir(exist_ok=True)
        results_dir.joinpath("weights").mkdir(exist_ok=True)
        Writer.writer = SummaryWriter(str(results_dir))

    @staticmethod
    def add_scalar(tag, val, step):
        if isinstance(val, torch.Tensor):
            val = val.item()

        Writer.writer.add_scalar(tag, val, step)

    @staticmethod
    def add_image(tag, val, step):
        
        if len(val.shape) == 4 and val.shape[0] == 1:
            val = val.squeeze()
        
        val = (val * 255).clamp(0, 255)
        val = np.uint8(np.round(val.detach().cpu().numpy()))

        Writer.writer.add_image(tag, val, step)

    @staticmethod
    def add_graph(model, *graph_inputs):
        Writer.writer.add_graph(model, *graph_inputs)

    @staticmethod
    def flush():
        Writer.writer.flush()