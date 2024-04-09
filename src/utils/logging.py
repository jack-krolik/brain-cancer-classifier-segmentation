import pathlib
import wandb
from datetime import datetime
import numpy as np
import uuid
import json
from typing import Dict
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import torch

from src.utils.config import TrainingConfig

LOGS_DIR = pathlib.Path(__file__).parent.parent.parent / "logs"
MODELS_DIR = pathlib.Path(__file__).parent.parent.parent / "model_registry"

class LoggerMixin:
    def __init__(self, config: TrainingConfig):
        self.config = config
    
    def init(self, **kwargs):
        """
        Initialize the logger with the given configuration

        Any setup that needs to be done before training should be done here
        """
        raise NotImplementedError("Method 'init' not implemented.")

    def finish(self):
        """
        Finish the logging process and clean up any resources

        Any cleanup that needs to be done after training should be done here

        This doesn't necesarily mean that the logger object is destroyed (it can be reused)
        However, init followed by finish when training multiple models to not mix up logs
        """
        raise NotImplementedError("Method 'finish' not implemented.")

    def log_metrics(self, metrics: Dict[str, float], step: int = None):
        """
        Log the metrics to the logger

        Args:
        - metrics (dict): a dictionary containing the metrics to log
        - step (int): the step number to log the metrics at
        """
        raise NotImplementedError("Method 'log_metrics' not implemented.")
    
    def save_model(self, model: torch.nn.Module, id: str):
        """
        Save the model to the logger

        Args:
        - path (pathlib.Path): the path to save the model to
        """
        raise NotImplementedError("Method 'save_model' not implemented.")

    def plot_metrics(self):
        """
        Plot the metrics that have been logged so far

        This method is optional and can be implemented if the logger supports plotting
        """
        pass
    

class WandbLogger(LoggerMixin):
    def __init__(self, config: TrainingConfig, wandb_API_key: str, project_name: str, tags: list):
        super().__init__(config)
        self.project_name = project_name
        self.tags = tags

        # Initialize wandb
        wandb.login(key=wandb_API_key, verify=True)

    def init(self, **kwargs):
        self.run = wandb.init(config=self.config.flatten(), project=self.project_name, tags=self.tags **kwargs)

    def finish(self):
        if self.run:
            self.run.finish()
        else:
            raise ValueError("Wandb run object not found. Make sure to call init before finish")

    def log_metrics(self, metrics, step=None):
        assert self.run, "Wandb run object not found. Make sure to call init before logging metrics"
        self.run.log(metrics, step=step) # Log the metrics to wandb

    def save_model(self, path: pathlib.Path):
        assert self.run, "Wandb run object not found. Make sure to call init before saving model"
        self.run.save(str(path)) # Save the model to wandb
    
class LocalLogger(LoggerMixin):
    def __init__(self, config: TrainingConfig, run_group: str = None, checkpointing: bool = False):
        super().__init__(config)
        self.metrics = {}
        self.checkpointing = checkpointing

        self.run_group = "_".join(run_group.split()) if run_group else 'Experiment'

    def init(self, **kwargs):
        # Initialize the metrics dictionary
        self.metrics = {}

        self.save_dir = LOGS_DIR / self.run_group if self.run_group else LOGS_DIR

        # create a random seed for the run id if not provided
        run_id = kwargs.get('name', None)
        if not run_id:
            run_id = str(uuid.uuid4())[:8]
        
        run_id = run_id + "_" + datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        # create a directory to store the logs
        self.save_dir.mkdir(exist_ok=True)
        self.log_dir = self.save_dir / run_id
        self.log_dir.mkdir(exist_ok=True)

        if self.checkpointing:
            self.model_dir = MODELS_DIR / self.run_group if self.run_group else MODELS_DIR
            self.model_dir.mkdir(exist_ok=True)
            self.model_dir = self.model_dir / run_id
            self.model_dir.mkdir(exist_ok=True)

    def finish(self):
        # save the metrics to a file
        with open(self.log_dir / "metrics.json", "w") as f: # should this be a csv file?
            json.dump(self.metrics, f)

    def log_metrics(self, metrics, step=None):
        if step is None:
            step = len(self.metrics) + 1
        self.metrics[step] = metrics
    
    def save_model(self, model: torch.nn.Module, model_id: str):
        torch.save(model.state_dict(), self.model_dir / f"{model_id}.pth")
    
    def plot_metrics(self):
        metrics_df = pd.DataFrame(self.metrics).T
        metrics_df.index.name = "Step"

        num_metrics = len(metrics_df.columns)
        n_rows = int(np.ceil(num_metrics / 2))

        fig = make_subplots(rows=n_rows, cols=2, subplot_titles=metrics_df.columns)

        for i, metric in enumerate(metrics_df.columns, start=1):
            row = (i+1) // 2
            col = 2 if i % 2 == 0 else 1

            fig.add_trace(go.Scatter(x=metrics_df.index, y=metrics_df[metric], mode='lines', name=metric), row=row, col=col)

        fig.update_layout(title='Metrics', xaxis_title='Epoch', yaxis_title='Value')
        fig.write_html(str(self.log_dir / "metrics.html"))
        fig.show()
        