import pathlib
import wandb
import torch
import uuid
import json
from typing import Dict

from src.utils.config import TrainingConfig

LOGS_DIR = pathlib.Path(__file__).parent.parent.parent / "logs"

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
        self.run = wandb.init(config=self.config.flatten(), project=self.project_name, tags=self.tags, **kwargs)

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
    def __init__(self, config: TrainingConfig):
        super().__init__(config)
        self.metrics = {}

    def init(self, **kwargs):
        # Initialize the metrics dictionary
        self.metrics = {}

        # create a random seed for the run id if not provided
        run_id = kwargs.get('run_id', None)
        if not run_id:
            run_id = str(uuid.uuid4())
            kwargs['run_id'] = run_id

        # create a directory to store the logs
        LOGS_DIR.mkdir(parents=True, exist_ok=True)
        self.log_dir = LOGS_DIR / run_id
        self.log_dir.mkdir(exist_ok=True)

    def finish(self):
        # save the metrics to a file
        with open(self.log_dir / "metrics.json", "w") as f: # should this be a csv file?
            json.dump(self.metrics, f)

    def log_metrics(self, metrics, step=None):
        if step is None:
            step = len(self.metrics) + 1
        self.metrics[step] = metrics

    def plot_metrics(self):
        raise NotImplementedError("Method 'plot_metrics' not implemented.")
