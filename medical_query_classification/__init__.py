from .cli import tyro_app
from .data_augmentation import augmentation_reflex, augmentation_transit, count_label_num, sample
from .evaluate import main as evaluate
from .fill_result import main as fill_result
from .train import main as train

def cli():
    tyro_app.cli()

