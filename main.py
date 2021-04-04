"""

"""

import torchvision.models as models

from training.training_manager import train_model
from dataset.cifar_data_loaders import Cifar10Dataset
from training.utils.utils import save_model, load_model, load_checkpoint
from training.metrics.metrics import print_accuracy_per_class, print_accuracy, count_model_parameters
import copy
import time

if __name__ == '__main__':
    # TODO Add as config
    model_dir = f"E:/Git/{project_name}/models/trained_models/"
    model_name = ""
    checkpoint_dir = f"E:/Git/{project_name}/training/checkpoints/checkpoint_{model_name}.pt"
    batch_size = 15
    epochs = 36
    cifar10_data = Cifar10Dataset(batch_size=batch_size)
    classes = cifar10_data.classes

    """
    M A I N
    """

    ResNet34 100 epochs
    resNet34 = models.resnet34(pretrained=False)
    print(f"Parameters {count_model_parameters(resNet34, False)}")
    start_time = time.time()
    save_model(train_model(epochs, resNet34, "resNet34", cifar10_data, batch_size, model_dir), "resNet34", model_dir)
    print(f"Training time for {epochs} epochs : {time.time() - start_time}")
    print_accuracy_per_class(resNet34, classes, batch_size, cifar10_data.test_loader)
    print_accuracy(resNet34, classes, batch_size, cifar10_data.test_loader)


