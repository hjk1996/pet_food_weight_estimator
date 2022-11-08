import tensorboard
import torch

from utils import indice_to_name_mapper

if __name__ == "__main__":
    mapper = indice_to_name_mapper("./data/indice_name_mapping.csv")

    print(mapper)
