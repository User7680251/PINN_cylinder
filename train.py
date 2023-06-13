import os
import torch

from tasks.task1_ODE import train as train_task1
from tasks.task2_laplacian import train as train_task2
from tasks.task3_cylinder_steady_flow import train as train_task3

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'


if __name__ == '__main__':
    train_task3()