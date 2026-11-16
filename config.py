import os

cifar_10_root = os.environ.get('CIFAR10_ROOT', 'D:/data/cifar10' if os.path.exists('D:/data/cifar10') else 'cifar10/')
cifar_100_root = os.environ.get('CIFAR100_ROOT', 'D:/data/cifar100' if os.path.exists('D:/data/cifar100') else 'cifar100/')
imagenet_root = os.environ.get('IMAGENET_ROOT', '/ImageNet/')
exp_root = 'dev_outputs/'
cub_root = os.environ.get('CUB_ROOT', 'D:/data/CUB_200_2011')
cars_root = os.environ.get('CARS_ROOT', 'D:/data/stanford_cars')
aircraft_root = os.environ.get('AIRCRAFT_ROOT', 'D:/data/fgvc_aircraft')
herbarium_root = os.environ.get('HERBARIUM_ROOT', 'D:/data/Herbarium 19')

