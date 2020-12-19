# Gradient Based CRAIG
Project conducted by Xiaofeng Lin under supervision of Dr.Baharan Mirzasoleiman. 

This project aims for improve training performance of Coreset for Accelerating Increment Gradient descent(CRAIG) algorithm. The essential part of the modification involves replacing the logits(forward pass output - target) with per example gradients on neurons/connections selected by Early Bird Ticket or Gradient Signal Preservation algorithm.

The github repositories quoted by this project:

CRAIG: https://github.com/baharanm/craig

Autohack: https://github.com/cybertronai/autograd-hacks

Early Bird tickets: https://github.com/RICE-EIC/Early-Bird-Tickets

GraSP: https://github.com/alecwangcq/GraSP


Usage: 

0, Search for EB tickets
```
python3 train_shallow_mnist.py -s 0.01 -w -b 512 -g --smtk 0 --save_subset --save-every 20 --save-dir 'output' --searching
```

1. Run CRAIG
```
python3 train_shallow_mnist.py -s 0.01 -w -b 512 -g --smtk 0 --save_subset --save-every 20 --save-dir 'output' --imp_type 1
```

2. Run EB CRAIG
```
python3 train_shallow_mnist.py -s 0.01 -w -b 512 -g --smtk 0 --save_subset --save-every 20 --save-dir 'output' --imp_type 2 --EB_path "/output/EB-95-5.pth.tar" -ppct 0.03
```

3. Run GraSP CRAIG
```
python3 train_shallow_mnist.py -s 0.01 -w -b 512 -g --smtk 0 --save_subset --save-every 20 --save-dir 'output' --imp_type 3 -ppct 0.003
```

4. Concatenation: Second to last layer + Logits
```
python3 train_shallow_mnist.py -s 0.01 -w -b 512 -g --smtk 0 --save_subset --save-every 20 --save-dir 'output' --imp_type 4 -ppct 0.003
```

5. Concatenation: First + second to last layers + Logits
```
python3 train_shallow_mnist.py -s 0.01 -w -b 512 -g --smtk 0 --save_subset --save-every 20 --save-dir 'output' --imp_type 5 -ppct 0.003
```

For CIFAR10 dataset, replace `train_shallow_mnist.py` with ` train_shallow_cifar10.py`.
