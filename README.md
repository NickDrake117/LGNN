# LGNN - Layered Graph Neural Network Model - TF 2.x
This repo contains a Tensorflow 2.x implementation of the Layered Graph Neural Network Model.

- **Authors:** [Niccolò Pancino](http://sailab.diism.unisi.it/people/niccolo-pancino/)

## Install
### Requirements
The LGNN framework requires the packages **tensorflow**, **numpy**, **pandas**, **scikit-learn**, **matplotlib**.

To install the requirements you can use the following command:

    pip install -U -r requirements.txt


## Simple usage example
In the following scripts, LGNN is by default a 5-layered GNN for binary node-focused classification task on graphs with random nodes/edges/targets.

Open the script `starter_lgnn` and set parameters in section *SCRIPT OPTIONS* to change dataset and/or LGNN architecture and behaviour.

In particular, set `use_MUTAG=True` to get the real-world dataset MUTAG for solving a graph-based problem ([details](https://github.com/NickDrake117/GNN_tf_2.x/blob/main/MUTAG_raw/Mutagenicity_label_readme.txt))

### Single model training and testing
GNN levels can be trained in parallel or in series by using `serial_training` argument when calling `model.train()`. Default is `False`.

    from starter_lgnn import lgnn, gTr, gVa, gTe
    
    epochs = 200
    
    # parallel training
    lgnn.train(gTr, epochs, gVa)
    
    # serial training: each GNN layer is trained separately, one by one
    # lgnn.train(gTr, epochs, gVa, serial_training=True)  
    
    # test the lgnn
    res = lgnn.test(gTe)

    # print test result
    for i in res: 
        print('{}: \t{:.4g}'.format(i, res[i]))


### K-fold Cross Validation
To perform a 10-fold cross validation in parallel mode, simply run:

    from starter_lgnn import lgnn, graphs
    from numpy import mean
    
    epochs = 200
    
    # LKO: as mentioned, arg serial_training affects LGNN training process
    lko_res = lgnn.LKO(graphs, 10, epochs=epochs, serial_training=False)
    
    # print test result
    for i in lko_res: 
        for i in m: print('{}: \t{:.4f} \t{}'.format(i, mean(lko_res[i]), lko_res[i]))


### TensorBoard
To visualize learning progress, use TensorBoard --logdir command providing the log directory. Default it's `writer`.

    ...\projectfolder> tensorboard --logdir writer
    

## Citing
To cite the LGNN implementation please use the following publication:

    Pancino, N., Rossi, A., Ciano, G., Giacomini, G., Bonechi, S., Andreini, P., Scarselli, F., Bianchini, M., Bongini, P. (2020),
    "Graph Neural Networks for the Prediction of Protein–Protein Interfaces",
    In ESANN 2020 proceedings (pp.127-132).
    
Bibtex:

    @inproceedings{Pancino2020PPI,
      title={Graph Neural Networks for the Prediction of Protein–Protein Interfaces},
      author={Niccolò Pancino, Alberto Rossi, Giorgio Ciano, Giorgia Giacomini, Simone Bonechi, Paolo Andreini, Franco Scarselli, Monica Bianchini, Pietro Bongini},
      booktitle={28th European Symposium on Artificial Neural Networks, Computational Intelligence and Machine Learning (online event)},
      pages={127-132},
      year={2020}
    }

---------

To cite LGNN please use the following publication:

    N. Bandinelli, M. Bianchini and F. Scarselli, 
    "Learning long-term dependencies using layered graph neural networks", 
    The 2010 International Joint Conference on Neural Networks (IJCNN), 
    Barcelona, 2010, pp. 1-8, doi: 10.1109/IJCNN.2010.5596634.
    
Bibtex:

    @inproceedings{Scarselli2010LGNN,
      title={Learning long-term dependencies using layered graph neural networks}, 
      author={Niccolò Bandinelli, Monica Bianchini, Franco Scarselli},
      booktitle={The 2010 International Joint Conference on Neural Networks (IJCNN)}, 
      year={2010},
      volume={},
      pages={1-8},
      doi={10.1109/IJCNN.2010.5596634}
    }

## Contributions
LGNN implementation is based on the TF2.x GNN model implementation [repo](https://github.com/NickDrake117/GNN_tf_2.x)

## License
Released under the 3-Clause BSD license (see `LICENSE.txt`):

    Copyright (C) 2004-2020 Niccolò Pancino
    Niccolò Pancino <niccolo.pancino@unifi.it>
    Pietro Bongini <pietro.bongini@gmail.com >
    Matteo Tiezzi <mtiezzi@diism.unisi.it>
