# LGNN - Layered Graph Neural Network Model - TF 2.x
This repo contains a Tensorflow 2.x implementation of the Layered Graph Neural Network Model.

- **Authors:** [Niccolò Pancino](http://sailab.diism.unisi.it/people/niccolo-pancino/), [Pietro Bongini](http://sailab.diism.unisi.it/people/pietro-bongini/)

## Install
### Requirements
The LGNN framework requires the packages **tensorflow**, **numpy**, **pandas**, **scikit-learn**, **matplotlib**.

To install the requirements you can use the following command:

    pip install -U -r requirements.txt


## Simple usage example
By default, LGNN is a 5-layered GNN for binary node-focused classification task on graphs with random nodes/edges/targets.

Open the script `starter_lgnn` and set parameters in section *SCRIPT OPTIONS* to change dataset and LGNN architecture and behaviour, then run one of the following scripts. 

In particular, set `use_MUTAG = True` to get the real-world dataset MUTAG for solving a graph-based problem ([details](https://github.com/NickDrake117/GNN_tf_2.x/blob/main/MUTAG_raw/Mutagenicity_label_readme.txt))

### Single model training and testing
In the following script, both LGNN training and LGNN testing are performed. 

    from starter_lgnn import lgnn, gTr, gVa, gTe
    
    epochs = 200
    
    # training
    lgnn.train(gTr, epochs, gVa)
    
    # test
    res = lgnn.test(gTe)

    # print test result
    for i in metrics: 
        print('{}: \t{:.4g}'.format(i, res[i]))

### K-fold Cross Validation
To perform a 10-fold cross validation, simply run:

    from starter_lgnn import lgnn, graphs
    
    epochs = 200
    
    lko_res = lgnn.LKO(graphs, 10, epochs=epochs)
    
To visualize learning progress, use TensorBoard --logdir providing the log directory. Default it's `writer`.

    ...\projectfolder> tensorboard --logdir writer
    




## Citing
To cite the GNN implementation please use the following publication:

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

To cite GNN please use the following publication:

    N. Bandinelli, M. Bianchini and F. Scarselli, 
    "Learning long-term dependencies using layered graph neural networks", 
    The 2010 International Joint Conference on Neural Networks (IJCNN), 
    Barcelona, 2010, pp. 1-8, doi: 10.1109/IJCNN.2010.5596634.
    
Bibtex:

    @inproceedings{Scarselli2010LGNN,
      title={Learning long-term dependencies using layered graph neural networks}, 
      author={N. {Bandinelli} and M. {Bianchini} and F. {Scarselli}},
      booktitle={The 2010 International Joint Conference on Neural Networks (IJCNN)}, 
      year={2010},
      volume={},
      pages={1-8},
      doi={10.1109/IJCNN.2010.5596634}
    }

## Contributions
LGNN implementation is based on the TF2.x GNN implementation [repo](https://github.com/NickDrake117/GNN_tf_2.x)

## License
Released under the 3-Clause BSD license (see `LICENSE.txt`):

    Copyright (C) 2004-2020 Niccolò Pancino
    Niccolò Pancino <niccolo.pancino@unifi.it>
    Pietro Bongini <pietro.bongini@gmail.com >
    Matteo Tiezzi <mtiezzi@diism.unisi.it>
