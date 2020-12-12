# GNN and LGNN - Graph Neural Network Models
This repo contains a Tensorflow 2.x implementations of the Graph Neural Network (GNN) and Layered Graph Neural Network (LGNN) Models.

**Authors**
- **GNN:** [Niccolò Pancino](http://sailab.diism.unisi.it/people/niccolo-pancino/), [Pietro Bongini](http://sailab.diism.unisi.it/people/pietro-bongini/)
- **LGNN:** [Niccolò Pancino](http://sailab.diism.unisi.it/people/niccolo-pancino/)


## Install
### Requirements
The GNN framework requires the packages **tensorflow**, **numpy**, **pandas**, **scikit-learn**, **matplotlib**.

To install the requirements you can use the following command:

    pip install -U -r requirements.txt


## Simple usage example
In the following scripts, gnn is a GNN trained by default to solve a binary node-focused classification task on graphs with random nodes/edges/targets, while lgnn is a 5-layered GNN.

Open the script `starter_gnn` and set parameters in section *SCRIPT OPTIONS* to change dataset and/or GNN/LGNN models architectures and learning behaviour.

In particular, set `use_MUTAG=True` to get the real-world dataset MUTAG for solving a graph-based problem ([details](https://github.com/NickDrake117/GNN_tf_2.x/blob/main/MUTAG_raw/Mutagenicity_label_readme.txt))

Note that a single layered LGNN behaves exaclty like a GNN, as it is composed of a single GNN.

### Single model training and testing
#### GNN

    from starter import gnn, gTr, gTe, gVa
    
    epochs = 200
    
    # training
    gnn.train(gTr, epochs, gVa)
    
    # test the gnn
    res = gnn.test(gTe)

    # print test result
    for i in res:  
        print('{}: \t{:.4g}'.format(i, res[i]))


#### LGNN
LGNN can be trained both in parallel or serial mode, by setting `serial_training` argument when calling `LGNN.train()`. Default is `False`.

In Parallel Mode, GNN layers are trained simultaneously, by processing loss on the LGNN output (i.e. the final GNN layer output), and backpropagating the error throughout the GNN layers.

In Serial Mode, each GNN layer is trained as a standalone GNN model, therefore becoming an *expert* which solves the considered problem using the original data and the experience obtained from the previous GNN layer, so as to "correct" the errors made by the previous network, rather than solving the whole problem.

the former the GNNs layers are trained simultaneously, in the second case each GNN layer is a standalone GNN which is trained separately, one by one, using the original data and the experience obtained from the gnn of the previous layer (in the form of vectors of outputs, states or both)

 
    from starter import lgnn, gTr, gTe, gVa
    
    epochs = 200
    
    # training in parallel mode
    gnn.train(gTr, epochs, gVa)
    
    # training in serial mode
    # gnn.train(gTr, epochs, gVa, serial_training=True)
    # test the lgnn
    res = gnn.test(gTe)

    # print test result
    for i in res:  
        print('{}: \t{:.4g}'.format(i, res[i]))

### K-fold Cross Validation
To perform a 10-fold cross validation on gnn, simply run:

    from starter_gnn import gnn, graphs
    from numpy import mean
    
    epochs = 200
    
    # LKO: as mentioned, arg serial_training affects LGNN training process
    lko_res = gnn.LKO(graphs, 10, epochs=epochs, serial_training=False)
    
    # print test result
    for i in lko_res: 
        for i in m: print('{}: \t{:.4f} \t{}'.format(i, mean(lko_res[i]), lko_res[i]))





### TensorBoard
To visualize learning progress, use TensorBoard --logdir command providing the log directory. Default it's `writer`.

    ...\projectfolder> tensorboard --logdir writer
   
   
### GNN implementation flow chart
The following image details the GNN model as it is implemented in `GNN / GNN.py`.
![GNN Convergence Loop](GNN/GNN_flow_chart.png)





## Citing
To cite the GNN/LGNN implementations please use the following publication:

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

    F. Scarselli, M. Gori,  A. C. Tsoi, M. Hagenbuchner, G. Monfardini, 
    "The Graph Neural Network Model", IEEE Transactions on Neural Networks,
    vol. 20(1); p. 61-80, 2009.
    
Bibtex:

    @article{Scarselli2009TheGN,
      title={The Graph Neural Network Model},
      author={Franco Scarselli, Marco Gori, Ah Chung Tsoi, Markus Hagenbuchner, Gabriele Monfardini},
      journal={IEEE Transactions on Neural Networks},
      year={2009},
      volume={20},
      pages={61-80}
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
Part of the code was inspired by [M. Tiezzi](http://sailab.diism.unisi.it/people/matteo-tiezzi/) and [A. Rossi](http://sailab.diism.unisi.it/people/alberto-rossi/) GNN implementation in TF 1.x ([repo](https://github.com/sailab-code/gnn)).

## License
Released under the 3-Clause BSD license (see `LICENSE.txt`):

    Copyright (C) 2004-2020 Niccolò Pancino
    Niccolò Pancino <niccolo.pancino@unifi.it>
    Pietro Bongini <pietro.bongini@gmail.com >
    Matteo Tiezzi <mtiezzi@diism.unisi.it>


## Simple usage example


Open the script `starter_lgnn` and set parameters in section *SCRIPT OPTIONS* to change dataset and/or LGNN architecture and behaviour.

In particular, set `use_MUTAG=True` to get the real-world dataset MUTAG for solving a graph-based problem ([details](https://github.com/NickDrake117/GNN_tf_2.x/blob/main/MUTAG_raw/Mutagenicity_label_readme.txt))



## License
Released under the 3-Clause BSD license (see `LICENSE.txt`):

    Copyright (C) 2004-2020 Niccolò Pancino
    Niccolò Pancino <niccolo.pancino@unifi.it>
    Pietro Bongini <pietro.bongini@gmail.com >
    Matteo Tiezzi <mtiezzi@diism.unisi.it>
