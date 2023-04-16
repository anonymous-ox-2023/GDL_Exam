# Geometric Deep Learning 

This is a supporting code repository for the GDL mini-project. 
The base of the code is the PointMLP [repository](https://github.com/ma-xu/pointMLP-pytorch), we only added a few modifications and deleted irrelevant code sections.

To setup the environment please follow the instructions:

## Install

```bash
git clone https://github.com/anonymous-ox-2023/GDL_Exam.git

conda env create
conda activate pointmlp
```

## Navigate to the main directory
```bash
cd classification_ModelNet40
```

## Setup the code
There are 3 new files added to the original PointMLP pipeline, all of them begin with prefix *my-* to indicate my changes:
*my_utils.py*, *my_train.py* and *my_test.py*.


We also changed the embedding step in the PointMLP model located in *classification_ModelNet40/models/pointmlp.py*. The main logic for the changed embedding procedure is in the *CustomEmbedding* class.
It has 4 different options to proceed, you have to specify the mode in the *my_train.py* file.

```bash
  # =============== Parameters to change ======================================================
  batch_size = 8
  workers = 4
  
  path = Path(r'Path to the raw folder of ModelNet40 dataset')
  
  # Select training mode: original, gat, edge, dynamic
  mode = "gat"
  # ===========================================================================================
```

In the code above (the code section from *my_train.py* file) you should specify the batch size and number of workers according to your computational resources.
Also, please specify the path to the folder with raw .off files with corresponding 3D models.
ModelNet40 can be downloaded from [here](https://www.kaggle.com/datasets/balraj98/modelnet40-princeton-3d-object-dataset).
Finally, select the mode: **original** will run the default PointMLP setup, **gat** will use attention embeddings, **edge** for *EdgeConv* and **dynamic** for *DynamicEdgeConv*.

Now you are ready to train the model:

## Run the training procedure
```bash
python my_train.py
```
