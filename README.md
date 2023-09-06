# AttentionAE-sc
Attention-based method for clustering scRNA-seq data

## run environment

python --- 3.8

pytorch --- 1.11.0

torchvision --- 1.12.0

torchaudio --- 0.11.0

scanpy --- 1.8.2

scipy --- 1.6.2

numpy --- 1.19.5

leidenalg --- 0.8.10


## Data Source
The preprocessing of two kinds of [datasets (.h5, .csv)](https://github.com/LiShenghao813/AttentionAE-sc/tree/main/Data) is provided by the ["preprocessing_h5.py"](https://github.com/LiShenghao813/AttentionAE-sc/blob/main/preprocessing_h5.py) and ["preprocessing_baron.py"](https://github.com/LiShenghao813/AttentionAE-sc/blob/main/preprocessing_baron.py). Then, the corresponding ".h5ad" files are output in the ["./Data/AnnData"](https://github.com/LiShenghao813/AttentionAE-sc/tree/main/Data/AnnData), where is the default for model input. If you want to analzye another scRNA-seq datasets, please copy your ".h5ad" files to here and set "-name" to the file name.


Other datasets: the breast cancer single-cell dataset used in our research can obtain from ["GSE173634"](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE173634).


## Usage
For applying AttentionAE-sc, the convenient way is  run ["run_AttentionAE-sc.py"](https://github.com/LiShenghao813/AttentionAE-sc/blob/main/run_AttentionAE-sc.py).

Please place the scRNA-seq dataset you want to analyze in the directory ["./Data/AnnData"](https://github.com/LiShenghao813/AttentionAE-sc/tree/main/Data/AnnData), where is the default for model input.
If you want to calculate the similarity between the predicted clustering resluts and the true cell labels (based on NMI or ARI score), please transmit your true labels into the "adata.obs['celltype']" and setting the argument "-celltype" to **True**.

argument:

    "-resolution": default: 1.0. Description: The resolution of Leiden. Advised settings in 0.1 to 1.0. 
    
    "-n_heads": default: 8. Description: The number of attention heads. Advised settings in 4 to 8. 

    "-n_hvg": default: 2500. Description: The number of highly variable genes. In general values should be in the range 500 to 3000. 

    "-connectivity_methods": default: 'gauss'. Description: Method for constructing the cell connectivity ("gauss" or "umap"). 

    "-knn": default: False. Description: If **True**, use a hard threshold to restrict the number of neighbors to n_neighbors, that is, 
                                        consider a knn graph. Otherwise, use a Gaussian Kernel to assign low weights to neighbors more 
                                        distant than the n_neighbors nearest neighbor.
    
    "-n_neighbors": default: 15. Description: The size of local neighborhood (in terms of number of neighboring data points) used 
                                    for manifold approximation. Larger values result in more global views of the manifold, while 
                                    smaller values result in more local data being preserved. In general values should be in the 
                                    range 2 to 100. default:15
    
other arguments:

    "-celltype": default: "known". Description: The true labels of datasets are placed in adata.obs["celltype"] for model evaluation.
    
    "-save_pred_label":default: False. Description: To choose whether saves the pred_label to the dict "./pred_label"
    
    "-save_model_para":default: False. Description: To choose whether saves the model parameters to the dict "./model_save"
    
    "-save_embedding":default: True. Description: To choose whether saves the cell embedding to the dict "./embedding"
    
    "-max_num_cell":  default: 4000. Description: Conduct random sampling training on large datasets. 4,000 is the maximum cells that a GPU (8RAM) can handle. In the experiment, AttentionAE-sc still performs well when 1/10 cells is sampled for model training. 
