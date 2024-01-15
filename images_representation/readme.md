


1. 

enter to images_representation
```
cd images_representation 
```

create conda environment:
```
conda env create -f env.yml
```

activate conda environment:
```
conda activate image_representation 
```

create experiment directory and config:
```
cd  *to your location*
mkdir *experimet name* 
create experiment config yaml in this directory
see example ender 
images_representation/experiment_configs/experiment1.yaml 
```

Run train:
```
python -m image_representation.main  --exp_dir **path to experiment dir**

Train will create weight dir under experiment dir with model weights after every epoc.
```

Run plot TSNE 

```
python -m image_representation.plot_tsne  --exp_dir **path to experiment dir** --weights_file_name *path to model weights file*
```

Run upsampling:

```
python -m image_representation.upsampling  --exp_dir **path to experiment dir**  --weights_file_name  *path to model weights file* --upsample_size **upsampling size for example 256**

```

Run interpolation
```
python -m image_representation.interpolation --exp_dir **path to experiment dir**  --weights_file_name  *path to model weights file* --pairs_csv *csv of pairs to interplolate 1,2,3,4 for example will create pairs [1,2] and [3,4]*

```
