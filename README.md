# BioFaceNet Implementation in Pytorch

## File contents

1. main.py contains train code 
2. test_model.py contains test code
3. load_data.py contains dataloader
4. utils.py contain supported function for training and inference 
5. models/ contain the best epoch weight file
6. data/ the validation data

## For Training

1. set in main.py --  dataset_dir = '/content/gdrive/My Drive/bioface/zx_7_d10_inmc_celebA_01.hdf5' (path of hdf5 file)
                      data_path = '/content/gdrive/My Drive/bioface/' (where hdf5 file is kept) 
2. python3 main.py
3. corresponding Jupyter notebook is train_bioface.ipynb

## For Inference

1. set in test_model.py   
						dataset_dir = '/content/gdrive/My Drive/bioface/sample.hdf5' (path of hdf5 file)
						data_path = '/content/gdrive/My Drive/bioface/'    (where hdf5 file is kept)
						data can be downloaded from here https://drive.google.com/file/d/1-2e0_Ke72c-Nq1lT7zYq9THcI_fWGFTe/view?usp=sharing
						weight_path = '/content/gdrive/My Drive/bioface/models/cnn_epoch_62.pth'  (weight file path) 
						weight file can be downloaded from here (https://drive.google.com/file/d/12xW3HbVJ04bhZylXOEJqU8icOla3pXox/view?usp=sharing) 
1. python3 test_model.py
2. corresponding Jupyter notebook is here infer_bioface.ipynb


