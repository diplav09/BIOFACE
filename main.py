import numpy as np
import cv2,os,math,sys
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
import torch
import torch.nn as nn
import imageio
from load_data import LoadData
from utils import *
from basic_blocks import MultipleDecoder
import h5py


np.random.seed(0)
torch.manual_seed(0)

dataset_dir = '/content/gdrive/My Drive/bioface/zx_7_d10_inmc_celebA_01.hdf5'
data_path = '/content/gdrive/My Drive/bioface/'
batch_size = 64
learning_rate = 1e-5
num_train_epochs = 200
blossweight = 1e-4  
appweight = 1e-3
Shadingweight = 1e-5 
sparseweight = 1e-5
load_checkpoint_path = None 

def loss_L2_regularization(output):
	loss = torch.sum(torch.square(output))
	return loss
def loss_shade(predictedShading, actualshading, actualmasks):
	scale = torch.sum( torch.sum( (actualshading * predictedShading) * actualmasks, 2), 2) / torch.sum( torch.sum( torch.square(predictedShading) * actualmasks, 2), 2)

	scale = torch.reshape(scale,(nbatchs,1,1,1))
	predictedShading = predictedShading * scale
	alpha = (actualshading - predictedShading) * actualmasks
	loss = torch.sum(torch.square(alpha))
	return loss
def loss_L1_regularization(output):
	loss = torch.sum(output)
	return loss

def train_model():
	# setup.m executed
	illF, illumDmeasured, illumA, Newskincolour, rgbCMF, Tmatrix, XYZspace = load_matfiles(data_path)
	Newskincolour = np.transpose(Newskincolour, (2, 0, 1))  # 256x 256 X 33 -> 33 X 256 X 256
	Newskincolour = np.tile(Newskincolour,(batch_size,1,1,1))
	mu,PC,EV = CameraSensitivityPCA(rgbCMF)
	LightVectorSize = 15
	wavelength = torch.tensor(33.)
	bSize = torch.tensor(2)
	illF = illF.reshape((1,1,33,12))
	illumDmeasured = illumDmeasured.T.reshape((1,1,33,22))
	illumA = illumA.astype(np.float32) / np.sum(illumA)             # 1,1,33
	illumA = np.tile(illumA,(1,1,1,batch_size))  # additional line
	illumDNorm = illumDmeasured.astype(np.float32)
	for i in range(0,22):
		illumDNorm[:,:,:,i] = illumDmeasured[:,:,:,i] / np.sum(illumDmeasured[:,:,:,i])

	illumFNorm = illF.astype(np.float32)
	for i in range(0,12):
		illumFNorm[:,:,:,i] = illF[:,:,:,i] / np.sum(illF[:,:,:,i])

	celebaimdb_averageImage = torch.tensor([129.1863,104.7624,93.5940])
	muim = torch.reshape(celebaimdb_averageImage,(1,3,1,1))
	bSize = 2

	illumA = torch.from_numpy(illumA)
	illumDNorm = torch.from_numpy(illumDNorm)
	illumFNorm = torch.from_numpy(illumFNorm)
	mu = torch.from_numpy(mu)
	PC = torch.from_numpy(PC)
	Newskincolour = torch.from_numpy(Newskincolour)
	Tmatrix = torch.from_numpy(Tmatrix)
	print("pre proc done")

	torch.backends.cudnn.deterministic = True
	device = torch.device("cuda")

	train_dataset = LoadData(dataset_dir, test=False)
	train_loader = DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True, num_workers=1,
						  pin_memory=True, drop_last=True)
	model = MultipleDecoder() 
	model = torch.nn.DataParallel(model) # multi GPU
	optimizer = Adam(params=model.parameters(), lr=learning_rate)

	if load_checkpoint_path is not None:
		checkpoint = torch.load(load_checkpoint_path)
		model.load_state_dict(checkpoint, strict=True)
	# Loss define
	MSE_loss = torch.nn.MSELoss()
	print("train start")
	for epoch in range(num_train_epochs):

		torch.cuda.empty_cache()

		train_iter = iter(train_loader)
		train_loss = 0
		for i in range(len(train_loader)):

			optimizer.zero_grad()
			images, actualshading, actualmasks = next(train_iter)
			images = images.to(device, non_blocking=True)
			actualshading = actualshading.to(device, non_blocking=True)
			actualmasks = actualmasks.to(device, non_blocking=True)
			lightingparameters,b,fmel,fblood,Shading,specmask = model(images)

			weightA,weightD,CCT,Fweights,b,BGrid,fmel,fblood,Shading,specmask = ScaleNet(lightingparameters,b,fmel,fblood,Shading,specmask,bSize)
			e = illuminationModel(weightA,weightD,Fweights,CCT,illumA,illumDNorm,illumFNorm)
			Sr,Sg,Sb = cameraModel(mu.float(),PC.float(),b,wavelength)
			lightcolour = computelightcolour(e,Sr.float(),Sg.float(),Sb.float())
			Specularities = computeSpecularities(specmask,lightcolour)
			R_total = BiotoSpectralRef(fmel,fblood,Newskincolour)
			rawAppearance,diffuseAlbedo = ImageFormation (R_total, Sr,Sg,Sb,e,Specularities,Shading)
			ImwhiteBalanced = WhiteBalance(rawAppearance,lightcolour)
			T_RAW2XYZ = findT(Tmatrix,BGrid)
			sRGBim = fromRawTosRGB(ImwhiteBalanced,T_RAW2XYZ)

			# Camera parameter loss:
			priorloss = loss_L2_regularization(b)
			# L2: appearance loss
			appearanceloss = MSE_loss(sRGBim * actualmasks, images * actualmasks)

			shadingloss = loss_shade(Shading, actualshading, actualmasks)
			# L1 sparsity loss
			sparsityloss = loss_L1_regularization(Specularities)

			total_loss = blossweight * priorloss + appweight * appearanceloss + Shadingweight * shadingloss + sparseweight * sparsityloss
			train_loss += total_loss.item()
			total_loss.backward()
			optimizer.step()
		train_loss = train_loss / len(train_loader)
		print("train_loss == ",train_loss , "  epoch == ",epoch)
		model.eval().cpu()
		torch.save(model.state_dict(), "models/cnn_epoch_" + str(epoch) + ".pth")
		model.to(device).train()


 if __name__ == '__main__':
	train_model()