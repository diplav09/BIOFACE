import numpy as np
import cv2,os,math,sys
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
import torch
import torch.nn as nn
import imageio
from scipy import misc
import h5py

to_image = transforms.Compose([transforms.ToPILImage()])
dataset_dir = '/content/gdrive/My Drive/bioface/sample.hdf5'
data_path = '/content/gdrive/My Drive/bioface/'
batch_size = 1
weight_path = '/content/gdrive/My Drive/bioface/models/cnn_epoch_199.pth'

def test_model():

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
	illumA = np.expand_dims(illumA, axis=3)
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

	illumA = torch.from_numpy(illumA).cuda()
	illumDNorm = torch.from_numpy(illumDNorm).cuda()
	illumFNorm = torch.from_numpy(illumFNorm).cuda()
	mu = torch.from_numpy(mu).cuda()
	PC = torch.from_numpy(PC).cuda()
	Newskincolour = torch.from_numpy(Newskincolour).cuda()
	Tmatrix = torch.from_numpy(Tmatrix).cuda()
	print("pre proc done")

	torch.backends.cudnn.deterministic = True
	device = torch.device("cuda")

	test_dataset = LoadData(dataset_dir, test=False)
	test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False, num_workers=1,
						  pin_memory=True, drop_last=True)
	model = MultipleDecoder() 
	model = torch.nn.DataParallel(model) # multi GPU
	model.load_state_dict(torch.load(weight_path), strict=True)
	model.eval()
	with torch.no_grad():

		test_iter = iter(test_loader)
		for j in range(0,2):
			print("Processing image " + str(j))
			torch.cuda.empty_cache()
			images, actualshading, actualmasks = next(test_iter)
			images = images.to(device, non_blocking=True).float()
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
			sRGBim = np.asarray(to_image(torch.squeeze(sRGBim.float().detach().cpu())))
			Shading = np.asarray(to_image(torch.squeeze(Shading.float().detach().cpu())))
			Specularities = np.asarray(to_image(torch.squeeze(Specularities.float().detach().cpu())))
			fmel = np.asarray(to_image(torch.squeeze(fmel.float().detach().cpu())))
			fblood = np.asarray(to_image(torch.squeeze(fblood.float().detach().cpu())))
			images = np.asarray(to_image(torch.squeeze(images.float().detach().cpu())))

			imageio.imwrite("results/" + str(j) + "_recons_.png", sRGBim)
			imageio.imwrite("results/" + str(j) + "_diffuse_.png", Shading)
			imageio.imwrite("results/" + str(j) + "_specular_.png", Specularities)
			imageio.imwrite("results/" + str(j) + "_melanin_.png", fmel)
			imageio.imwrite("results/" + str(j) + "_haemoglobin_.png", fblood)
			imageio.imwrite("results/" + str(j) + "_input_.png", images)

 if __name__ == '__main__':
	test_model()
