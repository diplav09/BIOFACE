import numpy as np
import cv2,os,math,sys
from torch.utils.data import DataLoader
from torchvision import transforms
from torch.optim import Adam
import torch
import torch.nn as nn
from sklearn.decomposition import PCA

def load_matfiles(data_path):
	illF = loadmat(data_path + 'illF.mat')
	illF = illF['illF']  # 1x 33 x 12
	
	illumDmeasured = loadmat(data_path + 'illumDmeasured.mat')
	illumDmeasured = illumDmeasured['illumDmeasured']	

	illumA = loadmat(data_path + 'illumA.mat')
	illumA = illumA['illumA']

	Newskincolour = loadmat(data_path + 'Newskincolour.mat')
	Newskincolour = Newskincolour['Newskincolour']

	rgbCMF = loadmat(data_path + 'rgbCMF.mat')
	rgbCMF = rgbCMF['rgbCMF']

	Tmatrix = loadmat(data_path + 'Tmatrix.mat')
	Tmatrix = Tmatrix['Tmatrix']

	XYZspace = loadmat(data_path + 'XYZspace.mat')
	XYZspace = XYZspace['XYZspace']

	return illF, illumDmeasured, illumA, Newskincolour, rgbCMF, Tmatrix, XYZspace
# to scale the parameter 
def ScaleNet(lightingparameters,b,fmel,fblood,Shading,specmask,bSize):
#     weightA  : B x 1 x 1 x 1  
#     weightD  : B x 1 x 1 x 1  
#     CCT      : B x 1 x 1 x 1  
#     Fweights : B x 12 x 1 x 1  
#     b        : B x 2 x 1 x 1  
#     fmel     : B x 1 x 224 x 224  
#     fblood   : B x 1 x 224 x 224  
#     Shading  : B x 1 x 224 x 224  
#     specmask : B x 1 x 224 x 224  
#     bSize    : 2
  nbatch = lightingparameters.size()[0]
  m = nn.Softmax(dim=1)
  lightingweights = m(lightingparameters[:,0:14,:,:])
  weightA  = lightingweights[:,0,:,:]
  weightA = torch.unsqueeze(weightA,1)
  weightD  = lightingweights[:,1,:,:]
  weightD = torch.unsqueeze(weightD,1)
  Fweights = lightingweights[:,2:14,:,:]
  CCT      =  lightingparameters[:,14,:,:]
  CCT      = ((22 - 1) / (1 + torch.exp(-CCT))) + 1;
  CCT = torch.unsqueeze(CCT,1)
  b = 6.*(torch.sigmoid(b))-3
  BGrid = torch.reshape(b,(bSize,1,1,nbatch)) # 2 x 1 x 1 x B check this reshape
  BGrid = BGrid / 3
  fmel = torch.sigmoid(fmel) *2 -1
  fblood = torch.sigmoid(fblood) * 2 -1
  Shading = torch.exp(Shading)
  specmask = torch.exp(specmask)
  return weightA,weightD,CCT,Fweights,b,BGrid,fmel,fblood,Shading,specmask

# create the illumination model from CIE standard illuminants: A,D,F
# extract illumA,illumDNorm,illumFNorm foom mat files in util
def illuminationModel(weightA,weightD,Fweights,CCT,illumA,illumDNorm,illumFNorm):
#     weightA    : B x 1 x 1 x 1 
#     weightD    : B x 1 x 1 x 1
#     CCT        : B x 1 x 1 x 1
#     Ftype      : B x 12 x 1 x 1 
#     illumA     : 1 x 1 x 33 x B
#     illumDNorm : 1 x 1 x 33 x 22
#     illumFNorm : 1 x 1 x 33 x 12
  illumA = illumA.permute(3,2,0,1)  # B x 33 x 1 x 1
  print(illumA.size())
  illuminantA = illumA * weightA # B x 33 x 1 x 1

  # don't know this layer. check where function vl_nnillumD is defined 
  # illumDlayer = Layer.fromFunction(@vl_nnillumD);
  # illD   = illumDlayer(CCT,illumDNorm);
  # illuminantD = illD.*weightD;

  # illuminantD should be converted to B x 33 x 1 x 1
  illumDNorm = illumDNorm.permute(3,2,0,1) # 22 x 33 x 1 x 1
  illumDNorm = torch.unsqueeze(torch.sum(illumDNorm,0),0) # 1 x 33 x 1 x 1
  illuminantD = CCT*illumDNorm*weightD  

  illumFNorm = illumFNorm.permute(0,2,3,1) #permute to 1 x 33 x 12 x 1
  Fweights = Fweights.permute(2,3,1,0)
  illuminantF = illumFNorm*Fweights # 1 x 33 x 12 x B
  illuminantF = torch.unsqueeze(torch.sum(illuminantF,2),2) # check if dimension is reduced 1 x 33 x 1 x B
  illuminantF = illuminantF.permute(3,1,0,2) # B x 33 x 1 x 1
  e = illuminantA + illuminantD +illuminantF
  esum = torch.unsqueeze(torch.sum(e,1),1) # sum across channel
  e = e / esum

  return e

def cameraModel(mu,PC,b,wavelength):
# Inputs:
#     mu         : B x 1 x 1 x 1 
#     PC         : B x 1 x 1 x 1 ?? actually is 99 x 2
#     b          : B x 2 x 1 x 1 
#     wavelength : 33

# Outputs:
#     Sr,Sg,Sb   : B x 33 x 1 x 1

    nbatch = b.size()[0]
    ## PCA model
    b = torch.reshape(b,(nbatch,2)).float()
    PC = torch.reshape(PC,(99,2)).float()
    # print(b.size(),PC.size())
    S = torch.matmul(PC,torch.transpose(b, 0, 1))  # 99 x nbatch
    mu = torch.unsqueeze(mu,1)
    S = S + mu  
    rel = nn.ReLU()
    S =  rel(S)
    S = torch.transpose(S, 0, 1)
    # print(S.size())
    wavelength = wavelength.int()    
    ## split up S into Sr, Sg, Sb 
    Sr = torch.reshape(S[:,0:wavelength],(nbatch, wavelength, 1, 1))                  
    Sg = torch.reshape(S[:,wavelength:wavelength*2],(nbatch, wavelength, 1, 1))     
    Sb = torch.reshape(S[:,wavelength*2:wavelength*3],(nbatch, wavelength, 1, 1))

    return Sr,Sg,Sb 

def CameraSensitivityPCA(rgbCMF):

  X = np.zeros((99,28))
  Y = np.zeros((99,28))
  redS = rgbCMF[0,0]
  greenS= rgbCMF[0,1]
  blueS = rgbCMF[0,2]
  for i in range(0,28):
    Y[0:33,i]  = redS[:,i] / np.sum(redS[:,i])
    Y[33:66,i] = greenS[:,i] / np.sum(greenS[:,i])
    Y[66:99,i] = blueS[:,i] / np.sum(blueS[:,i])
  pca = PCA(n_components=28)
  pca.fit(Y.T)
  PC = pca.components_    # 99,27
  EV = pca.explained_variance_  # 27,1
  mu = pca.mean_  # 1,99
  PC = np.matmul(PC.T[:,0:2],np.diag(np.sqrt(EV[0:2])))
  EV = EV[0:2]
  # [PC,~,EV,~,explained,mu] = pca(Y')
  # PC = single(PC(:,1:2)*diag(sqrt(EV(1:2)))) # 99,2  
  # mu = single(mu')  # 99,1
  # EV = single(EV(1:2)) # 2,1
  return mu,PC,EV

def computelightcolour(e,Sr,Sg,Sb):
# Inputs:
#     Sr,Sg,Sb         : B x 33 x 1 x 1 
#     e                : B x 33 x 1 x 1
#  Output:
#  lightcolour         : Bx 3 x 1 x 1 
  lightcolour  = torch.cat((torch.sum(Sr * e,1), torch.sum(Sg * e,1), torch.sum(Sb * e,1)),1)
  lightcolour = torch.unsqueeze(lightcolour,2)
  return lightcolour

def computeSpecularities(specmask,lightcolour):
# Inputs:
#     specmask          : B x 1 x H x W 
#     lightcolour      : B x 3 x 1 x 1 
#  Output:
#     Specularities    : B x 1 x H x W 
##
	Specularities = specmask * lightcolour
	return  Specularities 

def BiotoSpectralRef(fmel,fblood,Newskincolour):
# Inputs:
#     fmel             : B x 1 x H x W 
#     fblood           : B x 1 x H x W
#     Newskincolour    : B x 33 x 256 x 256 
#  Output:
#     R_total          : B x 33 x H x W 
##
	BiophysicalMaps = torch.cat((fblood,fmel),1) # B x 2 x H x W 
	BiophysicalMaps = BiophysicalMaps.permute(0, 2, 3, 1) # for troch grid shape should be B x H x W x 2  


	R_total  = nn.functional.grid_sample(Newskincolour, BiophysicalMaps, mode='bilinear')
	return R_total 

def ImageFormation (R_total, Sr,Sg,Sb,e,Specularities,Shading):
	#Inputs:,
	#     R_total       : nbatch X 33 X H X W 
	#     Shading       : nbatch X 1 X H X W 
	#     Specularities : nbatch X 1 X H X W 
	#     Sr,Sg,Sb      : nbatch x 33 x 1 x 1 
	#     e             : nbatch x 33 x 1 x 1 
	# Output:
	#     rgbim : nbatch x 1 x H x  W 
	#---------------------------Image Formation -------------------------------
	spectraRef = R_total * e # nbatch X 33 X H X W  
	#--------------------------------------------------------------------------
	rChannel = torch.unsqueeze(torch.sum(spectraRef * Sr,1),1)  
	gChannel = torch.unsqueeze(torch.sum(spectraRef * Sg,1),1)  
	bChannel = torch.unsqueeze(torch.sum(spectraRef * Sb,1),1)  

	diffuseAlbedo = torch.cat((rChannel,gChannel,bChannel),1)  # nbatch x 3 x H x W 

	#---------------------------Shaded Diffuse --------------------------------

	ShadedDiffuse = diffuseAlbedo * Shading  # nbatch x 3 x H x W

	# ShadedDiffuse = torch.unsqueeze(torch.sum(ShadedDiffuse,1),1) #added for dimension correction
	#---------------------------Raw appearance --------------------------------
	rawAppearance = ShadedDiffuse + Specularities 
	return rawAppearance,diffuseAlbedo  

def WhiteBalance(rawAppearance,lightcolour):
# Inputs:
#     rawAppearance    : B x 3 x H x W 
#     lightcolour      : B x 3 x 1 x 1  
#  Output:
#     ImwhiteBalanced  : B x 3 x H x W 
## --------------------------- White Balance ------------------------------
	WBrCh = torch.unsqueeze(rawAppearance[:,0,:,:]/lightcolour[:,0,:,:],1)  
	WBgCh = torch.unsqueeze(rawAppearance[:,1,:,:]/lightcolour[:,1,:,:],1)
	WBbCh = torch.unsqueeze(rawAppearance[:,2,:,:]/lightcolour[:,2,:,:],1)
	ImwhiteBalanced = torch.cat((WBrCh,WBgCh,WBbCh),1)
	return ImwhiteBalanced

def findT(Tmatrix,BGrid):
# Inputs:
#     Tmatrix          : 128 x 128 x 9 
#     BGrid            : 2 x 1 x 1 x B 
#  Output:
#     T_RAW2RGB        :  B x 9 x 1 x 1
##
    # to make dimesion B x 9 x 128 x 128
	nbatch = BGrid.size()[3] 
	Tmatrix = torch.unsqueeze(Tmatrix,0)
	Tmatrix = Tmatrix.permute(0,3,1,2) # 1 x 9 x 128 x 128
	Tmatrix = Tmatrix.repeat(nbatch,1,1,1) # B x 9 x 128 x 128

	BGrid = BGrid.permute(3,1,2,0) # B x 1 x 1 x 2 to match grid dimension
	
	T_RAW2XYZ =  nn.functional.grid_sample(Tmatrix,BGrid, mode='bilinear')
	#T_RAW2RGB.name ='T_RAW2RGB';
	return T_RAW2XYZ

def fromRawTosRGB(imWB,T_RAW2XYZ):

# Inputs:
#     imWB: B X 3 X H X W  
#     T_RAW2RGB  :  B x 9 x 1 x 1
# Output:
#     sRGBim : B X 3 X H X W
##
	Ix = T_RAW2XYZ[:,0,:,:] * imWB[:,0,:,:] + T_RAW2XYZ[:,3,:,:] * imWB[:,1,:,:] + T_RAW2XYZ[:,6,:,:] * imWB[:,2,:,:] # B X 1 X H X W
	Iy = T_RAW2XYZ[:,1,:,:] * imWB[:,0,:,:] + T_RAW2XYZ[:,4,:,:] * imWB[:,1,:,:] + T_RAW2XYZ[:,7,:,:] * imWB[:,2,:,:] 
	Iz = T_RAW2XYZ[:,2,:,:] * imWB[:,0,:,:] + T_RAW2XYZ[:,5,:,:] * imWB[:,1,:,:] + T_RAW2XYZ[:,8,:,:] * imWB[:,2,:,:] 
	Ix = torch.unsqueeze(Ix,1)
	Iy = torch.unsqueeze(Iy,1)
	Iz = torch.unsqueeze(Iz,1)
	Ixyz = torch.cat((Ix,Iy,Iz),1) # B X 3 X H X W
	Txyzrgb = torch.tensor([3.2406, -1.5372, -0.4986, -0.9689, 1.8758, 0.0415, 0.0557, -0.2040, 1.057 ])

	# if isa(imWB, 'Layer')
	#   Txyzrgb = Param('value',Txyzrgb,'learningRate',0); Txyzrgb.name='Txyzrgb';
	# end 

	R = Txyzrgb[0] * Ixyz[:,0,:,:] + Txyzrgb[3] * Ixyz[:,1,:,:] + Txyzrgb[6] * Ixyz[:,2,:,:]  # R
	G = Txyzrgb[1] * Ixyz[:,0,:,:] + Txyzrgb[4] * Ixyz[:,1,:,:] + Txyzrgb[7] * Ixyz[:,2,:,:]  # G
	B = Txyzrgb[2] * Ixyz[:,0,:,:] + Txyzrgb[5] * Ixyz[:,1,:,:] + Txyzrgb[8] * Ixyz[:,2,:,:]  # B

	R = torch.unsqueeze(R,1)
	G = torch.unsqueeze(G,1)
	B = torch.unsqueeze(B,1)
	sRGBim = torch.cat((R,G,B),1)
	rel = nn.ReLU()
	sRGBim =  rel(sRGBim)
	return sRGBim 

