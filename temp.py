# opts.numEpochs = 200 ;
# opts.learningRate = 1e-5;    
 # nimages = 50765;
 # batchSize= 64;


celebaimdb_averageImage = torch.tensor([129.1863,104.7624,93.5940])
muim = torch.reshape(celebaimdb_averageImage,(1,3,1,1))
#     sRGBim : B X 3 X H X W
scaleRGB = sRGBim * 255 # B X 3 X H X W

# implement data DataLoader
# actualshading = Input()
# images = Input()
# actualmasks = Input()

Y1 = torch.ones_like(muim)
Y1 = Y1 * muim # 1 x 3 x 1 x 1
rgbim = ((scaleRGB) - Y1)
nbatchs = scaleRGB.size()[0]
X1 = torch.ones_like(rgbim)
rgbim = rgbim * X1 # B X 3 X H X W

# predictedShading  : B x 1 x 224 x 224

# predictedShading  : 224 x 224 x 1 x B matlab format
scale = torch.sum( torch.sum( (actualshading * predictedShading) * actualmasks, 2), 2) / torch.sum( torch.sum( torch.square(predictedShading) * actualmasks, 2), 2)

scale = torch.reshape(scale,(nbatchs,1,1,1))
predictedShading = predictedShading * scale
alpha = (actualshading - predictedShading) * actualmasks

## CNN weights

blossweight = 1e-4  
appweight = 1e-3
Shadingweight = 1e-5 
sparseweight = 1e-5 

#  Camera parameter loss regularixzation loss
priorB = torch.sum(torch.square(b))
priorloss = priorB * blossweight
ZY = torch.ones_like(priorloss)
priorloss = priorloss * ZY

# L2: appearance loss

delta = (images - rgbim ) * actualmasks # B X 3 X H X W
appearanceloss = (torch.sum(torch.square(delta)) / (224. * 224.)) * appweight   
Y = torch.ones_like(appearanceloss)
appearanceloss = appearanceloss * Y

# shadding loss
shadingloss = torch.sum(torch.square(alpha)) * Shadingweight
ff = torch.ones_like(shadingloss)
shadingloss = shadingloss * ff

# L1 sparsity loss:
sparsityloss = torch.sum(Specularities) * sparseweight
J = torch.ones_like(sparsityloss)
sparsityloss = sparsityloss * J

# Final Loss:
 loss = appearanceloss + priorloss + sparsityloss + shadingloss
