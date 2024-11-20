Final Project for BIOL 8803 ML Methods for Neural/Behavioral Data 
Ben Doshna and Ellen Liu 
###

The Essence of this project is to train Binary Classifier Models to Predict whether the Motor Program Activity comes from A moth at the Beggining or end of feeding. 

The Comprehensive Motor Program (putney pnas 2019) Of Manduca Sexta has information in both spike count and precise spike timing of 10 muscles within a single wingstroke. 

The CMP was recorded as Moths Fed from a robotic flower. 

The data for these Models is in two csv files:
  highdimensionbigdata.csv includes each Moth Trial and wingbeat but is very sparse in some of the spike timing dimensions
  23dimcountime.csv includes only the spike timing of the first spike for a given muscle and only includes wingbeats where all muscles fire (not sparse) 

The Models Trained Are:
  Extreme Gradient Boosting 
  SVM (Both linear and rbf kernels) 

Feature importance is done using both importance metrics inherent to the models as well as SHAP and TreeShap
Shap is a game-theoretic approach to feature importance with nonlinear models as described in:
  Lundberg and Lee (Neurips 2017) 
  Lundberg et al (Nature Machine Intelligence 2020) 
