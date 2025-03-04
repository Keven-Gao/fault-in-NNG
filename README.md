# fault-in-NNG

This is a code repository for the paper **Fault Representation in Structural Modeling with Implicit Neural Representations (https://doi.org/10.1016/j.cageo.2025.105911)**. Including 2 jupyter notebooks of the cases implement from the paper.

## Required packages:  
torch  
pandas  
warnings  
os  
numpy  
time  
matplotlib  
pyvistaqt (optional)  
pyvista  
pyqt

## Workflow diagram: 
![image](https://github.com/Keven-Gao/fault-in-NNG/blob/main/workflow.png)
Illustration of fault modelling in implicit neural network. In part 1, faults are encoded as additional features, coordinates and fault features are concatenated to form the neural network input. Part 2 is the model training step, which is the same as GeoINR’s training process. Part 3 is the visualization step, the geological structure represented by iso-surfaces.
