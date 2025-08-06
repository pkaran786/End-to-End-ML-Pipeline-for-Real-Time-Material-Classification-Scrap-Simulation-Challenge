**End-to-End ML Pipeline for Real-Time Material Classification (Scrap Simulation Challenge)** :- 

This project simulates a real-time classification system for scrap materials using a Convolutional Neural Network (CNN). The goal is to detect and classify waste types (plastic, paper, metal, etc.) from images and simulate a conveyor-belt-like sorting loop.
It is built as part of internship assignment.

• Dataset Used & Why :- 
  Dataset : [Trashnet](https://github.com/garythung/trashnet)
  Classes : 'cardboard', 'glass', 'metal', 'paper', ,'plastic', 'trash'

  Why TrashNet?
  - Publicly available
  - Balanced class distribution with labeled images.
  - Includes common household waste types relevant to real-world sorting systems.
    
• Architecture & Training Process :- 
  - Model : Pretrained ResNet18 using transfer learning.
  - Modified :  the final layer to predict 6 classes.
  - Input Size : 224 x 224.
  - Transforms : Resize, Normalize, Augment (Random Flip, Rotate, ColorJitter).
  - Loss : CrossEntropyLoss.
  - Epochs : 5
  - Batch Size : 32

 Metrics Used :- 
 - Accuracy
 - Precision
 - Recall
 - Classification Report
   
Trained model is saved as `best_model.pt` whenever validation accuracy improves.

• Deployment Decisions :- 
  - Used *TorchScript* for lightweight deployment (`model_scripted.pt`)
  - Enables fast inference without needing full PyTorch
  - Single image inference supported via CLI (`inference.py`)
  - Simulated real-time deployment using `simulation_loop.py`, processing one image at a time and logging results

• Folder Structure :- 
    End-to-End ML Pipeline for Real-Time Material Classification/  
    ├── data/  
    │ └── conveyor_simulation/# Simulation images (input frames)  
    | └── cardboard/  
    | └── glass/  
    | └── metal/  
    | └── plastic/  
    | └── paper/   
    | └── trash/  
    | └── conveyor_simulation/  
    ├── models/  
    │ ├── best_model.pt # Trained PyTorch model  
    │ └── model_scripted.pt # TorchScript model for inference  
    ├── results/  
    │ └── predictions.csv # Output from simulation  
    ├── src/  
    │ ├── data_preparation.py # Cleans and loads dataset  
    │ ├── train_model.py # Trains ResNet18 with transfer learning  
    │ ├── inference.py # Predicts class of a single image  
    │ └── simulation_loop.py # Simulates real-time sorting system  
    ├── README.md  
    └── performance_report.pptx # Visuals + performance summary  

 
 • Instructions to run :- 

  - Step 1 :- Create a virtual enviornmnet and download the following packages  
    > Run python -m venv venv  
    > Run .\venv\Scripts\activate  
    > Run pip install torch torchvision Pillow scikit-learn to download the packages given below  
    `torch`, `torchvision`, `scikit-learn`, `Pillow`   

  - Step 2 :- Cleaning & Loading of Dataset  
    > First Relocate to src folder using cd .\src\  
    > Run python data_preparation.py to clean the dataset.  
    > After cleaning & loading the dataset, the next step is to train the model.  

  - Step 3 :- Train the Model  
    > To train the model using Resnet18, use the command below.  
    > Run python train_model.py  
    > After training the model now it's time to Convert your model to TorchScript and create a lightweight inference script.  

  - Step 4 :- Converting Model into TorchScript for Single Image Inference  
    > To execute this following step, use the command below.  
    > Run python inference.py ../data/conveyor_simulation/glass2.jpg for Single Image Inference.  
    > If you want to use other image you can run this command according to file directory - python inference.py ../data/conveyor_simulation/the_image_you_want_to_select.jpg  
    > After this step, now it's time to build a dummy conveyor simulation.  

  - Step 5 :- Dummy conveyor simulation ( mimics frames being captured at intervals from a video or image folder)  
    > For each frame: Classify, Log output to console + store in a result CSV, Print confidence threshold flag if low.  
    > To execute this, Run simulation_loop.py  

  - Sample Output :- 
     cardboard1.jpg →  cardboard (0.9691)   
     cardboard10.jpg →  cardboard (0.7219)   
     cardboard2.jpg →  cardboard (0.5978) ⚠️ LOW CONFIDENCE  
     glass1.jpg →  metal (0.4636) ⚠️ LOW CONFIDENCE  
     glass10.jpg →  glass (0.5032) ⚠️ LOW CONFIDENCE  
    
  - Results are saved in results/predictions.csv.
  - Images with confidence < 0.6 are flagged.
    
