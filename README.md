# 🚀 **End-to-End ML Pipeline for Real-Time Material Classification**  
### _(Scrap Simulation Challenge — Internship Assignment)_

This project simulates a **real-time classification system** for scrap materials using a **Convolutional Neural Network (CNN)**.  
The goal is to detect and classify waste types (_plastic_, _paper_, _metal_, etc.) from images and simulate a **conveyor-belt-like sorting loop**.

---

## 📊 **Dataset Used & Why**

- **Dataset**: [TrashNet](https://github.com/garythung/trashnet)  
- **Classes**: `'cardboard'`, `'glass'`, `'metal'`, `'paper'`, `'plastic'`, `'trash'`

**Why TrashNet?**
- 🟢 Publicly available
- ⚖️ Balanced class distribution with labeled images
- 🏠 Includes common household waste types relevant to real-world sorting systems

---

## 🧠 **Architecture & Training Process**

- **Model**: *Pretrained ResNet18* (Transfer Learning)
- **Modification**: Final layer adapted to predict **6 classes**
- **Input Size**: `224 x 224`
- **Transforms**: Resize, Normalize, Augment (*Random Flip*, *Rotate*, *ColorJitter*)
- **Loss Function**: `CrossEntropyLoss`
- **Epochs**: `5`
- **Batch Size**: `32`

**🧪 Metrics Used**:
- Accuracy  
- Precision  
- Recall  
- Classification Report

📌 The best model is saved as **`best_model.pt`** whenever validation accuracy improves.

---

## 🚀 **Deployment Decisions**

- Converted model to **TorchScript** (`model_scripted.pt`) for lightweight deployment
- Enables **fast inference** without full PyTorch
- Supports **single-image CLI inference** using `inference.py`
- Simulates **real-time classification** using `simulation_loop.py`, processing images one at a time and logging results

---

• Folder Structure :- 
  <pre> 
    End-to-End ML Pipeline for Real-Time Material Classification/
      ├── data/
      │   ├── cardboard/
      │   ├── glass/
      │   ├── metal/
      │   ├── plastic/
      │   ├── paper/
      │   ├── trash/
      │   └── conveyor_simulation/   # Input frames for simulation
      ├── models/
      │   ├── best_model.pt          # Trained PyTorch model
      │   └── model_scripted.pt      # TorchScript model for deployment
      ├── results/
      │   └── predictions.csv        # Output logs from simulation
      ├── src/
      │   ├── data_preparation.py    # Dataset cleaning & preparation
      │   ├── train_model.py         # ResNet18 training script
      │   ├── inference.py           # Single image inference
      │   └── simulation_loop.py     # Conveyor belt simulation loop
      ├── README.md
      └── performance_report.pptx    # Performance summary and visuals

     </pre>

 
 🛠️ Instructions to Run
🔹 Step 1: Setup Environment
bash
Copy
Edit
python -m venv venv
.\venv\Scripts\activate
pip install torch torchvision Pillow scikit-learn
🔹 Step 2: Clean & Load Dataset
bash
Copy
Edit
cd src
python data_preparation.py
🔹 Step 3: Train the Model
bash
Copy
Edit
python train_model.py
🔹 Step 4: Convert Model to TorchScript & Run Inference
bash
Copy
Edit
# Example single image inference
python inference.py ../data/conveyor_simulation/glass2.jpg

# Use your own image
python inference.py ../data/conveyor_simulation/<image_name>.jpg
🔹 Step 5: Simulate Conveyor Belt Sorting
bash
Copy
Edit
python simulation_loop.py
Simulates real-time classification — for each image:

Classifies the image

Logs result to console & saves to CSV

Flags low-confidence predictions (< 0.6)

📍 Sample Output
cardboard1.jpg   →  cardboard (0.9691)  
cardboard10.jpg  →  cardboard (0.7219)  
cardboard2.jpg   →  cardboard (0.5978) ⚠️ LOW CONFIDENCE  
glass1.jpg       →  metal (0.4636) ⚠️ LOW CONFIDENCE  
glass10.jpg      →  glass (0.5032) ⚠️ LOW CONFIDENCE  

📝 Results saved in: results/predictions.csv  
⚠️ Images with confidence < 0.6 are automatically flagged.
