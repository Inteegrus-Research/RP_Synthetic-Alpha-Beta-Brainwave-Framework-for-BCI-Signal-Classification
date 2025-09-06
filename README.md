# 🧠 EEG Alpha–Beta Classification Framework

This project presents a lightweight Brain-Computer Interface (BCI) simulation pipeline using synthetic EEG signals in the alpha (8–12 Hz) and beta (13–30 Hz) bands. It enables rapid prototyping and benchmarking with a compact 1D-CNN classifier, ideal for educational and research use.

---

## 🛠️ Tech Stack
- Python 3.x  
- NumPy & SciPy for signal generation  
- Scikit-learn for preprocessing  
- TensorFlow/Keras for 1D-CNN modeling  
- Matplotlib for visualization  

---

## 🚀 Features
- 🎛️ Synthetic EEG signal generation with noise and blink artifacts  
- 🧾 Metadata creation for supervised learning  
- 📊 Normalization and preprocessing pipeline  
- 🧠 1D-CNN classifier for alpha–beta band detection  
- 📈 Performance metrics: accuracy, confusion matrix, classification report  

---

<pre>
## 📂 Folder Structure
EEG_AlphaBeta_Classifier/
├── mnt/ # Synthetic EEG dataset (CSV files)
├── generate_eeg_dataset.py # Simulates EEG signals
├── create_metadata.py # Maps each file to its label
├── load_and_normalize_dataset.py # Prepares data for training
├── train_eeg_cm_classifier.py # Trains and evaluates the 1D-CNN
├── Paper.docx # Draft manuscript
└── screenshot.pdf # Visual reference of folder layout
</pre>

---

## 🧪 How It Works
1. **Generate Dataset** → Run `generate_eeg_dataset.py` to simulate EEG signals.  
2. **Create Metadata** → Execute `create_metadata.py` to label each sample.  
3. **Normalize Data** → Use `load_and_normalize_dataset.py` to preprocess the dataset.  
4. **Train Model** → Launch `train_eeg_cm_classifier.py` to train and evaluate the classifier.  

---

## 📌 Applications
- BCI Prototyping  
- Cognitive Signal Processing Education  
- Neuroadaptive System Simulations  
- Assistive Communication Research  

---

## 📄 Documentation
- **Paper.docx**: Academic write-up of methodology and results  
- **screenshot.pdf**: Folder structure overview  

---

## 🔓 License
This project is licensed under the **MIT License**. Feel free to adapt or build upon it.  

---

## 👤 Author
**Keerthi Kumar K J**  
📧 inteegrus.research@gmail.com
