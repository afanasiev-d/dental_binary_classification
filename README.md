# **Dental Image Classification using CNN** 🦷📊  

This project focuses on **binary classification of dental images** using a **Convolutional Neural Network (CNN)**. The goal is to classify teeth as **Anterior (Front) or Posterior (Back)** using **multi-modal grayscale and color image data**.  

The model is built using **TensorFlow** and **Keras**, with preprocessing steps, data augmentation strategies, and evaluation metrics to improve classification accuracy.  

---

## **📂 Project Structure**  

```
📎 dental-classification
📁 technical_challenge_data
    📁 modality1 (grayscale images)
    📁 modality2 (grayscale images)
    📁 modality3 (grayscale images)
    📁 modality4 (RGB images)
📁 models (saved trained models)
📁 logs (TensorBoard logs)
📃 dental_classification.py (Main script)
📃 requirements.txt (Dependencies)
📃 README.md (Project documentation)
```

---

## **📌 Key Features**  

✅ Multi-modal data: **4 different imaging modalities per sample**  
✅ CNN-based architecture optimized for **image classification**  
✅ Automatic **dataset preprocessing** and feature extraction  
✅ **Class balancing strategy** for **imbalanced data**  
✅ **Evaluation metrics**: Confusion matrix, accuracy, precision, recall, and F1-score  
✅ **Model saving & reusability** for further improvements  

---

## **🚀 Getting Started**  

### **1️⃣ Installation**  

Clone this repository and install dependencies:

```bash
git clone https://github.com/your-username/dental-classification.git
cd dental-classification
pip install -r requirements.txt
```

### **2️⃣ Data Preparation**  

The dataset is provided in a **ZIP file** and needs to be extracted:

```python
from zipfile import ZipFile  

path_zip_file = "technical_challenge_data.zip"  

with ZipFile(path_zip_file, 'r') as zip_ref:
    zip_ref.extractall()
print("Dataset extracted successfully!")
```

Each sample consists of **4 modalities** (3 grayscale and 1 RGB), which will be stacked together as a **(256, 256, 6) tensor**.

---

## **🪟 Data Preprocessing**  

1️⃣ **Extract sample names**  
2️⃣ **Assign labels (Anterior = 1, Posterior = 0)**
3️⃣ **Split data into Train (70%), Validation (22.5%), and Test (7.5%)**  
4️⃣ **Load images from different modalities and merge them into a single tensor**  

```python
df = pd.DataFrame()
df["Sample"] = list_sample_names
df["ToothID"] = df["Sample"].str.findall(r'(\d+(?:\.\d+)?)').str[1].astype(int)
df["target"] = 0
df.loc[df["ToothID"].isin(list(range(6, 12))) | df["ToothID"].isin(list(range(22, 28))), "target"] = 1
```

👍 **Class Balance**:  
- **Anterior Teeth (Class 1)** = **26.9%**  
- **Posterior Teeth (Class 0)** = **73.1%**  

⚠️ **Data is slightly imbalanced!**  
- One approach to improve performance: **Generate synthetic images using augmentation or GenAI models**.

---

## **🛠 Model Development**  

The CNN follows a **standard architecture**:  
- **Convolutional layers** for feature extraction  
- **MaxPooling layers** to reduce dimensionality  
- **Flatten layer** for dense connections  
- **Dense layers** with **Dropout** for regularization  
- **Sigmoid activation** for binary classification  

```python
model = Sequential()
model.add(Conv2D(16, (3,3), activation='relu', input_shape=(256,256,6)))
model.add(MaxPooling2D())
model.add(Conv2D(32, (3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Conv2D(16, (3,3), activation='relu'))
model.add(MaxPooling2D())
model.add(Flatten())
model.add(Dense(256, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(1, activation='sigmoid'))
```

---

## **📊 Model Evaluation**  

### **1️⃣ Training Performance**
```python
plt.plot(history.history["accuracy"], label="Train Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")
plt.legend()
plt.title("Training & Validation Accuracy")
plt.show()
```

### **2️⃣ Test Performance**  
```python
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_acc:.4f}")
```

### **3️⃣ Confusion Matrix**
```python
from sklearn.metrics import confusion_matrix
import seaborn as sns

cm = confusion_matrix(y_test, model.predict(X_test).round())
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["Posterior", "Anterior"], yticklabels=["Posterior", "Anterior"])
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title("Confusion Matrix")
plt.show()
```

### **4️⃣ Classification Report**
```python
from sklearn.metrics import classification_report
print(classification_report(y_test, model.predict(X_test).round()))
```

---

## **📀 Saving and Loading the Model**  

```python
model.save("models/dental_classification.h5")
```

To load the model:
```python
from tensorflow.keras.models import load_model
model = load_model("models/dental_classification.h5")
```

---

## **📚 Summary**  

✅ Successfully developed a **CNN-based Dental Classification Model**  
✅ Processed **multi-modal grayscale & color images**  
✅ Achieved **93.8% accuracy** on the test set  
✅ Saved & deployed model for **further improvements**  

---

## **📃 References**  

- TensorFlow Docs: [https://www.tensorflow.org/](https://www.tensorflow.org/)  
- Keras API: [https://keras.io/](https://keras.io/)  
- Scikit-learn: [https://scikit-learn.org/](https://scikit-learn.org/)  


📜 References
TensorFlow Docs: https://www.tensorflow.org/
Keras API: https://keras.io/
Scikit-learn: https://scikit-learn.org/
