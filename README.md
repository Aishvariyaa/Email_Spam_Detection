
## 📧 Spam Email Classifier (Binary Classification)  

### 📌 Overview  
This project builds a **Spam Email Classifier** that detects whether an email is **Spam or Not Spam** using **machine learning algorithms**. 📩🚫  

### 🔍 Key Steps  
✅ **Dataset Preprocessing:**  
   - Removed **punctuation & special characters** ✂️  
   - Converted text to **lowercase** 🔡  
   - Applied **Tokenization & Lemmatization** using **NLTK** 📝  
   - Transformed text into numerical vectors using **TF-IDF Vectorization** 🔢  
✅ **Model Training:**  
   - **Logistic Regression** 📊  
   - **Support Vector Machine (SVM)** 📈  
   - **Naïve Bayes Classifier** 🤖  
✅ **Model Evaluation:**  
   - **Accuracy, Precision, Recall, F1-Score** for performance comparison 📊  

### 📂 Project Structure  
```
Spam-Email-Classifier/
│── README.md  # Documentation  
│── spam_classifier.ipynb  # Jupyter Notebook (Model Training & Evaluation)    
```  

### 🔗 Dataset Link  
📌 **Dataset Source:** [Spam SMS Dataset](#) 

### 📊 Dataset  
For this project, we use a **Spam SMS dataset** containing:  

📌 **Features**:  
- **Text Message:** The actual content of the email/SMS  
- **Word Frequency Features:** TF-IDF scores of words  

🎯 **Target Variable**: **Spam (1) / Not Spam (0)**  

### 🔧 Technologies Used  
🔹 Python  
🔹 Pandas & NumPy (Data Processing)  
🔹 Scikit-learn (ML Models & Evaluation)  
🔹 NLTK (Natural Language Processing)  
🔹 Matplotlib & Seaborn (Data Visualization)  

### 📜 How to Run the Project?  
#### 1️⃣ Clone the Repository  
```bash
git clone https://github.com/Aishvariyaa/Spam-Email-Classifier.git
cd Spam-Email-Classifier
```  

#### 2️⃣ Install Dependencies  
```bash
pip install -r requirements.txt
```

#### 4️⃣ Run the Jupyter Notebook  
```bash
jupyter notebook spam_classifier.ipynb
```  

### 📈 Model Performance  
📌 **Model Results Based on Your Provided Data:**  

#### 🔹 **Naïve Bayes Model**  
- **Accuracy:** **100.0%**  
- **Precision:** 1.00  
- **Recall:** 1.00  
- **F1-Score:** 1.00  

#### 🔹 **Logistic Regression Model**  
- **Accuracy:** **100.0%**  
- **Precision:** 1.00  
- **Recall:** 1.00  
- **F1-Score:** 1.00  

#### 🔹 **Support Vector Machine (SVM) Model**  
- **Accuracy:** **100.0%**  
- **Precision:** 1.00  
- **Recall:** 1.00  
- **F1-Score:** 1.00  

📌 **Key Observations**  
- All models achieved **100% accuracy, precision, recall, and F1-score**, indicating a **perfect classification** on the test dataset. 🎯  
- This might suggest the dataset is **too clean or too small**, or there could be potential **overfitting** that needs further analysis.  

### 📌 Next Steps  
🔹 Test on a **larger, real-world dataset** to check for overfitting.  
🔹 Experiment with **Deep Learning models (LSTMs)** for enhanced text classification.  
🔹 Deploy the model as a **Flask API** for real-time email filtering.  
