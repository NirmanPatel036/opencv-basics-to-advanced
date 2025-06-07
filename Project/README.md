# Simpsons Character Prediction Model 🎭

![Simpsons Banner](https://upload.wikimedia.org/wikipedia/commons/thumb/b/bc/The_Simpsons_Logo.svg/1200px-The_Simpsons_Logo.svg.png)

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.0%2B-orange?logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-2.4%2B-red?logo=keras&logoColor=white)
![Kaggle](https://img.shields.io/badge/Kaggle-Notebook-blue?logo=kaggle&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-yellow?logo=mit&logoColor=white)

A deep learning project that uses Convolutional Neural Networks (CNN) to classify and predict Simpsons characters from images. This is an ongoing project with moderate performance that serves as a foundation for further improvements.

## 📋 Table of Contents

- [Project Overview](#project-overview)
- [Dataset](#dataset)
- [Model Architecture](#model-architecture)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [Key Features](#key-features)
- [Technical Details](#technical-details)
- [Current Challenges](#current-challenges)
- [Contributing](#contributing)
- [License](#license)

## 🚀 Project Overview

This project implements a computer vision solution to automatically classify Simpsons characters using deep learning techniques. The model is currently trained on a diverse dataset of character images and achieves moderate accuracy, with significant room for improvement and optimization.

### Objectives
- Build a CNN model for character classification
- Establish a baseline for Simpsons character recognition
- Implement data augmentation techniques for better generalization
- Create a foundation for future model improvements
- Develop a framework for character prediction

### Key Technologies
- **Deep Learning Framework**: TensorFlow/Keras
- **Computer Vision**: OpenCV for image preprocessing
- **Data Analysis**: Pandas, NumPy for data manipulation
- **Visualization**: Matplotlib, Seaborn for plotting results
- **Platform**: Kaggle Notebooks for development and training

## 📊 Dataset

### Dataset Source
- **Primary Source**: [Kaggle Simpsons Dataset](https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset)
- **Characters**: 20+ main Simpsons characters
- **Images**: 1000+ images per character (where available)
- **Format**: JPEG/PNG images of various sizes

### Dataset Structure
```
simpsons-dataset/
├── abraham_grampa_simpson/
├── apu_nahasapeemapetilon/
├── bart_simpson/
├── chief_wiggum/
├── comic_book_guy/
├── edna_krabappel/
├── homer_simpson/
├── krusty_the_clown/
├── lisa_simpson/
├── marge_simpson/
├── milhouse_van_houten/
├── moe_szyslak/
├── ned_flanders/
├── nelson_muntz/
├── principal_skinner/
├── sideshow_bob/
└── ... (additional characters)
```

### Data Preprocessing
- **Image Resizing**: Standardized to 224x224 pixels
- **Normalization**: Pixel values scaled to [0,1] range
- **Data Augmentation**: Rotation, zoom, shift, and flip transformations
- **Train-Test Split**: 80% training, 20% validation

## 🏗️ Model Architecture

### CNN Architecture
The model uses a custom Convolutional Neural Network with the following layers:

```python
# Model Architecture Overview
Input Layer: (224, 224, 3)
├── Conv2D(32, (3,3)) + ReLU + MaxPooling2D(2,2)
├── Conv2D(64, (3,3)) + ReLU + MaxPooling2D(2,2)
├── Conv2D(128, (3,3)) + ReLU + MaxPooling2D(2,2)
├── Conv2D(256, (3,3)) + ReLU + MaxPooling2D(2,2)
├── Flatten()
├── Dense(512) + ReLU + Dropout(0.5)
├── Dense(256) + ReLU + Dropout(0.3)
└── Dense(num_classes) + Softmax
```

### Model Specifications
- **Input Shape**: (224, 224, 3) - RGB images
- **Total Parameters**: ~15M parameters
- **Optimizer**: Adam (learning_rate=0.001)
- **Loss Function**: Categorical Crossentropy
- **Metrics**: Accuracy, Top-5 Accuracy
- **Regularization**: Dropout layers, L2 regularization

### Training Configuration
- **Batch Size**: 32
- **Epochs**: 50-100 (with early stopping)
- **Learning Rate**: 0.001 with ReduceLROnPlateau
- **Data Augmentation**: Enabled during training
- **Callbacks**: EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

## 🛠️ Installation

### Prerequisites
```bash
Python 3.8+
pip package manager
```

### Required Libraries
```bash
# Core deep learning libraries
pip install tensorflow>=2.0.0
pip install keras>=2.4.0

# Data manipulation and analysis
pip install numpy>=1.19.0
pip install pandas>=1.1.0

# Image processing
pip install opencv-python>=4.5.0
pip install pillow>=8.0.0

# Visualization
pip install matplotlib>=3.3.0
pip install seaborn>=0.11.0

# Jupyter notebook support
pip install jupyter
pip install ipykernel
```
### Kaggle Notebook 🔗
https://www.kaggle.com/code/nirmanmpatel/simpsons-prediction-model

### Setup Instructions
1. **Clone or download the Kaggle notebook**
2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```
3. **Download the dataset** from Kaggle
4. **Run the notebook** in Kaggle or local Jupyter environment

## 💻 Usage

### Training the Model
```python
# Load and preprocess data
train_generator, validation_generator = create_data_generators()

# Build model
model = create_cnn_model(num_classes=20)

# Compile model
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Train model
history = model.fit(
    train_generator,
    validation_data=validation_generator,
    epochs=50,
    callbacks=[early_stopping, model_checkpoint]
)
```

### Making Predictions
```python
# Load trained model
model = load_model('simpsons_model.h5')

# Predict character
def predict_character(image_path):
    img = load_and_preprocess_image(image_path)
    prediction = model.predict(img)
    character = classes[np.argmax(prediction)]
    confidence = np.max(prediction)
    return character, confidence

# Example usage
character, confidence = predict_character('test_image.jpg')
print(f"Predicted: {character} (Confidence: {confidence:.2%})")
```

### Visualization
```python
# Plot training history
def plot_training_history(history):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.legend()
    
    plt.show()
```

## 📈 Results

### Current Model Performance
- **Training Accuracy**: 46.88%
- **Validation Accuracy**: 52.71%
- **Performance Gap**: The model shows room for significant improvement
- **Validation vs Training**: Validation accuracy is higher than training accuracy, suggesting the model may benefit from additional training or architecture adjustments

### Performance Analysis
The current model performance indicates several areas for improvement:
- **Moderate Accuracy**: With ~53% validation accuracy, the model correctly identifies characters about half the time
- **Baseline Established**: This serves as a solid baseline for future improvements
- **Learning Pattern**: The validation accuracy being higher than training accuracy suggests potential for better optimization

### Current Challenges Observed
- Multi-class classification with 20+ characters is complex
- Character similarity poses classification difficulties
- Dataset quality and balance may need attention
- Model architecture may require optimization

## ✨ Key Features

### Data Augmentation
- **Geometric Transformations**: Rotation (±15°), zoom (±20%), shift (±20%)
- **Flip Transformations**: Horizontal flipping enabled
- **Brightness/Contrast**: Random adjustments for robustness
- **Noise Addition**: Gaussian noise for improved generalization

### Model Optimization
- **Transfer Learning**: Option to use pre-trained models (VGG16, ResNet50)
- **Fine-tuning**: Layer-wise learning rate adjustment
- **Regularization**: Dropout, L1/L2 regularization, batch normalization
- **Early Stopping**: Prevents overfitting with patience-based stopping

### Evaluation Metrics
- **Accuracy Metrics**: Top-1 and Top-5 accuracy tracking
- **Per-Class Metrics**: Precision, recall, F1-score analysis planned
- **Confusion Matrix**: Detailed error analysis for improvement insights
- **Learning Curves**: Training progress visualization

## 🔧 Technical Details

### Image Preprocessing Pipeline
```python
def preprocess_image(image_path, target_size=(224, 224)):
    # Load image
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Resize
    img = cv2.resize(img, target_size)
    
    # Normalize
    img = img.astype('float32') / 255.0
    
    # Add batch dimension
    img = np.expand_dims(img, axis=0)
    
    return img
```

### Data Augmentation Configuration
```python
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=15,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True,
    zoom_range=0.2,
    shear_range=0.1,
    fill_mode='nearest'
)
```

### Model Callbacks
```python
callbacks = [
    EarlyStopping(patience=10, restore_best_weights=True),
    ModelCheckpoint('best_model.h5', save_best_only=True),
    ReduceLROnPlateau(factor=0.5, patience=5, min_lr=1e-7)
]
```

## 🚧 Current Challenges

### Performance Issues
- **Accuracy Gap**: Current ~53% accuracy indicates significant room for improvement
- **Class Imbalance**: Some characters may have insufficient training data
- **Feature Complexity**: Animated characters present unique classification challenges
- **Overfitting Risk**: Need to balance model complexity with generalization

### Technical Challenges
- **Dataset Quality**: Image quality and consistency variations
- **Character Similarity**: Some characters share similar visual features
- **Background Noise**: Images may contain distracting background elements
- **Pose Variation**: Characters appear in various poses and expressions

### Identified Areas for Improvement
- **Model Architecture**: Current architecture may be too simple or complex
- **Hyperparameter Tuning**: Learning rate, batch size, and regularization need optimization
- **Data Preprocessing**: Enhanced preprocessing techniques needed
- **Training Strategy**: Longer training or different optimization strategies

## 📊 Experimental Analysis

### Current Performance Breakdown
- **Training Accuracy**: 46.88% - Indicates model is still learning
- **Validation Accuracy**: 52.71% - Better than training, suggests potential
- **Performance Gap**: Room for significant improvement
- **Baseline Established**: Good foundation for iterative improvements

### Next Experiments Planned
1. **Architecture Comparison**: Test different CNN architectures
2. **Transfer Learning**: Compare pre-trained model performance
3. **Hyperparameter Grid Search**: Systematic parameter optimization
4. **Data Augmentation Impact**: Measure effect of different augmentation strategies
5. **Class Balance Analysis**: Investigate per-class performance

## 🤝 Contributing

Contributions are especially welcome given the current performance challenges! Here's how you can help:

### Priority Areas for Contribution
1. **Model Architecture**: Experiment with different CNN designs
2. **Hyperparameter Tuning**: Help optimize training parameters
3. **Data Quality**: Improve dataset cleaning and preparation
4. **Transfer Learning**: Implement and compare pre-trained models
5. **Performance Analysis**: Deep dive into model performance issues

### Development Setup
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Test thoroughly
5. Submit a pull request with performance comparisons

### Contribution Guidelines
- Include performance metrics for any model changes
- Document hyperparameter choices and reasoning
- Provide before/after accuracy comparisons
- Add comprehensive docstrings
- Update documentation with new findings

## 📚 References and Resources

### Academic Papers
- [ImageNet Classification with Deep Convolutional Neural Networks](https://papers.nips.cc/paper/4824-imagenet-classification-with-deep-convolutional-neural-networks.pdf)
- [Very Deep Convolutional Networks for Large-Scale Image Recognition](https://arxiv.org/abs/1409.1556)
- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

### Datasets and Competitions
- [Kaggle Simpsons Dataset](https://www.kaggle.com/datasets/alexattia/the-simpsons-characters-dataset)
- [ImageNet Large Scale Visual Recognition Challenge](http://www.image-net.org/challenges/LSVRC/)

### Tutorials and Guides
- [TensorFlow Image Classification Tutorial](https://www.tensorflow.org/tutorials/images/classification)
- [Keras Documentation](https://keras.io/)
- [Computer Vision with Deep Learning](https://cs231n.github.io/)

## 🐛 Troubleshooting

### Current Known Issues

#### Low Accuracy Solutions
- **Increase Model Complexity**: Add more layers or parameters
- **Better Data Preprocessing**: Improve image quality and consistency
- **Transfer Learning**: Use pre-trained models as feature extractors
- **Extended Training**: Allow model to train for more epochs
- **Learning Rate Adjustment**: Fine-tune learning rate schedule

#### Common Performance Issues
```python
# Try different optimizers
optimizer = tf.keras.optimizers.SGD(learning_rate=0.01, momentum=0.9)
# or
optimizer = tf.keras.optimizers.RMSprop(learning_rate=0.001)

# Adjust learning rate schedule
lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=0.001,
    decay_steps=1000,
    decay_rate=0.9
)
```

#### Memory and Training Issues
```python
# Reduce batch size if memory issues
batch_size = 16  # Instead of 32

# Use mixed precision training
from tensorflow.keras.mixed_precision import experimental as mixed_precision
policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_policy(policy)
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **The Simpsons Dataset**: Thanks to the Kaggle community for providing the dataset
- **TensorFlow Team**: For the amazing deep learning framework
- **Kaggle Platform**: For providing free GPU resources for training
- **The Simpsons**: For creating such memorable and distinct characters
- **Open Source Community**: For contributing to the tools and libraries used

## 📞 Contact

- Email: nirman0511@gmail.com
- LinkedIn: www.linkedin.com/in/nirmanpatel

---

⭐ **This project is actively seeking contributions to improve model performance!**

*Current Status: Baseline established at ~53% accuracy - significant room for improvement!* 🚀
