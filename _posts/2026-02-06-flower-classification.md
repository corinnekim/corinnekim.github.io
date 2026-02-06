---
layout: single
title: "When Pretrained Models Are Already Good Enough"
date: 2026-02-06
categories: [datascience]
tags: [deep-learning, tensorflow]
toc: true
toc_sticky: true
---
<!-- <span style="background-color: #fff5b1">text</span> -->
<!-- ![](/assets/images/arima/arima.png){: width='80%'} -->

![](/assets/images/flowers/flowers-la.png){: width='60%'}
<br>

I trained a deep learning model using transfer learning and compared different models, hyperparameters, and fine-tuning strategies to see how they affected performance. 

I used the popular [tensorflow flower dataset](https://www.tensorflow.org/datasets/catalog/tf_flowers), which contains about **3,670** images across **5** classes.

```text
0 - dandelion
1 - daisy
2 - tulips
3 - sunflowers
4 - roses
```

<br>

![](/assets/images/flowers/flowers.png){: width='100%'}

<br>

## Step 1. Baseline Model- EfficientNetB0

Building a model from scratch did not make sense for a dataset of this size, so I started with **EfficientNetB0**. 

It comes pre-trained on ImageNet and provides strong baseline features out of the box. While larger variants like **EfficientNetB7** exist, I chose B0 because it is lightweight and runs well in a local setup.

```python
import tensorflow as tf
from tensorflow.keras import layers, models

# 1. Define Model Architecture (Transfer Learning with EfficientNetB0)
base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(160, 160, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False # Freeze pre-trained weights

model = tf.keras.Sequential([
    layers.Input(shape=(160, 160, 3)),
    # EfficientNet expects [0, 255] input directly
    base_model,
    # Global Average Pooling to flatten 2D feature maps into 1D vectors
    layers.GlobalAveragePooling2D(),
    # Dropout layer to prevent overfitting
    layers.Dropout(0.2),
    # Output layer for classification (5 classes)
    layers.Dense(5, activation='softmax')
])

# 2. Compile Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
    metrics=['accuracy']
)

model.summary()
```

I froze the pre-trained weights of EfficientNetB0, resized the input images to 160×160, and trained the model with a batch size of 32. The model used the Adam optimizer and finished training in about 1.5 minutes.

```python
# Train baseline model
history = model.fit(
    train_batches,
    epochs=10,
    validation_data=validation_batches
)

test_loss, test_acc = model.evaluate(test_batches, verbose=1)
print(f"\n[Basline Model Test Result]")
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
print(f"\nTest Loss: {test_loss:.4f}")
```
 
**[Baseline Model Test Result]**     
Test Accuracy: 92.10%  
Test Loss: 0.2220    
<br>
![](/assets/images/flowers/confusion-matrix.png){: width='70%'}

Our baseline model did a decent job, but it often confused roses with tulips.

## Step 2. Optimizing the Baseline
While keeping the EfficientNetB0 layers frozen, I made the following changes:

- Data augmentation (2×): Added horizontal flips and random brightness adjustments.
- Increased the input size to 224×224 to retain more visual detail.
- Reduced L2 regularization from 0.01 to 1e-4.
- Early stopping: Stops training and restores the best weights after five epochs without improvement.
- Label smoothing (0.1): Reduces overconfident predictions.
- Learning rate scheduler: Reduces the learning rate when validation loss does not improve for two epochs.

```python
from tensorflow.keras import regularizers, callbacks

# 1. Environment & Hyperparameters
IMG_SIZE = 224
BATCH_SIZE = 32
NUM_CLASSES = 5

# 2. Preprocessing & Augmentation Functions
def preprocess(image, label):
    # Keep values in [0, 255] range (float32)
    image = tf.cast(image, tf.float32)
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, tf.one_hot(label, NUM_CLASSES)

def augment(image, label):
    # Augmentation on [0, 255] range data
    image = tf.image.random_flip_left_right(image)
    # Adjusted max_delta for 0-255 range (0.1 * 255 ≈ 25)
    image = tf.image.random_brightness(image, max_delta=25.0) 
    image = tf.clip_by_value(image, 0.0, 255.0) 
    return image, label

# 3. Dataset Pipeline
train_orig = raw_train.map(preprocess)
validation_batches = raw_validation.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test_batches = raw_test.map(preprocess).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# Generate and concatenate augmented data
train_aug = train_orig.map(augment)
train_combined = train_orig.concatenate(train_aug)

# Final training batches
train_batches = train_combined.shuffle(2000).batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

# 4. Model Architecture
base_model = tf.keras.applications.EfficientNetB0(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights='imagenet'
)
base_model.trainable = False

model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.Dropout(0.2),
    layers.Dense(NUM_CLASSES, 
                 activation='softmax',
                 kernel_regularizer=regularizers.l2(1e-4))
])

# 5. Compile Model
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# 6. Callbacks
early_stopping = callbacks.EarlyStopping(
    monitor='val_loss', 
    patience=5,
    restore_best_weights=True
)

lr_scheduler = callbacks.ReduceLROnPlateau(
    monitor='val_loss', 
    factor=0.5,
    patience=2,
    verbose=1,
    min_lr=1e-6
)

# 7. Train
history = model.fit(
    train_batches,
    epochs=30,
    validation_data=validation_batches,
    callbacks=[early_stopping, lr_scheduler]
)

# 8. Evaluation
test_loss, test_acc = model.evaluate(test_batches, verbose=1)
print(f"\n[Result After Optimization]")
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
print(f"\nTest Loss: {test_loss:.4f}")
```
![](/assets/images/flowers/step2-result.png){: width='100%'}

Training stopped at epoch 11, and the model restored the best weights from epoch 6.
Test accuracy increased by **2.72%p** over the baseline.

## Step 3. Fine-Tuning EfficientNetB0

I then fine-tuned the model by unfreezing the top 20 layers of EfficientNetB0 while keeping the rest frozen. To avoid disrupting the learned features, I recompiled the model with a much lower learning rate (1e-5).

```python
# 1. Unfreeze the top 20 layers of the base model
base_model.trainable = True

# Freeze all layers except the last 20
for layer in base_model.layers[:-20]:
    layer.trainable = False

# 2. Recompile with a very low learning rate (1e-5)
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss=tf.keras.losses.CategoricalCrossentropy(label_smoothing=0.1),
    metrics=['accuracy']
)

# 3. Continue training (using the same augmented 'train_batches')
# We train for a few more epochs to refine the weights
history_fine = model.fit(
    train_batches,
    epochs=10, 
    validation_data=validation_batches,
    callbacks=[early_stopping] 
)

# 4. Evaluation
test_loss, test_acc = model.evaluate(test_batches, verbose=1)
print(f"\n[Result After Fine-Tuning]")
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")
print(f"\nTest Loss: {test_loss:.4f}")
```

**[Result After Fine-Tuning]**     
Test Accuracy: 92.92%    
Test Loss: 0.5935
<br>

Surprisingly, the accuracy dropped after fine-tuning. Why is that?       

With a relatively small dataset like tf_flowers (3,670 images), unfreezing the top layers likely led to **overfitting**. Instead of learning general patterns, the model fit too closely to the data. The pre-trained features were already sufficient, and further training added noise.

This was a reminder that fine-tuning is not always helpful. In some cases, frozen models already perform well, and pushing them further can hurt performance.

## Step 4. More Experiments
I ran a few additional experiments. Unlike many machine learning models, deep learning usually requires choosing hyperparameters by hand. So it's best to run many small experiments quickly. Below are a few things I tested.

First, I removed label smoothing to see whether it was actually helping on this dataset.

![](/assets/images/flowers/result-without-label-smoothing.png){: width='100%'}

Without label smoothing, accuracy increased to **95.37%**, compared to 94.82% with label smoothing.    
For this task, the model performed better without label smoothing.

Next, since I had some extra time I tried a heavier model, **EfficientNetB3**, using the same hyperparameters.

![](/assets/images/flowers/result-b3.png){: width='100%'}

Accuracy improved from 95.37% to **95.64%**, but the improvement was small compared to the additional compute required.


## Summary
For a small dataset, pre-trained models worked better than fine-tuned ones, and label smoothing didn't improve performance. Fast experimentation with a lightweight model like EfficientNetB0 was most effective. Although EfficientNetB3 achieved slightly higher accuracy, it requires more than three times the resources, so B0 is enough for simple classification tasks.


<script>
  MathJax = {
    tex: {
      inlineMath: [['$', '$'], ['\\(', '\\)']]
    },
    svg: {
      fontCache: 'global'
    }
  };
</script>
<script type="text/javascript" id="MathJax-script" async
  src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-chtml.js">
</script>