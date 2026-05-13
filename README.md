# LW5: Comparative Analysis of Pre-trained CNN Models for Custom Image Classification

---

## 🔗 Google Colab Notebook

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1144ToM9A7cfPHLfNQPOJu9n1L9hngUrc?authuser=1#scrollTo=ICrK5hxzT7FM)

**Direct Link:** https://colab.research.google.com/drive/1144ToM9A7cfPHLfNQPOJu9n1L9hngUrc?authuser=1#scrollTo=ICrK5hxzT7FM

---

## 📊 PART 12: Model Performance Comparison Table

> All pre-trained models trained with frozen ImageNet weights + custom classification head.  
> Dataset: 80% train / 20% validation split, 224×224 input size, 32 batch size.  
> EfficientNetB0 was retrained with corrected preprocessing (no Rescaling layer).

| Model | Train Acc | Train Loss | Test Acc | Test Loss | Precision | Recall | F1-score | ROC AUC |
|---|---|---|---|---|---|---|---|---|
| **VGG16** | 65.78% | 1.1241 | 80.10% | 0.7540 | 0.7956 | 0.7936 | 0.7916 | 0.9822 |
| **ResNet50** | 92.28% | 0.2830 | 93.70% | 0.2425 | 0.9353 | 0.9355 | 0.9347 | 0.9975 |
| **EfficientNetB0** ⚠️ | 5.65% | 2.9956 | 5.80% | 2.9954 | 0.0029 | 0.0500 | 0.0055 | 0.5807 |
| **Teachable Machine** | ~98.00% | ~0.0200 | 76.30%* | ~0.3600 | 0.7614 | 0.7636 | 0.7614 | 0.8981 |
| Your 1st Model (LW3 Baseline) | 98.28% | 0.0616 | 77.30% | 0.6922 | 0.7825 | 0.7719 | 0.7629 | 0.9865 |
| Your 2nd Model (LW3 Enhanced) | 75.91% | 0.6186 | 73.40% | 0.8102 | 0.7358 | 0.7326 | 0.7249 | 0.9816 |
| Your 3rd Model — The Good Model (LW4) | 92.75% | 0.2157 | 88.00% | 0.8264 | ~0.878 | ~0.875 | ~0.872 | ~0.990 |

> ⚠️ *EfficientNetB0 achieved only 5.80% accuracy due to a preprocessing incompatibility (double-scaling bug). See Guide Question 8 and 12 for full explanation.*
> 📝 *Teachable Machine Train Acc/Loss are approximate values from TM's own training graph (50 epochs). `*` Test Acc = 76.30% is from Colab re-evaluation on the same val set as other models (TM's internal interface reported 92.37% on its own test split). `~` values are visual estimates from TM graphs.*



## ❓ GUIDE QUESTIONS (FINAL REFLECTION)

### A. Model Performance

**1. Which pre-trained model achieved the highest accuracy? Why?**  
I found that ResNet50 achieved the highest accuracy at 93.70%. This was likely due to its residual connections, which allowed it to learn deeper, more complex features from the plant dataset without suffering from vanishing gradients during my fine-tuning process.

**2. Which model had the lowest performance? What could be the reason?**  
My EfficientNetB0 model had the lowest performance, achieving only 5.80% accuracy (which is essentially random guessing for 20 classes). I discovered this was caused by a preprocessing error: I had applied a `Rescaling(1./255)` layer to the dataset, but EfficientNetB0 already has built-in normalization and expects raw [0, 255] pixels, causing it to receive near-zero inputs.

**3. How did loss values compare across models?**  
I observed that ResNet50 had the lowest test loss (0.2425), aligning with its high accuracy. VGG16 had a higher test loss (0.7540), showing less confidence in its predictions. EfficientNetB0 had the highest test loss (2.9954), which is close to the theoretical maximum for random guessing in a 20-class problem, reflecting its failure to learn.

---

### B. Evaluation Metrics

**4. Why is accuracy not enough to evaluate a model?**  
I realized that accuracy alone can be misleading. While my dataset had roughly 200 images per class, metrics like Precision, Recall, and F1-score helped me ensure my model wasn't just predicting the majority class or performing well overall while failing completely on specific, difficult plant species.

**5. Which model had the best F1-score? What does it indicate?**  
ResNet50 had the best F1-score of 0.9347 in my tests. This indicated to me that the model maintained an excellent balance between precision (not falsely identifying a plant) and recall (successfully identifying all instances of a specific plant) across all 20 classes.

**6. How did Precision and Recall differ across models?**  
I noticed that for ResNet50, precision (0.9353) and recall (0.9355) were very tight and consistent. For VGG16, precision (0.7956) and recall (0.7936) were lower but still balanced. EfficientNetB0's precision (0.0029) and recall (0.0500) completely collapsed due to the preprocessing bug.

---

### C. Confusion Matrix Analysis

**7. Which classes were frequently misclassified?**  
While analyzing my confusion matrices, I observed that visually similar plant species were the ones most frequently confused. For example, VGG16 struggled more with classes that had similar leaf shapes or petal colors. ResNet50 cleared up most of these misclassifications.

**8. What patterns did you observe in the confusion matrix?**  
In my ResNet50 matrix, I saw a strong diagonal line with very few off-diagonal activations, confirming high confidence across almost all classes. In contrast, my EfficientNetB0 matrix showed a single vertical band where the model predicted the exact same class for every single test image, confirming it had learned nothing.

---

### D. ROC and AUC

**9. Which model had the highest AUC score?**  
My ResNet50 model achieved the highest Mean AUC score of 0.9975.

**10. What does AUC tell us about model performance?**  
The AUC score of 0.9975 told me that my ResNet50 model has near-perfect class separation ability. It means that if I present the model with a positive and a negative example for any plant class, there is a 99.75% chance it will correctly rank the positive example higher than the negative one, regardless of the classification threshold.

---

### E. Explainability (Grad-CAM)

**11. What did Grad-CAM reveal about model decision-making?**  
Grad-CAM heatmaps showed me exactly where the models were "looking". I saw that poorly performing models often highlighted random background areas, whereas my best model focused sharply on the plant itself.

**12. Did the model focus on relevant image regions?**  
Yes, for my best models (like ResNet50), I observed that the heatmaps were tightly clustered around relevant botanical features, such as the distinct edges of the leaves, the center of the flowers, and specific stem patterns, rather than the dirt or background.

**13. Which model produced the most meaningful heatmaps?**  
ResNet50 produced the most meaningful and precise heatmaps for me. VGG16's heatmaps were much more diffuse and spread out across the image, while ResNet50 pinpointed the exact distinct features of the plants.

---

### F. Model Comparison & Improvement

**14. Which model would you recommend for deployment? Why?**  
I would recommend ResNet50 for deployment. In my comparative analysis, it provided the best absolute performance (93.70% test accuracy) and feature extraction capability, making it the most reliable choice for a robust plant classification system.

**15. How can you further improve your best-performing model?**  
To further improve my ResNet50 model, I could unfreeze the top few layers of the base model and gently fine-tune them with a very low learning rate. I could also apply more aggressive data augmentation (like color jitter or random crops) to make it even more robust against different lighting conditions.

---

### G. Real-World Application

**16. How can your model be applied in real-world scenarios?**  
I could deploy my model into agricultural monitoring systems, where drones or cameras in greenhouses automatically identify plant species or even detect early signs of plant diseases based on visual features.

**17. What are the risks of deploying an inaccurate model?**  
If I deployed an inaccurate model like my broken EfficientNetB0, it could misidentify poisonous plants as safe, or misclassify crops in an automated sorting facility, leading to severe safety risks or economic losses.

**18. How can this system be integrated into a mobile/web app?**  
I can convert my trained `.keras` model into TensorFlow Lite format. This would allow me to embed the model directly into a smartphone application (like Android/iOS) so users could point their camera at a plant and get real-time identification without needing an internet connection.
