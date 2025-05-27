# Towards Open World Object Detection

## Research Information
- **Paper**: https://arxiv.org/abs/2103.02603
- **Github**: https://github.com/JosephKJ/OWOD
- **Year**: *2021*
- **Authors**: *K J Joseph, Salman Khan, Fahad Shahbaz Khan, Vineeth N Balasubramanian*

## Notes

### Introduction
- Open World Object Detection: 
    1. A test image might contain objects from unknown classes, which should be classified as unknown. 
    2. As and when information (labels) about such identified un-
knowns become available, the model should be able to in-
crementally learn the new class.
- Open World object detection is unexplored, owing to the difficulty of the problem setting.
- Challenge: The object detector is trained to detect unknown objects as background.
- Identify unknown instances without forgetting of earlier instances without retraining from scratch.

### ORE

- Initialize Model with Known Classes
- Region Proposal and Feature Extraction
    - Region Proposal Network (RPN) 
- Contrastive Clustering of Latent Features
- Auto-Labelling Unknown Instances
- Energy-Based Unknown Identification

### Contrastive Clustering
- **Goal**: Improve the **separation of different object classes** in feature space so that unknown objects (not seen during training) can be identified more easily.
- **How?** By using a **contrastive clustering** loss:
    - Pull features of the **same class** close together.
    - Push features of **different classes** far apart.
- **Each class has a `prototype vector (p_i)`**:
    - This represents the average position (mean) of that class in the feature space.
    - These prototypes are **updated gradually** as the network trains.
- **Loss Function (`L_cont`)**:
    - Encourages features to move **closer to the correct class prototype**.
    - Encourages features to be **far from prototypes of other classes**.
- **Feature Queue**:
    - Keeps a limited **rolling memory** of past features for each class to calculate updated prototypes over time.
- **Training Logic**:
    - The clustering loss is only calculated **after a “burn-in” period** so early noisy features don’t confuse the system.
    - Prototypes are updated **every few steps** using a **momentum-based update**.
