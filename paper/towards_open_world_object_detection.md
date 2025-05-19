# Towards Open World Object Detection

## Research Information
**Paper**: https://arxiv.org/abs/2103.02603

**Github**: https://github.com/JosephKJ/OWOD

**Year**: *2021*

**Authors**: *K J Joseph, Salman Khan, Fahad Shahbaz Khan, Vineeth N Balasubramanian*

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

