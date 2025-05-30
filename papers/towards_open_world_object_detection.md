# Towards Open World Object Detection

## Research Information
- **Paper**: https://arxiv.org/abs/2103.02603
- **Github**: https://github.com/JosephKJ/OWOD
- **Year**: *2021*
- **Authors**: *K J Joseph, Salman Khan, Fahad Shahbaz Khan, Vineeth N Balasubramanian*

## Notes

## Mindmap
- **Problem Definition**:
    - **Identify Unknowns**:
        - Without Explicit Supervision
    - **Incremental Learning**:
        - Identified Unknown Categories
        - Without Forgetting Learned Classes
    - **Motivation**:
        - Human Instinct (Unknown Identification)
        - Intrinsic Curiosity
    - **Distinction from Related Problems**:
        - Open Set Classification
        - Open World Classification
        - Open Set Detection
- **Proposed Solution**:
    - **ORE Components**:
        - **Contrastive Clustering (CC)**:
            - Latent Space Separation
            - Identifies Unknowns (Novelty)
            - Facilitates Incremental Learning (No Forgetting)
            - Class Prototype
            - Feature Store (Queues)
            - Contrastive Loss
        - **Auto-Labelling Unknowns (ALU)**:
            - Addresses Infeasible Manual Annotation
            - Uses Class-agnostic RPN
            - Pseudo-labels Unknowns
            - Top-k Background Proposals (High Objectness)
        - **Energy-Based Unknown Identifier (EBUI)**:
            - Based on Energy Models (EBMs)
            - Assigns Low Energy to In-distribution Data
            - Helmholtz Free Energy
            - Transforms Classification Head to Energy Function
            - Separation of Known/Unknown Energy Levels
            - Uses Shifted Weibull Distributions
    - **Base Detector**:
        - Faster R-CNN
        - ResNet-50 Backbone
    - **Alleviating Forgetting**:
        - Catastrophic Forgetting Issue
        - Uses Exemplar Reply (Balanced Finetuning)
        - Stores Balanced Set of Exemplars
- **Evaluation**
    - **Evaluation Protocol**:
        - Tasks (Incremental Class Introduction)
        - Data Split (Pascal VOC, MS-COCO)
        - Training/Test Sets
        - Validation Set
    - **Evaluation Metrics**:
        - **Wilderness Impact (WI)**:
            - Characterizes Confusion (Unknown as Known)
        - **Absolute Open-Set Error (A-OSE)**:
            - Count of Unknowns Misclassified as Known
        - **Mean Average Precision (mAP)**:
            - Incremental Learning Capability
    - **Results**:
        - Outperforms Faster R-CNN Baseline
        - Lower WI and A-OSE
        - Better Incremental Learning Performance
        - State-of-the-art on Incremental Object Detection
- **Analysis & Discussion**:
    - **Ablation Studies**:
        - Contribution of CC, ALU, EBUI
    - **Sensitivity Analysis**:
        - Exemplar Memory Size (Nex)
        - Queue Size of FStore
        - Momentum Parameter (η)
        - Clustering Loss Margin (Δ)
        - Temperature Parameter (T)
    - **Comparison with Open Set Detector**:
        - Lower Performance Drop (ORE vs. Miller et al.)
    - **Clustering Loss and t-SNE Visualization**:
        - Distinct Clusters
        - Unknown Instances Clustered
        - Loss Convergence
    - **Time and Storage Expense**:
        - Additional Training/Inference Time
        - Negligible FScore Storage
        - Exemplar Memory Storage (34 MB for Nex=50)
    - **Failure Cases**:
        - Occlusions and Crowding
        - Difficult Viewpoints
        - Small Unknown Objects with Large Known Objects
- **Conclusion**:
    - Novel Problem Setting Introduced
    - ORE Addresses Challenges
    - Energy-Based Classifier
    - Contrastive Clustering
    - Future Research Direction


## Problem Definition
- **Current Object Detection Limitation**:
    - Assumes all classes are known at training time.
    - But in the real world, you encounter unknown objects.
- **Two Open-World Challenges**:
    1. 🆘 Detect unknowns at test time and label them as "unknown".
    2. ♻️ Incrementally learn new classes once labeled data becomes available without forgetting learned classes.
- **Motivation**:
    - Human curiosity and learning involve noticing what we don’t know and then learning it.
    - Datasets like Pascal VOC and COCO only cover a small subset of the vast real-world object categories.
- **Prior Work is Limited**:
    - Open Set and Open World classification has been explored.
    - Open World object detection is still unexplored and more challenging.
- **Why Object Detection Is Harder**:
    - Detectors treat unknown objects as background.
    - So they can mistakenly classify unknowns as known classes.
    - Existing detectors often make overconfident mistakes.
- **This Paper's Contribution**:
    - Proposes the Open World Object Detection (OWOD) problem formally.
    - **Introduces a new method called ORE, combining**:
        - 📌 Contrastive clustering
        - 📦 Unknown-aware proposal network
        - 🔋 Energy-based unknown detection
    - Provides new benchmarks and settings for open-world evaluation.
    - Achieves state-of-the-art results on incremental learning as a side effect.

## Proposed Solution (ORE)

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
