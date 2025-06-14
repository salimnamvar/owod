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
- **Related Work**:
    - **Open Set Classification**:
        - Limitations: These methods detect unknonws but can't update themselves when new class labels are introduced.
    - **Open WOrld Classification**:
        - Limitations: Most approaches are focused on classification, not object detection and often on narrow tasks.
    - **Open Set Detection**:
        - Limitations: These models don't learn new classes over time — they just try to reject unknowns.
- **Open World Object Detection**:
  - **Formal Setup**:
    - At any time $t$, the detector knows a set of classes:
      - $K^t = \{1, 2, \ldots, C\} \subset \mathbb{N}^+$ → these are the knonw object classes.
    - There are also unknown classes, labeled:
      - $U = \{ C + 1, \ldots \}$ → not seen during training but appear during testing.
  - **Dataset Structure**:
    - The dataset at time $t$ is $D^t = \{ X^t, Y^t \}$, where:
      - $X^t$ = training images: $X^t = \{ I_1, \ldots, I_M \}$
      - $Y^t$ = labels for those images: $Y^t = \{ Y_1, \ldots, Y_M \}$
    - Each object label $y_k$ in $Y_i = \{ y_1, y_2, \ldots, y_K \}$ is $y_k = [l_k, x_k, y_k, w_k, h_k]$ and includes: 
      - $l_k$ → class label $l_k \in K^t$ (must be in $K^t$)
      - $x_k, y_k$ → center of the bounding box
      - $w_k, h_k$ → width and height of the bounding box
  - **Model Behavior**:
    - The model $M_C$ is trained on $C$ known classes.
    - At test time, it should:
      - Detect known objects (label $1 \ldots C$)
      - Detect unknown objects and assign them a special label: $0$
  - **Incremental Learning Loop**:
    - When unknown objects are detected:
        1. A human annotator reviews them and provides new class labels for $n$ new classes.
        2. These $n$ new classes are added to the model.
        3. The model updates itself to form a new model $M_{C+n}$.
        4. The knowns class set becomes $K^{t+1} = K^t + \{C + 1, \ldots, C + n\}$.
        5. This process repeats over time.
   
## Proposed Solution (ORE)

The ORE is built on top of the Faster R-CNN architecture, with three main components that work together to enable open world object detection:

1. **Contrastive Clustering**:
   - 🔍 Purpose: 
     - Ensures the latent feature space is structured in a way that:
       - ✅ Known class features are grouped (clustered) tightly.
       - 🚫 Features from unknown or different classes are separated.
   - 📌 How it works:
     - Applies contrastive learning to the features from the Region of Interest (RoI) head.
     - This helps the model:
       - Detect unknowns by seeing they don't match any existing cluster.
       - Avoid catastrophic forgetting by keeping new class features separated from previously learned ones.
  
2. **Auto-Labeling via Region Proposal Network (RPN)**:
   - 🔍 Purpose:
     - Automatically identifies unknown object instances in input images without any human labels.
   - 📌 How it works:
     - The RPN proposes objectlike regions in an image
     - If a region does not match any known class, it's pseudo-labeled as "unknown".
     - These unknown regions are used during training to help the model learn to differentiate between known vs unknown.
  
3. **Energy-Based Classification**:
    - 🔍 Purpose:
      - Decides whether a detected object belongs to:
        - ✅ A known class, or
        - 🚫 Is unknown, based on energy scores.
    - 📌 How it works:
      - Uses the concept of Helmholtz Free Energy:
        - Known instances produce lower energy (high model confidence).
        - Unknown instances produce higher energy (low confidence).
   - This head is trained to reject high-energy (unknown) samples during inference.

**🧠 Bonus: Base Detector – Faster R-CNN**
   - ORE modifies Faster R-CNN because:
     - It's a two-stage detector, more robust for open set tasks.
     - It uses:
       - A shared backbone to extract features.
       - A Region Proposal Network to propose candidate object locations.
       - A second stage head to classify and refine the bounding boxes.

### Contrastive Clustering (CC)
- **Goal of Contrastive Clustering (CC)**:
  - The core idea is to ensure that the latent (hidden) feature space of the neural network:
    - Pulls same-class object features closer together.
    - Pushes different-class object features further apart.
  - This clear separation helps the model:
    - Detect unknown/novel instances (because they won't fit into any known cluster).
    - Learn new classes incrementally without forgetting older ones.
- **Class Prototype**:
  - For each known class $i \in K^t$, ORE maintains a prototype vector $p_i$:
    - This is essentially the average feature representation of all objects from that class.
  - As the model trains and learns, the feature change, so prototypes must evolve too.
- **Contrastive Loss Function**:
  - The model computes a contrastive loss, defined as:
    $$
        L_{cont}(f_c) = \sum_{i=0}^{C} \ell(f_c, p_i)
    $$

    where:

    $$
        \ell(f_c, p_i) =
        \begin{cases}
        D(f_c, p_i), & \text{if } i = c \ (\text{same class}) \\
        \max(0, \Delta - D(f_c, p_i)), & \text{if } i \ne c \ (\text{different class})
        \end{cases}
    $$

    $D$ is a distance function (e.g., Euclidean), and $\Delta$ is a margin to control separation.

    This loss encourages:
      - Small distance for same-class features.
      - Large distance for different-class features.
- **Feature Store with Queues**:
  - To manage class prototypes efficiently:
    - A queue $q_i$ is maintained for each class, storing recent feature vectors.
    - Together, these queues form the feature store:
        $$
        F_{\text{store}} = \{ q_0, q_1, \ldots, q_C \}
        $$
    - This is scalable, since it stores at most $C \times Q$ vectors, where $Q$ is the queue size.
- **Prototype Update Algorithm (Momentum-based)**:
    - Burn-in period ($I_b$): Clustering starts after a few iterations, so feature embeddings stabilize.
    - Updating Prototypes:
      - Every $I_p$ iterations:
        - New prototypes $P_{\text{new}}$ are calculated.
        - Prototypes are updated using a momentum update:
            $$
            P \leftarrow \eta P + (1 - \eta) P_{\text{new}}
            $$
        - This prevents sudden changes, allowing gradual learning.
    - The contrastive loss is combined with standard detection loss and backpropagated to train the entire model.

### Auto-Labelling Unknowns with RPN (ALU)
- **Problem: Need Labels for Unknowns**:
  - To use contrastive clustering, the model needs to compare input features ($f_c$) against prototypes, including for the unknown class (labeled as $0$).
  - But real datasets don't label unknown objects, because:
    - It's impractical to manually re-label all images.
    - Annotating every unknown object is too time-consuming and expensive.
- **Solution: Auto-labelling via RPN (Region Proposal Network)**:
  - RPN is class-agnostic: it generates region proposals (bounding boxes) without caring which class the object belongs to.
  - For each image, RPN outputs:
    - Boxes likely to contain foreground objects.
    - An objectness score indicating how confident it is that the region has an object (not just background).
- **Pseudo-labelling Unknowns**:
  - They use a simple yet effective heuristic:
    1. Get region proposals from RPN.
    2. Filter proposals that:
       1. Have high objectness scores (i.e., likely to be objects).
       2. Do not overlap with any ground truth box (i.e., not known objects).
    3. Select the top-k such proposals.
    4. Label them as unknown class (label $0$).
   - This way, the model automatically identifies potential unknowns and feeds them into contrastive clustering for learning.
 - **Why It Works**:
   - This simple heuristic gives surprisingly good performance.
   - It enables the model to learn about unknowns without any manual labeling.

