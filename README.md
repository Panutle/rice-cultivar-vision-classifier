# Smart Grain Phenotyping: Multi-Cultivar Rice Grain Inspection & Classification via Deep CNN

An automated Computer Vision and Deep Learning framework engineered for non-destructive agricultural grain quality inspection and phenotypic analysis. The system detects, segments, and normalizes individual rice grains distributed across an inspection surface and classifies distinct cultivars (`BN`, `HR`, `WG`) with color-coded visual bounding overlays.

## 📌 Demonstration & Pipeline Output


*Figure 1: Automated grain contour extraction, perspective alignment, and multi-class cultivar identification visualized directly on the source frame.*

## ⚙️ Architectural & Pipeline Design

```
flowchart TD
    subgraph Preprocessing["1. Morphological Grain Isolation"]
        A[Raw Multi-Grain Image] --> B[Grayscale & Binary Thresholding: Val 70]
        B --> C[Morphological Erode / Dilate: 6x6 Kernel]
        C --> D[Contour Detection & Area Filtering: 2k - 20k px]
    end

    subgraph Alignment["2. Geometric & Color Normalization"]
        D --> E[Oriented Bounding Box: minAreaRect + 20% Padding]
        E --> F[Perspective Transform & Major-Axis Rotation]
        F --> G[HSV_FULL Color-Space Conversion: 300x150x3]
    end

    subgraph DeepCNN["3. Deep Convolutional Classification"]
        G --> H[Cascaded Conv2D: 400, 600, 800, 1600 Filters]
        H --> I[Deep Dense Classifier: 4096 to 128 Units + Dropout]
        I --> J[Softmax Multi-Class Probability: BN, HR, WG]
    end

    subgraph VisualOverlay["4. Field Reporting & Visual Analytics"]
        J --> K[Color-Coded Contour Overlay on Source Frame]
        J --> L[Automated Confusion Matrix & Loss Plots]
        J --> M[Real-time Training Telemetry via ntfy.sh Webhook]
    end

```

## 🔬 Core Technical Highlights

### 1. Invariant Morphological Segmentation & Spatial Normalization

* **Noise Rejection**: Applies calibrated binary thresholding ($\text{threshold}=70$) followed by dual-pass erosion and dilation using a $6 \times 6$ structural element to sever touching grain boundaries and suppress surface dust.

* **Size-Constrained Area Filtering**: Restricts extraction to contours within $2,000 \le \text{Area} \le 20,000$ pixels, filtering out unhulled debris and clustered aggregates.

* **Perspective Rectification**: Computes minimum bounding rectangles (`cv2.minAreaRect`), applies dynamic 20% padding expansion, and rectifies perspective distortion via Homography matrices (`cv2.getPerspectiveTransform`). All segmented grains are standardized to a vertical orientation through automated 90°/180° rotations.

### 2. Chromatic Representation (HSV Color Space)

* Segmented grains are projected into the **HSV_FULL** color spectrum prior to neural feature extraction.

* Isolates chromatic characteristics (pigmentation, translucency, and chalkiness) from luminance shifts and ambient shadows cast under industrial top-down lighting.

### 3. High-Capacity Deep CNN Architecture

* **Backbone Feature Extractor**: 4-stage convolutional hierarchy scaling from 400 to 1,600 filters with $4 \times 4$ kernel sizes and `BatchNormalization` layers.

* **High-Dimensional Latent Projections**: Multi-layer Dense head structured across 4096, 2048, 1024, 512, 256, and 128 units, regulated by staged Dropout (0.5 to 0.2) to prevent feature co-adaptation.

* **Convergence Strategy**: Adam optimizer parameterized with an ultra-fine learning rate ($\eta = 10^{-7}$) and early stopping mechanisms (`patience = 300`, restoring optimal validation loss checkpoints).

### 4. MLOps Observability & Automated Reporting

* Integrates automated evaluation hooks saving high-resolution Confusion Matrices and training Loss curves.

* Dispatches real-time training telemetry payloads (accuracy, total parameters, elapsed execution time) to remote operations endpoints using Webhook APIs (`ntfy.sh`).

## 📂 Repository Layout

```
rice-cultivar-vision-classifier/
├── README.md
├── requirements.txt
├── .gitignore
├── docs/
│   └── demo_sample.png            # Visual classification sample
├── src/
│   ├── inference_pipeline.py      # Grain segmentation, perspective alignment & visual overlay
│   └── train_cnn.py               # Deep CNN architecture, training loops & evaluation
└── models/
    └── .gitkeep                   # Target directory for exported .keras model weights

```

## 📊 Target Cultivars

| **Class Label** | **Cultivar Identifier** | **Dominant Morphology & Visual Profile** | 
| **BN** | Cultivar 1 (Blue Overlay) | Extra-long slender profile, high translucency | 
| **HR** | Cultivar 2 (Green Overlay) | Medium slender grain, distinct curvature | 
| **WG** | Cultivar 3 (Red Overlay) | Short rounded bold grain, opaque endosperm | 

## 🚀 Setup & Execution

### 1. Environment Installation

```
git clone https://github.com/Panutle/rice-cultivar-vision-classifier.git
cd rice-cultivar-vision-classifier
pip install -r requirements.txt

```

### 2. Model Training & Evaluation

```
python src/train_cnn.py

```

*(Select the source image training directory via the GUI file picker when prompted.)*

### 3. Bulk Detection & Multi-Cultivar Inference

```
python src/inference_pipeline.py

```

*(Select the inspection image and pre-trained `.keras` model via the GUI dialog.)*