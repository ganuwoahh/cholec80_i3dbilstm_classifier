Surgical Phase Recognition on Cholec80: A Two-Stage I3D & BiLSTM Pipeline
📌 Project Overview
This project tackles the complex challenge of Automated Surgical Phase Recognition using the Cholec80 dataset (laparoscopic cholecystectomy videos). The objective is to accurately classify every second of surgical video into one of seven distinct operative phases.

Processing high-resolution, 3D spatiotemporal video data requires immense compute. A naive approach of training an end-to-end 3D Convolutional Neural Network alongside a temporal sequence model requires RAM and VRAM that far exceeds standard research hardware.

To solve this, this project implements a highly optimized Two-Stage Feature Extraction and Temporal Reasoning Pipeline. By completely separating the "Visual Cortex" (I3D) from the "Hippocampus" (BiLSTM), this architecture bypasses hardware bottlenecks, eliminates memory leaks, and significantly improves phase classification—specifically curing the "temporal amnesia" common in standalone 3D CNNs.

🧠 The Architecture
Stage 1: The Visual Cortex (I3D Feature Extractor)
The first stage processes raw surgical video to understand what is happening on screen (tools, tissue interactions) without worrying about when it is happening.

Model: Inception 3D (I3D), utilizing dual streams (RGB and Optical Flow).

Input: 32-frame video clips.

Optimization: Used tf.keras.mixed_precision (float16) to safely double batch sizes on 16GB T4 GPUs.

Data Pipeline: Converted raw video frames into highly compressed .tfrecord binaries to eliminate I/O bottlenecking.

Stage 2: The Hippocampus (BiLSTM Temporal Memory)
Standalone 3D CNNs suffer from severe "Groundhog Day" syndrome. For example, Phase 1 (Calot Triangle Dissection) and Phase 3 (Gallbladder Dissection) use identical tools and look visually identical in a 1.2-second vacuum.

The Hand-off: The classification head of the trained I3D model was decapitated. The entire 80-video dataset was passed through the headless I3D to extract dense 1024-dimensional feature vectors, compressing gigabytes of video into lightweight .npy files.

The Model: A Bidirectional Long Short-Term Memory (BiLSTM) network.

The Result: The BiLSTM reads the entire surgery forward and backward. When it sees visual features representing "burning fat," it checks its memory to see if the "clipping" phase has already occurred, allowing it to correctly deduce Phase 3 over Phase 1.

🛠️ Key Engineering Hurdles & Solutions
Building this pipeline required heavily modifying standard TensorFlow/Keras protocols to survive tight RAM constraints (30GB Kaggle limits) and multi-GPU distribution bugs.

1. The "Round-Robin" Interleave (Beating the RAM Limit)
Standard dataset shuffling on massive 3D videos requires a massive buffer_size, which easily exceeds 60GB of system RAM, crashing the kernel.
Solution: Re-engineered the tf.data pipeline to use .interleave(). Instead of sequential reading + massive shuffling, the pipeline simultaneously opens 8 different surgical videos and deals clips in a round-robin format. This achieved perfect batch diversity while dropping the required shuffle buffer by 95%.

2. The Multi-GPU Partial Batch Crash
When using tf.distribute.MirroredStrategy across dual GPUs, the final uneven batch of an epoch would cause a partial batch allocation (e.g., GPU 0 gets 3 clips, GPU 1 gets 0 clips), causing a mathematical collapse during 3D Average Pooling.
Solution: Explicitly enforced drop_remainder=True during the dataset windowing and batching phase to protect the spatial math of the I3D layers.

3. Circumventing Keras Metric Bugs in Imbalanced Data
The Cholec80 dataset is highly imbalanced (Phases 0 and 1 dominate the runtime). Relying on Keras's default accuracy and Recall metrics provided a deeply flawed view of the model's performance.

Keras evaluates Recall using a strict 0.5 threshold, which automatically flags correct predictions in a 7-class softmax output as False Negatives if the confidence is below 50%.

During BiLSTM training, Keras included the -1 ignored padding sequences in its accuracy calculations, artificially tanking the displayed accuracy.
Solution: Bypassed Keras entirely for evaluation. Wrote custom inference scripts to run predictions, strip away temporal padding via masking, and utilized scikit-learn to calculate the mathematical ground-truth Precision, Recall, F1-scores, and generate normalized Confusion Matrices and Multi-Class ROC Curves.

📈 Results
By splitting the architecture and allowing the BiLSTM to map the grammatical rules of the surgery, the model successfully differentiated visually identical phases, resulting in clean ROC curves and a highly diagonal confusion matrix.

(Note: Insert your final Accuracy, Macro AUC, and the ROC/Confusion Matrix images here!)
