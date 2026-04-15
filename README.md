# Surgical Phase Recognition on Cholec80: A Two-Stage I3D & BiLSTM Pipeline

## 1. The Cholec80 Dataset
Cholec80 is a dataset consisting of 80 laparoscopic cholecystectomy surgeries of varying lengths. The original videos have a resolution of 854 x 480 at 25 FPS. Every single frame in the dataset is annotated as one of 7 distinct surgical phases:
1. Preparation
2. Calot Triangle Dissection
3. Clipping and Cutting
4. Gallbladder Dissection
5. Gallbladder Packaging
6. Cleaning and Coagulation
7. Gallbladder Retrieval

*(Note: I didn't realise that retrieval and retraction weren't the same. So if you see retraction anywhere it's supposed to be retrieval)*

<img width="1879" height="352" alt="image" src="https://github.com/user-attachments/assets/e134312b-6908-4d98-87dd-e411ee798e51" />

To make this computationally feasible, the videos were downsampled to 1 FPS. 

**Data Structure:**
The dataset contains 3 main folders: `frames`, `phase_annotations`, and `tool_annotations` (tool annotations were not used in this project). Because the original framerate was 25 FPS, for every frame in the `frames` folder, there are 25 corresponding phase annotations in the text files. 

**Sources:**
* Downloaded originally from the Endonet Paper: [cholec80.tar.gz](https://s3.unistra.fr/camma_public/datasets/cholec80/cholec80.tar.gz)
* Publicly available on Kaggle: [Cholec80 Dataset](https://www.kaggle.com/datasets/ganumatta/cholec80)
* *FYI: Do not use the `prepare.py` file found in the original Endonet repo. It's heavily bugged.*

---

## 2. The Model: I3D & Optical Flow
For the base computer vision model, I cloned a Keras I3D implementation from [dlpbc/keras-kinetics-i3d](https://github.com/dlpbc/keras-kinetics-i3d) and decapitated the original classification layer: [I3D No Top Layer](https://www.kaggle.com/datasets/ganumatta/i3d-no-top-3).

I3D utilizes a Two-Stream architecture: an **RGB stream** (standard video) and an **Optical Flow stream** (a mathematical calculation of individual pixel movement from frame to frame).

This is frame 1017 from video1: <img width="854" height="480" alt="video01_001017" src="https://github.com/user-attachments/assets/13ae9f17-9268-41dd-a64b-2c01861fc3fd" />

This is frame 1018 from video2: <img width="854" height="480" alt="video01_001018" src="https://github.com/user-attachments/assets/e55b17c8-a3ac-4007-bc21-cf591a24d5cb" />

This is frame 1018's optical flow in video1: <img width="854" height="480" alt="video01_001018" src="https://github.com/user-attachments/assets/d6929d96-98f0-46f3-a92a-440ad6bd2b2e" />

### Generating the Optical Flow
* **Dataset Link:** [Cholec80 Optical Flow](https://www.kaggle.com/datasets/ganumatta/cholec80-optical-flow)
* **Methodology:** A pre-trained RAFT (Recurrent All-Pairs Field Transforms) model with "small" weights was used to calculate the flow. Because the video is downsampled to 1 FPS, the frame-to-frame movements are massive and choppy. Older algorithms fail here. RAFT uses a GRU that maps each pixel from one frame to another as a 4D correlation matrix and calculates the difference between those pairs. It loops 12 times, and I took the 12th/final prediction as the output for that frame pair. For a video that is *n* frames long, you generate *n-1* optical flow images. To format the Optical Flow so the I3D could read it like RGB, the raw motion vectors were clamped with a bound of `-15` to `15`, shifted to `0` to `30` and then normalized to an integer scale of `0` to `255`.
* *Note: Because OpenCV saves images in BGR format instead of RGB, the resulting optical flow colors are visually inverted.*
* *_Note: The quality of the uploaded dataset is only at 85%. Please use atleast 95 if you can__

---

## 3. Hardware Issues
With both the RGB and Optical Flow datastreams ready, fine-tuning the I3D should have been straightforward. Instead, it became a massive hardware battle against the Kaggle VM limits.

**Bottleneck:** Optical Flow was initially saved as a collection of `.jpeg` files to stay under Kaggle's input limits. However, unpacking these JPEGs dynamically during training heavily throttled the CPU, creating a massive bottleneck.

**RAM issue:** To actually learn anything, the 3D dataset needed to be shuffled. Attempting to load this shuffled data into Kaggle's 30GB of RAM instantly crashed the kernel.

**Solution:**
To survive the hardware limits, every video was repackaged into a `.tfrecord`. This format acts as a single, highly compressed bytestream. For each frame in a video, the TFRecord tightly packages:
1. The RGB frame
2. The corresponding Optical Flow frame *(left empty for the very first frame)*
3. The Phase Annotation

This is saved as one continuous file per video which is very easy to parse during training as long as you provide TensorFlow with the correct mapping blueprint.

---

## 4. Training the I3D 
Using a Kaggle notebook with the pre-trained I3D and the new TFRecord dataset, the pipeline was built as follows:
1. A function to parse the `.tfrecord` byte-strings back into tensors.
2. A sliding window function to stack the parsed frames on top of each other to create overlapping 3D video clips.
3. A builder function to compile the training and validation datasets.

It required some serious duct-tape engineering (including a round-robin interleave to bypass the shuffle buffer memory explosion) to keep the system stable, but a custom classification layer was slapped on top, and the model was successfully fine-tuned.

<img width="1521" height="982" alt="success2" src="https://github.com/user-attachments/assets/0156e45d-6e2c-4504-9838-4d1a69e49290" />
<img width="1241" height="967" alt="success1" src="https://github.com/user-attachments/assets/21d43e3c-4d5b-4bd3-a2cc-8ff263c711bc" />
<img width="677" height="392" alt="success3" src="https://github.com/user-attachments/assets/2b2cb494-84b1-4d7b-8044-b698aa8004f9" />

---

## 5. BiLSTM Classification Layer
You can see that the model doesn't understand the difference between stage 1 and stage 3. This is because the stages are visually similar. The only real difference is that one stage happens before clipping and cutting and the other happens after. But since the model wasn't trained on long enough sequences due to the hardware issues, it doesn't know that so it predicts them all as gallbladder dissection since it's the most common class.

The solution is a BiLSTM layer to replace the softmax classification layer. This ideally would have been trained together with the I3D but because of hardware limitations (again), I trained them separately.

**The Feature Extraction:**:
1. The fully trained I3D model was run over the entire dataset again in a pure forward pass.
3. The top classification layers were chopped off, exposing the network's internal 1024-Dimensional feature vector.
4. These feature vectors were stacked chronologically for an entire video, resulting in a tiny 2D matrix representing the whole surgery. These matrices were saved to disk alongside their label vectors.

A 2-layer Bidirectional LSTM (BiLSTM) was then trained exclusively on these `.npy` feature matrices. 

---

## 6. Evaluation Metrics
To ensure the pipeline was mathematically sound and to prove the model wasn't hallucinating its predictions, the final evaluation relied heavily on:
* **AUC (Area Under the ROC Curve):** To accurately measure performance and bypass Keras softmax thresholding bugs in highly imbalanced classes.
* **Correlation Matrices**
* **Grad-CAM++:** To visually map the spatial activations and prove the network was looking at the physical tools and tissue interactions, rather than just memorizing static background noise.

<img width="1541" height="995" alt="success6" src="https://github.com/user-attachments/assets/99ea10cb-1a69-41ca-928d-d5555c4881bb" />
<img width="1263" height="997" alt="success5" src="https://github.com/user-attachments/assets/0627ee61-a49f-4928-88f4-7f9f3b9b29e2" />
<img width="643" height="382" alt="success4" src="https://github.com/user-attachments/assets/e619ad16-34fa-4323-8285-1bd75d8e9467" />
<img width="1778" height="607" alt="success_gradcam" src="https://github.com/user-attachments/assets/6ecc83ec-d4c3-4b44-925f-225b401f2efc" />

## 7. Future Improvements

* Obviously the big one is not using the tool annotations. It would solve the problem of the current BiLSTM not predicting clipping and cutting at all.
* Using a Transformer over a BiLSTM
* Train the I3D and Classification Head together (lol)
* Optical Flow would benefit from 3FPS footage instead
* Don't use JPEGs
* Use Focal Loss


**This readme is mostly AI** 
