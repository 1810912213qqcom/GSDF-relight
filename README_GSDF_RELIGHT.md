# GSDF-Relight Project Documentation

## 1. Overview
**GSDF-Relight** is a project that integrates Signed Distance Fields (SDF) into the **gs-relight** (3D Gaussian Splatting for Relighting) framework. The goal is to leverage the geometric consistency of SDFs to improve the underlying geometry of the 3D Gaussian representation, which is critical for high-quality relighting tasks.

This project is built upon:
*   **gs-relight**: The base codebase for 3DGS relighting.
*   **GSDF / Instant-NSR**: The source for the SDF implementation (`VolumeSDF_gaussian`).

## 2. Directory Structure
*   `GSDF-relight/`: Main project directory.
    *   `train.py`: Main training script (Modified).
    *   `gaussian_renderer/__init__.py`: Renderer implementation (Modified).
    *   `instant_nsr/`: SDF model definitions (Imported from GSDF).
    *   `scene/`: 3DGS scene and model definitions (from gs-relight).

## 3. Key Modifications

### 3.1. Renderer (`gaussian_renderer/__init__.py`)
We modified the `render` function to support auxiliary geometric outputs required for SDF supervision:
*   **Depth Rendering (`out_depth=True`)**: Rasterizes the depth of 3D Gaussians to produce a depth map (`depth_hand`).
*   **Normal Rendering (`return_normal=True`)**: Computes normals from the Gaussian rotation matrices (using the shortest axis as the normal direction) and rasterizes them to produce a normal map (`gs_normal`).

### 3.2. Training Script (`train.py`)
We integrated the SDF model initialization and training loop directly into the main 3DGS training process.

#### **Initialization**
*   Imported `VolumeSDF_gaussian` from `instant_nsr`.
*   Added `get_sdf_config()` to configure the SDF model (HashGrid encoding, MLP decoder).
*   Initialized `sdf_model` and its optimizer `sdf_optimizer`.

#### **Training Loop Integration**
Inside the training loop (after the main 3DGS rendering and loss calculation):
1.  **Geometric Extraction**: We extract the **Depth** and **Normal** maps from the current 3DGS render.
2.  **SDF Sampling**:
    *   We identify visible Gaussians using the `visibility_filter`.
    *   We randomly sample a subset of these visible Gaussian centers (`sampled_xyz`).
3.  **SDF Loss Calculation**:
    *   **Zero-Level Set Loss**: We query the SDF value at `sampled_xyz`. Since these points lie on the visible surface represented by Gaussians, we enforce `SDF(sampled_xyz) ≈ 0`.
    *   **Eikonal Loss**: We compute the gradient of the SDF at these points and enforce the Eikonal constraint (`||∇SDF|| ≈ 1`) to ensure valid signed distance field properties.
4.  **Optimization**: We backpropagate the SDF losses and step the `sdf_optimizer`.

## 4. Data Flow & Interaction

### **Forward Pass**
1.  **3DGS Model**: Takes Viewpoint Camera -> Renders **RGB Image**, **Depth Map**, **Normal Map**.
2.  **SDF Model**: Takes Sampled 3D Points (from Gaussian centers) -> Predicts **SDF Value**, **SDF Gradient**.

### **Loss Calculation**
*   **Relighting Loss (3DGS)**:
    *   `L1 Loss(Rendered RGB, GT RGB)`
    *   `SSIM Loss(Rendered RGB, GT RGB)`
    *   *Gradients update 3DGS parameters (Position, Rotation, Opacity, SH, Material).*
*   **Geometry Loss (SDF)**:
    *   `L1 Loss(SDF(Gaussian Centers), 0)`: Constrains SDF to be zero at Gaussian locations.
    *   `MSE Loss(||∇SDF||, 1)`: Regularizes the SDF field.
    *   *Gradients update SDF parameters (HashGrid, MLP).*

### **Interaction**
Currently, the interaction is **unidirectional (3DGS -> SDF)** for supervision:
*   The 3DGS geometry (Gaussian centers) acts as "ground truth" surface points to supervise the SDF.
*   This forces the SDF to learn a continuous implicit surface that fits the discrete Gaussian cloud.
*   *Future Work*: This can be made bidirectional by using the SDF normals to regularize the 3DGS normals (`local_q`) or by using the SDF to guide Gaussian densification/pruning (as in the original GSDF paper).

## 5. How to Run
Simply run the training script as you would for gs-relight. The SDF training happens automatically in the background.

```bash
python train.py -s <path_to_dataset> --model_path <path_to_output>
```



