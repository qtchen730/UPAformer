# UPAFormer

PyTorch implementation of **UPAFormer: A Universal Perception and Aggregation Transformer for Heterogeneous Multimodal Fault Diagnosis**.

UPAFormer is designed to learn unified diagnostic representations from industrial datasets with different signal lengths, sampling frequencies, modality combinations, and operating conditions. The framework is evaluated under a multi-source domain generalization setting, where data from three operating conditions are available for training and the remaining condition is treated as an unseen domain.

## Method Overview

UPAFormer consists of two main components:

1. **Adaptive Residual Modality Perception Embedding (ARMPE)**  
   ARMPE applies Adaptive Residual Modality Normalization (ARMNorm) and configurable perception kernels to heterogeneous multimodal signals. It produces aligned time, time-frequency, and frequency representations with a unified embedding dimension.

2. **Private-Shared Aggregation Encoder (PSAE)**  
   PSAE contains private time and frequency branches together with a shared time-frequency branch. It models branch-specific characteristics and complementary information among the three representations.

A shared classification head produces three prediction-logit vectors. Their average is used for the final prediction. During training, branch-wise classification supervision is combined with inter-domain classification consistency regularization.

## Repository Structure

```text
UPAformer_code/
├── main.py
├── dataset/
│   └── file_path.py
├── data_loader/
│   └── dataloader.py
├── Embedding/
│   └── ARMPE.py
├── models/
│   ├── Diagnositic_model.py
│   ├── UPAformer.py
│   ├── PSAEncoder.py
│   └── PSAE_Attention.py
├── MMD/
│   └── MMD_calculation.py
├── utils_dsbn/
│   ├── save_other_functions.py
│   └── train_utils.py
└── save_dir/
```

The principal files are:

| File | Description |
|---|---|
| `main.py` | Training, validation, unseen-domain evaluation, repeated experiments, and result export |
| `dataset/file_path.py` | Dataset directories and source/unknown-domain task definitions |
| `data_loader/dataloader.py` | NumPy data loading and source-domain splitting |
| `Embedding/ARMPE.py` | ARMNorm and unified time, time-frequency, and frequency perception |
| `models/UPAformer.py` | Overall UPAFormer architecture |
| `models/PSAEncoder.py` | Private-Shared Aggregation Encoder |
| `models/PSAE_Attention.py` | Private and shared attention operations |
| `MMD/MMD_calculation.py` | Inter-domain classification consistency calculation |

## Environment

A Python environment with PyTorch and the following packages is required:

```bash
pip install numpy pandas scipy scikit-learn matplotlib einops tqdm
```

Install a PyTorch version compatible with the local CUDA toolkit before running the experiments.

An example environment can be created using:

```bash
conda create -n upaformer python=3.10
conda activate upaformer
```

## Datasets

Four heterogeneous multimodal fault-diagnosis datasets are used. They cover a SCARA robot ball-screw system, a rolling bearing, a wind-turbine gearbox, and a motor.

The term **modality** in this repository denotes an individual signal channel supplied to the model. Different channels may originate from different sensor types or electrical measurements.

### Dataset Summary

| Dataset | Institution and platform | Operating conditions | Health states | Sampling rate | Sample length | Input modalities | Window duration |
|---|---|---:|---:|---:|---:|---:|---:|
| SUDA/SCARA | Soochow University SCARA robot ball-screw system | 4 load conditions with time-varying speed | 4 | 1.6 kHz | 1024 points | 7 | 0.640 s |
| PU | Paderborn University bearing test rig | 4 combined speed–torque–force conditions | 4 | 64 kHz | 4096 points | 3 | 0.064 s |
| BJUT | Beijing Jiaotong University wind-turbine gearbox | 4 operating-frequency conditions | 5 | 48 kHz | 4096 points | 3 | 0.0853 s |
| HUST | Huazhong University of Science and Technology motor platform | 4 operating-frequency conditions | 6 | 25.6 kHz | 2048 points | 4 | 0.080 s |

### 1. SUDA/SCARA Robot Ball-Screw Dataset

The SUDA dataset was collected from a SCARA robot ball-screw experimental platform at **Soochow University**. The dataset is referred to as `SUDA` in the manuscript and `SCARA_multi_modal_datasets` in the code.

#### Operating conditions

Signals were collected under time-varying rotational speeds and four payload conditions:

| Condition | Payload | Code identifier |
|---|---:|---|
| 0 kg | 0 kg | `SCARA_0kg_4000r_e1600Hz_v1600Hz` |
| 3 kg | 3 kg | `SCARA_3kg_4000r_e1600Hz_v1600Hz` |
| 6 kg | 6 kg | `SCARA_6kg_4000r_e1600Hz_v1600Hz` |
| 9 kg | 9 kg | `SCARA_9kg_4000r_e1600Hz_v1600Hz` |

Each payload condition is treated as an independent domain.

#### Health states

The dataset contains four ball-screw health states:

| State | Description |
|---|---|
| Healthy | Normal ball-screw operation |
| Screw-nut jamming | Jamming of the screw-nut mechanism |
| Spline-nut jamming | Jamming of the spline-nut mechanism |
| Ball loss | Loss or shedding of recirculating balls |

#### Signal modalities

Each model input contains seven signal channels:

| Modality group | Signals | Number |
|---|---|---:|
| Electrical | Current-feedback signal | 1 |
| Electrical | \(U\)-, \(V\)-, and \(W\)-phase currents | 3 |
| Electrical | Motor \(d\)-axis current-feedback signal | 1 |
| Mechanical | Two vibration signals | 2 |
| **Total** |  | **7** |

The raw SUDA arrays may contain additional channels. In the current implementation, `data_loader/dataloader.py` selects channels `[1, 2, 3, 4, 5, 6, 7]` and truncates each sample to the first 1024 points. Therefore, the channel order in newly prepared files must follow the same convention.

#### Input configuration

```text
Sampling rate: 1,600 Hz
Sample length: 1,024 points
Input shape:   [number of samples, 1024, 7]
Number of classes: 4
```

### 2. PU Bearing Dataset

The PU dataset was collected from the bearing experimental platform at **Paderborn University**. Each operating condition combines a rotational speed, load torque, and radial force.

#### Operating conditions

| No. | Code identifier | Rotational speed | Load torque | Radial force |
|---:|---|---:|---:|---:|
| 0 | `N15_M07_F10` | 1500 r/min | 0.7 N·m | 1000 N |
| 1 | `N09_M07_F10` | 900 r/min | 0.7 N·m | 1000 N |
| 2 | `N15_M01_F10` | 1500 r/min | 0.1 N·m | 1000 N |
| 3 | `N15_M07_F04` | 1500 r/min | 0.7 N·m | 400 N |

The corresponding processed filenames begin with `PU_`, for example:

```text
PU_N15_M07_F10_data.npy
PU_N15_M07_F10_label.npy
```

#### Health states

The four-class PU configuration used in the experiments contains:

| State | Description |
|---|---|
| Healthy | Bearing without an introduced fault |
| Outer-race fault | Localized outer-race damage |
| Inner-race fault | Localized inner-race damage |
| Combined fault | Simultaneous inner- and outer-race faults |

#### Signal modalities

| Modality group | Signals | Number |
|---|---|---:|
| Electrical | Two motor-current signals | 2 |
| Mechanical | One vibration signal | 1 |
| **Total** |  | **3** |

#### Input configuration

```text
Sampling rate: 64,000 Hz
Sample length: 4,096 points
Input shape:   [number of samples, 4096, 3]
Number of classes: 4
```

The code also retains a three-class PU configuration, but the default experiments use:

```python
PU_datasets_4classes_4096_1samples
```

### 3. BJUT Wind-Turbine Gearbox Dataset

The BJUT dataset was collected from a wind-turbine gearbox platform at **Beijing Jiaotong University**.

#### Operating conditions

Four operating-frequency conditions are used:

| Condition | Code identifier |
|---:|---|
| 20 Hz | `BJUT_20Hz_5` |
| 25 Hz | `BJUT_25Hz_5` |
| 30 Hz | `BJUT_30Hz_5` |
| 35 Hz | `BJUT_35Hz_5` |

Each frequency condition forms an independent domain.

#### Health states

| State | Description |
|---|---|
| Healthy | Normal gearbox operation |
| Broken tooth | Partial or complete tooth breakage |
| Missing tooth | Complete loss of one gear tooth |
| Tooth-root crack | Crack initiated near the tooth root |
| Wear | Gear-tooth surface wear |

#### Signal modalities

| Modality group | Signals | Number |
|---|---|---:|
| Mechanical | Two vibration signals | 2 |
| Rotational reference | One encoder signal | 1 |
| **Total** |  | **3** |

#### Input configuration

```text
Sampling rate: 48,000 Hz
Sample length: 4,096 points
Input shape:   [number of samples, 4096, 3]
Number of classes: 5
```

### 4. HUST Motor Dataset

The HUST dataset was collected from a motor experimental platform at **Huazhong University of Science and Technology**.

#### Operating conditions

| Condition | Code identifier |
|---:|---|
| 5 Hz | `HUST_05Hz_6` |
| 10 Hz | `HUST_10Hz_6` |
| 20 Hz | `HUST_20Hz_6` |
| 30 Hz | `HUST_30Hz_6` |

Each operating-frequency condition is treated as an independent domain.

#### Health states

| State | Description |
|---|---|
| Healthy | Normal motor operation |
| Bearing fault | Motor-bearing damage |
| Bowed rotor | Rotor bending or bowing |
| Broken rotor bars | Damage to one or more rotor bars |
| Rotor misalignment | Misalignment of the rotor system |
| Voltage unbalance | Unbalanced motor-supply voltage |

#### Signal modalities

| Modality group | Signals | Number |
|---|---|---:|
| Mechanical | Three vibration signals | 3 |
| Acoustic | One acoustic signal | 1 |
| **Total** |  | **4** |

#### Input configuration

```text
Sampling rate: 25,600 Hz
Sample length: 2,048 points
Input shape:   [number of samples, 2048, 4]
Number of classes: 6
```

## Processed Data Format

The current data loader reads NumPy files. Each operating domain requires one data file and one label file:

```text
<domain_name>_data.npy
<domain_name>_label.npy
```

The expected array formats are:

| File | Shape | Description |
|---|---|---|
| `*_data.npy` | `[N, L, C]` | \(N\) samples, \(L\) time points, and \(C\) modalities |
| `*_label.npy` | `[N]` | Integer class labels corresponding to the samples |

Labels must be zero-based integer indices in the range:

```text
0, 1, ..., number_of_classes - 1
```

The exact correspondence between label indices and health states must remain identical across all operating domains of the same dataset.

### Expected Directory Layout

```text
UPAformer_code/
└── dataset/
    ├── SCARA_datasets/
    │   └── electrical_vibration_1600Hz_vib/
    │       └── 第1批/
    │           └── 4000r/
    │               └── 1600Hz_elc/
    │                   ├── SCARA_0kg_4000r_e1600Hz_v1600Hz_data.npy
    │                   ├── SCARA_0kg_4000r_e1600Hz_v1600Hz_label.npy
    │                   ├── SCARA_3kg_4000r_e1600Hz_v1600Hz_data.npy
    │                   ├── SCARA_3kg_4000r_e1600Hz_v1600Hz_label.npy
    │                   ├── SCARA_6kg_4000r_e1600Hz_v1600Hz_data.npy
    │                   ├── SCARA_6kg_4000r_e1600Hz_v1600Hz_label.npy
    │                   ├── SCARA_9kg_4000r_e1600Hz_v1600Hz_data.npy
    │                   └── SCARA_9kg_4000r_e1600Hz_v1600Hz_label.npy
    ├── PU_datasets_4classes_4096_1samples/
    │   ├── PU_N15_M07_F10_data.npy
    │   ├── PU_N15_M07_F10_label.npy
    │   ├── PU_N09_M07_F10_data.npy
    │   ├── PU_N09_M07_F10_label.npy
    │   ├── PU_N15_M01_F10_data.npy
    │   ├── PU_N15_M01_F10_label.npy
    │   ├── PU_N15_M07_F04_data.npy
    │   └── PU_N15_M07_F04_label.npy
    ├── BJUT_WT_64_samples_all_speed/
    │   ├── BJUT_20Hz_5_data.npy
    │   ├── BJUT_20Hz_5_label.npy
    │   ├── BJUT_25Hz_5_data.npy
    │   ├── BJUT_25Hz_5_label.npy
    │   ├── BJUT_30Hz_5_data.npy
    │   ├── BJUT_30Hz_5_label.npy
    │   ├── BJUT_35Hz_5_data.npy
    │   └── BJUT_35Hz_5_label.npy
    └── HUST_motor_2048/
        ├── HUST_05Hz_6_data.npy
        ├── HUST_05Hz_6_label.npy
        ├── HUST_10Hz_6_data.npy
        ├── HUST_10Hz_6_label.npy
        ├── HUST_20Hz_6_data.npy
        ├── HUST_20Hz_6_label.npy
        ├── HUST_30Hz_6_data.npy
        └── HUST_30Hz_6_label.npy
```

## Domain-Generalization Protocol

Each operating condition is regarded as one domain. For every task:

- Three domains are used as labeled source domains.
- The remaining domain is treated as the unseen domain.
- No unseen-domain sample contributes to gradient calculation.
- Source-domain samples are randomly divided into 70% training and 30% validation subsets.
- The entire unseen-domain dataset is used for evaluation.
- Model selection is based on source-domain validation accuracy and validation loss.
- Each task is repeated ten times using seeds 1–10.

The unseen-domain labels are loaded only to calculate diagnostic accuracy and export predictions. They are not included in the optimization loss.

### SUDA Transfer Tasks

| Task | Source domains | Unseen domain |
|---|---|---|
| T0 | 3, 6, and 9 kg | 0 kg |
| T3 | 0, 6, and 9 kg | 3 kg |
| T6 | 0, 3, and 9 kg | 6 kg |
| T9 | 0, 3, and 6 kg | 9 kg |

### PU Transfer Tasks Implemented in the Code

The following mapping follows `dataset/file_path.py` in the current release.

| Task | Source conditions | Unseen condition |
|---|---|---|
| T0 | No. 2, No. 3, and No. 0 | No. 1 (`N09_M07_F10`) |
| T1 | No. 1, No. 3, and No. 0 | No. 2 (`N15_M01_F10`) |
| T2 | No. 1, No. 2, and No. 0 | No. 3 (`N15_M07_F04`) |
| T3 | No. 1, No. 2, and No. 3 | No. 0 (`N15_M07_F10`) |

### BJUT Transfer Tasks

| Task | Source domains | Unseen domain |
|---|---|---|
| T20 | 25, 30, and 35 Hz | 20 Hz |
| T25 | 20, 30, and 35 Hz | 25 Hz |
| T30 | 20, 25, and 35 Hz | 30 Hz |
| T35 | 20, 25, and 30 Hz | 35 Hz |

### HUST Transfer Tasks

| Task | Source domains | Unseen domain |
|---|---|---|
| T5 | 10, 20, and 30 Hz | 5 Hz |
| T10 | 5, 20, and 30 Hz | 10 Hz |
| T20 | 5, 10, and 30 Hz | 20 Hz |
| T30 | 5, 10, and 20 Hz | 30 Hz |

## Default Experimental Settings

| Parameter | Default value |
|---|---:|
| Optimizer | Adam |
| Learning rate | \(3\times10^{-4}\) |
| Batch size | 32 per source domain |
| Training epochs | 100 |
| Repeated runs | 10 |
| Number of perception kernels | 128 |
| Kernel length | 72 |
| Unified embedding dimension | 160 |
| PSAE layers | 1 |
| Attention heads | 2 |
| Dropout | 0.1 |
| ARMNorm | Enabled |
| ARMNorm residual weight | 0.1 |
| Inter-domain consistency weight | 0.3 |
| Classification head | Shared |
| Final prediction | Average of three branch logits |

The modality width of each perception kernel is dataset dependent:

| Dataset | Total modalities | Modalities per kernel |
|---|---:|---:|
| SUDA | 7 | 6 |
| PU | 3 | 2 |
| BJUT | 3 | 2 |
| HUST | 4 | 3 |

## Running the Experiments

Run commands from the `UPAformer_code` directory:

```bash
cd UPAformer_code
python main.py
```

By default, `main.py` sequentially evaluates:

- Four datasets;
- Four unseen-domain tasks per dataset;
- Ten repeated runs per task.

This corresponds to 160 complete training runs for one learning-rate and loss-weight configuration.

Scalar parameters can be changed from the command line:

```bash
python main.py --cuda 0 --batch_size 32 --n_epoch 100
```

Experiment grids such as `dataset_names`, `learning_rates`, `alpha1_params`, and the transfer-task lists are stored as Python lists in `main.py`. For reliable modification, edit their default values directly before execution.

For example, to evaluate only BJUT, set:

```python
parser.add_argument(
    '--dataset_names',
    type=list,
    default=['BJUT_WT_datasets']
)
```

To evaluate only task T20, set:

```python
parser.add_argument(
    '--BJUT_transfer_tasks',
    type=list,
    default=[20]
)
```

## Output Files

Results are written to:

```text
save_dir/UPAformer/
```

The output hierarchy follows:

```text
save_dir/UPAformer/
└── lr_<learning_rate>/
    └── weighting_<consistency_weight>/
        └── <dataset_name>/
            └── <task_name>/
                ├── <repeat>-<accuracy>/
                │   ├── *_training_history.csv
                │   ├── *_prediction_labels.csv
                │   └── *.pth
                ├── *_best_acc_and_total_time.csv
                └── The mean and std deviation of per task.csv
```

The code additionally exports:

- Source-domain training and validation losses;
- Time, time-frequency, and frequency branch losses;
- Unseen-domain diagnostic accuracy;
- Prediction labels for the selected checkpoint;
- Mean and standard deviation over repeated experiments;
- Training time for each repeated run;
- A complete copy of the experimental arguments in `args.txt`.

## Reproducibility Checklist

Before running an experiment, verify that:

1. Every `*_data.npy` file has shape `[N, L, C]`.
2. Every `*_label.npy` file contains exactly `N` labels.
3. Class-index assignments are identical across all domains of a dataset.
4. Sampling rates in `main.py` match the processed signals.
5. SUDA channel ordering is compatible with the selected indices `[1, 2, 3, 4, 5, 6, 7]`.
6. The intended unseen-domain task agrees with `dataset/file_path.py`.
7. Training is launched from the `UPAformer_code` directory.
8. The selected CUDA device exists and has sufficient memory.
9. Reported results are calculated from the ten repeated runs rather than from the best individual run.

## Citation

Citation information will be added after publication. When using this code or the associated datasets, please cite the UPAFormer paper and the original dataset sources where applicable.

## Acknowledgment

The experiments use heterogeneous multimodal datasets collected from Soochow University, Paderborn University, Beijing Jiaotong University, and Huazhong University of Science and Technology.
