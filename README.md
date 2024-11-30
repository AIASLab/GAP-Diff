# GAP-Diff: Protecting JPEG-Compressed Images from Diffusion-based Facial Customization

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.14249397.svg)](https://doi.org/10.5281/zenodo.14249397)

This repository contains the artifact for the paper _GAP-Diff: Protecting JPEG-Compressed Images from Diffusion-based Facial Customization_ that will appear at [NDSS Symposium 2025](https://www.ndss-symposium.org/ndss2025/). The paper will be available at [here](https://dx.doi.org/10.14722/ndss.2025.242088) once officially published. 

This branch contains the exact code that will be evaluated during the Artifact Evaluation (AE) process of NDSS'25. We are now applying for the _Available_ and _Functional_ badges. 
Since the code requires high hardware specifications, AEC can directly **access our rented server** (SSH address has been submitted to the HotCRP Instructions) to finish the whole AE process.

## Abstract
#### A Synthetic Description of the Research
This research addresses privacy and portrait rights challenges arising from text-to-image diffusion model fine-tuning techniques. We propose GAP-Diff, a framework for **G**enerating data with **A**dversarial **P**erturbations for text-to-image **Diff**usion models. GAP-Diff employs unsupervised learning-based optimization to generate protective noise resistant to JPEG compression and other preprocessing techniques. It learns robust representations by backpropagating gradient information through a preprocessing simulation module while developing adversarial characteristics to disrupt fine-tuned models. Our method achieves an adversarial mapping from clean to protected images, generating stronger protective noise within milliseconds. Facial benchmark experiments demonstrate that our approach significantly enhances the resilience of protective noise against JPEG compression, thereby better safeguarding user privacy and copyrights in digital environments.

#### How the Proposed Artifacts Support the Research
Our artifacts include the source code of our framework, evaluation methods, a test dataset and benchmarks to reproduce all training and testing process presented in the paper. Due to the temporal constraints of the AE process, we provide pre-trained weights and a scaled-down testing dataset. By running our scripts, AEC can evaluate the counter-customization effects of the protective noise output by GAP-Diff, as well as its resistance to JPEG compression. Quantitative results using four facial metrics and qualitative results can demonstrate the effectiveness and superiority of our method compared to benchmarks. Further, the whole evaluation process will conform to the expectations set by our paper regarding functionality, usability, and relevance.

## Artifact Evaluation

Below you can find detailed instructions to verify the functionality of our method. We provide instructions for running the experiments using our pre-built environments on the rented cloud server. However, if desired and GPU resources are sufficient, the [Getting Started](#getting-started) section contains extensive instructions for setting up all experimental environments.

### Overview

* [Getting started](#getting-started) (~1-10 human-minutes, ~10-30 compute-minutes)
* [Kick-the-tires stage](#kick-the-tires-stage) (~2 human-minutes, ~20 compute-minutes)
* [Counter customization](#counter-customization) (~2 human-minutes, ~1.5 compute-hours)
* [Against JPEG compression](#against-jpeg-compression) (~10 human-minutes, ~7.5-10.5 compute-hours)
* [Adverse setting](#adverse-setting) (~10 human-minutes, ~9 compute-hours)

### Getting started
**AEC using our rented cloud server for testing can skip this section, and directly go to [Kick-the-tires stage](#kick-the-tires-stage)**

All tasks can be completed on a single NVIDIA A800 80G GPU, and the most testing tasks can be down on an NVIDIA RTX 4090 24G GPU. A recent Linux operating system >= Ubuntu 20.04 with Anaconda (or miniconda3) and CUDA 11.8 is needed.

You should firstly clone the code of GAP-Diff and an additional fine-tuning method [SVDiff](https://github.com/mkshing/svdiff-pytorch).

- To build the environment for GAP-Diff.
```bash
$ cd GAP-Diff
$ conda create -n gap-diff python=3.9
$ conda activate gap-diff
$ pip install -r requirements.txt
```
- To build the environment for one of the benchmark [CAAT](https://github.com/CO2-cityao/CAAT).
```bash
$ cd GAP-Diff/benchmark
$ conda create -n CAAT python=3.8 
$ conda activate CAAT  
$ pip install -r requirements.txt 
```
- To build the environment for evaluation.
```bash
$ conda create -n fdfr_ism python=3.8
$ conda activate fdfr_ism
$ cd GAP-Diff/evaluations/deepface/
$ pip install -e .
$ cd ..
$ cd retinaface/
$ pip install -e . # to this step the tensorflow in our environment is 2.13.1
$ pip install torch
$ conda create -n serfiq python=3.8
$ conda activate serfiq
$ cd ..
$ cd FaceImageQuality/
$ pip install -r requirements.txt
```
- To compute FDFR and ISM. If you get block by proxy, manually download weight of [ArcFace](https://github.com/serengil/deepface_models/releases/download/v1.0/arcface_weights.h5) and [Retinaface](https://github.com/serengil/deepface_models/releases/download/v1.0/retinaface.h5), place it in folder `{home}/.deepface/weights/`, where `{home}` means the user home directory, e.g., the `$HOME` directory in Linux.

- To compute SER-FIQ, [download the model files](https://drive.google.com/file/d/17fEWczMzTUDzRTv9qN3hFwVbkqRD7HE7/view?usp=sharing) and place them in the `GAP-Diff/evaluations/FaceImageQuality/insightface/model`.

- To build the environment for the additional fine-tuning method SVDiff
```bash
$ cd svdiff-pytorch
$ conda create -n svdiff python=3.9
$ conda activate svdiff
$ pip install -r requirements.txt
```
- Pretrained checkpoints of different Stable Diffusion versions can be **downloaded** from provided links in the table below, you should modify the path in scripts after downloading, and they should be placed following the structure `GAP-Diff/stable-diffusion/stable-diffusion-2-1-base`:
  <table style="width:100%">
  <tr>
    <th>Version</th>
    <th>Link</th>
  </tr>
  <tr>
    <td>2.1</td>
    <td><a href="https://huggingface.co/stabilityai/stable-diffusion-2-1-base">stable-diffusion-2-1-base</a></td>
  </tr>
  <tr>
    <td>1.5</td>
    <td><a href="https://huggingface.co/runwayml/stable-diffusion-v1-5">stable-diffusion-v1-5</a></td>
  </tr>
  <tr>
    <td>1.4</td>
    <td><a href="https://huggingface.co/CompVis/stable-diffusion-v1-4">stable-diffusion-v1-4</a></td>
  </tr>
  </table>
- You can also download [FFHQ](https://www.kaggle.com/datasets/arnaud58/flickrfaceshq-dataset-ffhq), [VGGFace2](https://drive.google.com/drive/folders/1JX4IM6VMkkv4rER99atS4x4VGnoRNByV) and [CelebA-HQ](https://drive.google.com/drive/folders/1JX4IM6VMkkv4rER99atS4x4VGnoRNByV) (preprocessed by authors of [Anti-DreamBooth](https://github.com/VinAIResearch/Anti-DreamBooth)) for the whole training and testing, which should be placed in folder `GAP-Diff/data`. 
- The pretrained weight of GAP-Diff can be downloaded from [here](https://drive.google.com/file/d/1UVYFNOLlXl2xZNgVzBkXmMsDx693R6ud/view?usp=drive_link), please place the weight file in `GAP-Diff/weights/model`.
- If you are a user with 24GB or less GPU memory, please download the `class-person-2-1` folder from [here](https://drive.google.com/drive/folders/1zKPJfgfCETle1tpFtqAPbK_FNjHdEsb2?usp=drive_link) and place it to match `GAP-Diff/data/class-person-2-1`.



### Kick-the-tires stage
In this step, we will test our setup by running a minimal working example to ensure that our artifacts run correctly.

Before running the scripts, if you want the output of all script files to be directed to different output files instead of displaying directly in the console, change the statement in all `sh` files from `exec > >(cat) 2> results/evaluate_fdfr_ism_error.txt` to something like `exec > results/evaluate_fdfr_ism_output.txt 2> results/evaluate_fdfr_ism_error.txt`.

#### STEP 1: Run the generator with the pre-trained weight, train DreamBooth to get a FT-T2I-DM and infer customized outcomes.(~1 human-minutes, ~14-15 compute-minutes)
```bash
$ cd GAP-Diff
$ conda activate gap-diff
$ bash scripts/kick-the-tires/ktt_generate_db_infer.sh
# Expected output:
# - Command line output "the images are saved", "Finish training" and infer log
# - Protected images are saved to GAP-Diff/protected_images/kick-the-tires/gap_diff_per16
# - The customized outcomes are saved to GAP-Diff/infer/kick-the-tires/gap_diff_per16
```
#### STEP 2: Run the two evaluation scripts to get four facial metrics of protective effect.(~1 human-minutes, ~2 compute-minutes)
```bash
$ conda activate fdfr_ism
$ bash scripts/kick-the-tires/ktt_evaluate_fdfr_ism.sh
# Expected output:
# - Command line output the mean FDFR and ISM of the identity for each prompt in the customized outcomes
$ conda activate serfiq
$ bash scripts/kick-the-tires/ktt_evaluate_brisque_serfiq.sh
# Expected output:
# - Command line output the mean BRISQUE and SER-FIQ of the identity for each prompt in the customized outcomes
```

#### STEP 3: Run the script on the local machine to download the customized outcomes and the clean images for evaluating the qualitative results of the methods.
```bash
$ sftp -P <PORT> <USERNAME>@<HOST>
sftp> get -r /root/GAP-Diff/infer/kick-the-tires/gap_diff_per16/n000089/a_photo_of_sks_person <YOURPATH> 
sftp> get -r /root/GAP-Diff/data/test_dataset/n000089/set_B <YOURPATH> 
```

We have provided keywords `PORT`, `USERNAME` and `HOST` in the Instructions on HotCRP and the artifact appendix. If you are a Linux user, please format `YOURPATH` as `/username/xxx`, or you are a Windows user, please format as `D:\xxx`


### Counter customization

GAP-Diff demonstrates effective facial protection using pre-trained weights when countering T2I-DM customization. This is verified by the following experiments whose results are reported in Fig. 9 (in our original paper) under Quality=100.


The steps are the same as done in kick-the-tires stage, but this time we will carry out the experiment on 5 different identities.

In total, this experiment should take around **1.5 compute-hours**.

```bash
# go to the `GAP-Diff` folder
cd GAP-Diff

# Step 1: Run the GAP-Diff generator with pre-trained weights to obtain the protected images.(~1 minute)
$ conda activate gap-diff
$ bash scripts/generate.sh
# Expected output:
# - Command line output "the images are saved" for each identity
# - The protected images are saved to GAP-Diff/protected_images/gap-diff-per16

# Step 2: Use the protected images in DreamBooth training to obtain the FT-T2I-DM and infer the customized outcomes with 4 prompts.(~70-75 minutes)
$ bash scripts/db_infer.sh
# Expected output:
# - Command line output "Finish training" and infer log for each identity
# - The customized outcomes are saved to GAP-Diff/infer/gap_diff_per16

# Step 3: Evaluate the customized outcomes for the protective effect of GAP-Diff.(~10 minutes)
$ conda activate fdfr_ism
$ bash scripts/evaluate_fdfr_ism.sh
# Expected output:
# - Command line output the mean FDFR and ISM of all identities for each prompt in the customized outcomes
$ conda activate serfiq
$ bash scripts/evaluate_brisque_serfiq.sh
# Expected output:
# - Command line output the mean BRISQUE and SER-FIQ of all identities for each prompt in the customized outcomes
```

### Against JPEG compression

The protective noise generated by GAP-Diff shows superior resistance to JPEG compression compared to benchmark methods. This is proven by the following experiments, with results illustrated in Table I and Fig. 2 and Fig. 8 (in our original paper) . Further, during these experiments, the resistance to different JPEG quality of GAP-Diff can also be evaluated with results illustrated in Fig. 9 (in our original paper) .

In total, this experiment should take around **10.5 compute-hours** if you do all the steps.

#### Part 1: Evaluate the JPEG compression resistance of GAP-Diff (~1.5 compute-hours)
```bash
# go to the `GAP-Diff` folder
cd GAP-Diff

# Step 1: Apply JPEG compression with a quality of Q=70 to the protected images
$ conda activate gap-diff
$ bash scripts/preprocess/jpeg.sh
# Expected output:
# - Command line output "Image compression completed successfully!" 
# - The JPEG compressed protected images are saved to GAP-Diff/infer/gap_diff_per16_jpeg70

# Step 2: Modify the GAP-Diff/scripts/db_infer.sh, export TASK_NAME="gap_diff_per16_jpeg70" in line 6

# Step 3: Use the JPEG compressed protected images in DreamBooth training to obtain the FT-T2I-DM and infer the customized outcomes with 4 prompts.(~70-75 minutes)
$ bash scripts/db_infer.sh
# Expected output:
# - Command line output "Finish training" and infer log for each identity
# - The customized outcomes are saved to GAP-Diff/infer/gap_diff_per16_jpeg70

# Step 4: Modify the GAP-Diff/scripts/evaluate_fdfr_ism.sh and evaluate_brisque_serfiq.sh, export DATA_PATH="./infer/gap_diff_per16_jpeg70"  in line 6

# Step 5: Evaluate the customized outcomes for the JPEG compression resistance of GAP-Diff.(~10 minutes)
$ conda activate fdfr_ism
$ bash scripts/evaluate_fdfr_ism.sh
# Expected output:
# - Command line output the mean FDFR and ISM of all identities for each prompt in the customized outcomes
$ conda activate serfiq
$ bash scripts/evaluate_brisque_serfiq.sh
# Expected output:
# - Command line output the mean BRISQUE and SER-FIQ of all identities for each prompt in the customized outcomes
```

#### Part 2: Evaluate the JPEG compression resistance of benchmarks (~6 compute-hours)

Due to time and memory constraints, we provide implementations for Anti-DB and CAAT in benchmarks, while for SimAC and MetaCloak, we only supply the protected images.

Please follow the steps outlined below for evaluation.
```bash
# go to the `GAP-Diff` folder
cd GAP-Diff

# generate protected images and JPEG compressed ones of benchmarks, DreamBooth training, infer and evaluate.
$ conda activate gap-diff
$ bash benchmark/scripts/antidb.sh
$ conda activate CAAT
$ bash benchmark/scripts/caat.sh
$ conda activate gap-diff
$ bash benchmark/scripts/jpeg.sh
$ bash benchmark/scripts/db_infer.sh
$ conda activate fdfr_ism
$ bash benchmark/scripts/evaluate_fdfr_ism.sh
$ conda activate serfiq
$ bash benchmark/scripts/evaluate_brisque_serfiq.sh
# Expected output:
# - The protected images and JPEG compressed ones are saved to GAP-Diff/benchmark/protected_images
# - The customized outcomes of benchmarks are saved to GAP-Diff/benchmark/infer
# - Command line output the four quantitative results of all identities for each prompt in the customized outcomes of benchmarks
```

You can download the customized outcomes to compare the qualitative results of JPEG compression resistance of GAP-Diff and benchmarks.
```bash
$ sftp -P <PORT> <USERNAME>@<HOST>
sftp> get -r /root/GAP-Diff/data/test_dataset/n000061 <YOURPATH>
sftp> get -r /root/GAP-Diff/infer/gap_diff_per16/n000061 <YOURPATH> 
sftp> get -r /root/GAP-Diff/benchmark/infer/antidb_jpeg70/n000061/set_B <YOURPATH>
sftp> get -r  <PATH_TO_OTHER_BENCHMARKS> <YOURPATH>
```

We have provided keywords `PORT`, `USERNAME` and `HOST` in the Instructions on HotCRP and the artifact appendix. If you are a Linux user, please format `YOURPATH` as `/username/xxx`, or you are a Windows user, please format as `D:\xxx`

#### Part 3: Free to try GAP-Diff to other JPEG compression qualities like Q=50, Q=90 (>= 1.5 compute-hours)

```bash
# go to the `GAP-Diff` folder
cd GAP-Diff

# Step 1: Modify the GAP-Diff/scripts/preprocess/jpeg.sh, export QUALITY=50 (quality you want to set) in line 2 

# Step 2: Apply JPEG compression with a quality of Q you set like 50 before to the protected images
$ conda activate gap-diff
$ bash scripts/preprocess/jpeg.sh
# Expected output:
# - Command line output "Image compression completed successfully!" 
# - The JPEG compressed protected images are saved to GAP-Diff/infer/gap_diff_per16_jpeg50

# Step 2: Modify the GAP-Diff/scripts/db_infer.sh, export TASK_NAME="gap_diff_per16_jpeg50" in line 6

# Step 3: Use the JPEG compressed protected images in DreamBooth training to obtain the FT-T2I-DM and infer the customized outcomes with 4 prompts.(~70-75 minutes)
$ bash scripts/db_infer.sh
# Expected output:
# - Command line output "Finish training" and infer log for each identity
# - The customized outcomes are saved to GAP-Diff/infer/gap_diff_per16_jpeg50

# Step 4: Modify the GAP-Diff/scripts/evaluate_fdfr_ism.sh and evaluate_brisque_serfiq.sh, export DATA_PATH="./infer/gap_diff_per16_jpeg50"  in line 6

# Step 5: Evaluate the customized outcomes for the JPEG compression resistance of GAP-Diff.(~10 minutes)
$ conda activate fdfr_ism
$ bash scripts/evaluate_fdfr_ism.sh
$ conda activate serfiq
$ bash scripts/evaluate_brisque_serfiq.sh
# Expected output:
# - Command line output the four quantitative results of all identities for each prompt in the customized outcomes
```

### Adverse setting
GAP-Diff maintains robust facial protection performance under adverse settings, including prompt mismatch, different fine-tuning methods, and different preprocessing techniques. This is demonstrated by the following experiments, with results presented in Fig. 13 and Tables IV, V, VI (in our original paper) .

Note that **clear faces** may be generated in **Part 1** and **Part 2**, which could result in lower FDFR and BRISQUE, and higher SER-FIQ scores. This is due to the diffusion model's customization not being fully achieved. However, we will observe that **ISM remains at a low level**, indicating that the generated faces do not conform to the customized identities.

#### Part 1: Train DreamBooth on prompt mismatch, and infer customized outcomes.  (~1.5 compute-hours)
```bash
# go to the `GAP-Diff` folder
cd GAP-Diff

$ conda activate gap-diff
$ bash scripts/db_infer_prompt_mismatch.sh
$ conda activate fdfr_ism
$ bash scripts/ex/evaluate_fdfr_ism_ex.sh
$ conda activate serfiq
$ bash scripts/ex/evaluate_brisque_serfiq_ex.sh
# Expected output:
# - The customized outcomes are saved to GAP-Diff/infer/gap_diff_per16_ex 
# - Command line output the four quantitative results of all identities for each prompt in the customized outcomes
```

#### Part 2: Run different fine-tuning methods (here SVDiff). (~1.6 compute-hours)
Before proceeding, if you **cloned the svdiff branch directly**, please prepare as follows:
- Place the `GAP-Diff/svdiff/svd.sh` file to match `svdiff-pytorch/scripts/svd.sh`.
- Place the `GAP-Diff/svdiff/infer.py` file to match `svdiff-pytorch/infer.py`.
- Modify line 30 of `svdiff-pytorch/scripts/svd.sh`: change `export INFER_OUTPUT=$(realpath "/root/GAP-Diff/infer/svdiff/$ID")` by replacing the absolute path `/root/GAP-Diff` with the path of your cloned `GAP-Diff` folder.
- Place the `GAP-Diff/protected_images/gap_diff_per16` folder to match `svdiff-pytorch/data/gap_diff_per16`.
- Change line 577 of `svdiff-pytorch/train_svdiff.py` from `logging_dir=logging_dir,` to `project_dir=logging_dir,`.
- Create a folder named `svdiff` in the `GAP-Diff/infer directory`.
- After activating the svdiff environment, run `pip install huggingface_hub==0.25.2`.

**AEC using our rented cloud server for testing can skip the above steps **
```bash
# go to the `svdiff-pytorch` folder
$ cd ..
$ cd svdiff-pytorch

# Step 1: Train SVDiff with the images protected by GAP-Diff and infer customized outcomes. (~1 hour)
$ conda activate svdiff
$ bash scripts/svd.sh
# Expected output:
# - The customized outcomes are saved to GAP-Diff/infer/svdiff 

# Step 2: Modify the GAP-Diff/scripts/evaluate_fdfr_ism.sh and evaluate_brisque_serfiq.sh, export DATA_PATH="./infer/svdiff"  in line 6

# Step 3: Evaluate the customized outcomes for SVDiff.(~10 minutes)
$ conda activate fdfr_ism
$ bash scripts/evaluate_fdfr_ism.sh
$ conda activate serfiq
$ bash scripts/evaluate_brisque_serfiq.sh
# Expected output:
# - Command line output the four quantitative results of all identities for each prompt in the customized outcomes
```

#### Part 3: Run different preprocessing methods.(~1.5 compute-hours)
```bash
# go to the `GAP-Diff` folder
cd GAP-Diff

# Step 1: preprocess the protected images with different methods (gaussianblur, quantization, random_noise, resize), here we choose gaussianblur for example(~1 minute)
$ conda activate gap-diff
$ bash scripts/preprocess/other_preprocess.sh
# Expected output:
# - The preprocessed images are saved to GAP-Diff/protected_images/gap_diff_per16_gb

# Step 2: Choose the preprocessing method you want to test, modify the GAP-Diff/scripts/db_infer.sh export TASK_NAME="gap_diff_per16_gb" in line 6

# Step 3: Use the preprocessed images in DreamBooth training to obtain the FT-T2I-DM and infer the customized outcomes with 4 prompts.(~70-75 minutes)
$ bash scripts/db_infer.sh
# Expected output:
# - Command line output "Finish training" and infer log for each identity
# - The customized outcomes are saved to GAP-Diff/infer/gap_diff_per16_gb

# Step 4: Modify the GAP-Diff/scripts/evaluate_fdfr_ism.sh and evaluate_brisque_serfiq.sh, export DATA_PATH="./infer/gap_diff_per16_gb"  in line 6

# Step 5: Evaluate the customized outcomes for the resistance of GAP-Diff to the preprocessing method.(~10 minutes)
$ conda activate fdfr_ism
$ bash scripts/evaluate_fdfr_ism.sh
$ conda activate serfiq
$ bash scripts/evaluate_brisque_serfiq.sh
# Expected output:
# - Command line output the four quantitative results of all identities for each prompt in the customized outcomes
```
