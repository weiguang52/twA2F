---
license: other
license_name: nvidia-open-model-license
license_link: >-
  https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/
---
## Model Overview

### Description:
Audio2Face-3D-v2.3.1-Claire generates 3D facial animations from audio inputs, for use in applications such as video conferencing, virtual reality, and digital content creation.
This model is ready for commercial/non-commercial use. For source code, documentation, helper scripts, packaged builds, and links to all components in the Audio2Face-3D technology stack, visit the [Audio2Face-3D GitHub repository](https://github.com/NVIDIA/Audio2Face-3D) <br> 

### License/Terms of Use

Use of this model is governed by the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/) <br>

### Deployment Geography: Global <br>

### Use Case: <br>
Audio2Face-3D-v2.3.1-Claire is designed for developers and researchers working on audio-driven animation and emotion detection applications, such as virtual assistants, chatbots, and affective computing systems. <br>

### Release Date:  <br>
Hugging Face: 09/24/2025 via https://huggingface.co/nvidia/Audio2Face-3D-v2.3.1-Claire <br> 

## References(s):
NVIDIA, Audio2Face-3D: Audio-driven Realistic Facial Animation For Digital Avatars, 2025. <br>
[https://arxiv.org/abs/2508.16401](https://arxiv.org/abs/2508.16401)


## Model Architecture:
**Architecture Type:** Transformer, CNN <br>
**Network Architecture:** Wav2vec2.0 <br>
**Number of model parameters:** 3.98x10^7 <br>

## Input: <br>
**Input Type(s):** Audio <br>
**Input Format:** Array of float <br>
**Input Parameters:** One-Dimensional (1D) <br>
**Other Properties Related to Input:** All audio is resampled to 16KHz

## Output: <br>
**Output Type(s):** Facial pose <br>
**Output Format:** Array of float <br>
**Output Parameters:** One-Dimensional (1D) <br>
**Other Properties Related to Output:** Facial pose on skin, tongue, jaw, and eyeballs <br>

Our AI models are designed and/or optimized to run on NVIDIA GPU-accelerated systems. By leveraging NVIDIA’s hardware (e.g. GPU cores) and software frameworks (e.g., CUDA libraries), the model achieves faster training and inference times compared to CPU-only solutions. <br> 

## Software Integration:
**Runtime Engine(s):** 
* Audio2Face-SDK <br>

**Supported Hardware Microarchitecture Compatibility:** <br>
* NVIDIA Ampere <br>
* NVIDIA Blackwell <br>
* NVIDIA Hopper <br>
* NVIDIA Lovelace <br>
* NVIDIA Pascal <br>
* NVIDIA Turing <br>

**Preferred/Supported Operating System(s):**
* Linux <br>
* Windows <br>

The integration of foundation and fine-tuned models into AI systems requires additional testing using use-case-specific data to ensure safe and effective deployment. Following the V-model methodology, iterative testing and validation at both unit and system levels are essential to mitigate risks, meet technical and functional requirements, and ensure compliance with safety and ethical standards before deployment. <br>
This AI model can be embedded as an Application Programming Interface (API) call into the software environment described above. <br>

## Model Version(s):
Audio2Face-3D-v2.3.1-Claire  <br>

## Training, Testing, and Evaluation Datasets:

## Training Dataset:

Data Modality <br>
* Audio <br>
* 3D facial motion <br>

Audio Training Data Size <br>
* Less than 10,000 Hours <br>

Data Collection Method by dataset <br>
* Human - 3D facial motion data and audio <br>

Labeling Method by dataset <br>
* Human - Commercial capture solution and internal labeling <br>

**Properties (Quantity, Dataset Descriptions, Sensor(s)):** Audio and 3D facial motion from multiple speech sequences <br>

### Testing Dataset:

Data Collection Method by dataset:  <br>
* Human - 3D facial motion data and audio <br>

Labeling Method by dataset:  <br>
* Human - Commercial capture solution and internal labeling <br>

**Properties (Quantity, Dataset Descriptions, Sensor(s)):** Audio and 3D facial motion from multiple speech sequences <br>


### Evaluation Dataset:

Data Collection Method by dataset:  <br>
* Human - 3D facial motion data and audio <br>

Labeling Method by dataset:  <br>
* Human - Commercial capture solution and internal labeling <br>

**Properties (Quantity, Dataset Descriptions, Sensor(s)):** Audio and 3D facial motion from multiple speech sequences <br>


## Inference:
**Acceleration Engine:** TensorRT <br>
**Test Hardware:** <br>  
* T4, T10, A10, A40, L4, L40S, A100 <br>
* RTX 6000ADA, A6000, Pro 6000 Blackwell  <br>
* RTX 3080, 3090, 4080, 4090, 5090  <br>

## Ethical Considerations:
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal model team to ensure this model meets requirements for the relevant industry and use case and addresses unforeseen product misuse.

For more detailed information on ethical considerations for this model, please see the Model Card++ Bias, Explainability, Safety & Security, and Privacy Subcards. 
Please report model quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://www.nvidia.com/en-us/support/submit-security-vulnerability/)

## Bias
Field                                                                                               |  Response
:---------------------------------------------------------------------------------------------------|:---------------
Participation considerations from adversely impacted groups [protected classes](https://www.senate.ca.gov/content/protected-classes) in model design and testing:  |  None
Measures taken to mitigate against unwanted bias:                                                   |  None

## Explainability
Field                                                                                                  |  Response
:------------------------------------------------------------------------------------------------------|:---------------------------------------------------------------------------------
Intended Task/Domain:                                                                   |  Customer Service, Media & Entertainment
Model Type:                                                                                            |  Transformer, CNN
Intended Users:                                                                                        |  Interactive avatar developers, Digital content creators
Output:                                                                                                |  Facial pose
Describe how the model works:                                                                          |  Audio input is encoded and concatenated with emotion label, then passed into CNN-based pose decoder to output facial pose.
Name the adversely impacted groups this has been tested to deliver comparable outcomes regardless of:  |  Not Applicable
Technical Limitations & Mitigation:                                                                    |  This model may not work well with poor audio input.
Verified to have met prescribed NVIDIA quality standards:  |  Yes
Performance Metrics:                                                                                   |  Lipsync accuracy, Latency, Throughput
Potential Known Risks:                                                                                 |  This model may generate inaccurate lip poses given low-quality audio input.
Licensing:                                                                                             |  Use of this model is governed by the [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)

## Privacy
Field                                                                                                                              |  Response
:----------------------------------------------------------------------------------------------------------------------------------|:-----------------------------------------------
Generatable or reverse engineerable personal data?                                                     |  No
Personal data used to create this model?                                                                                       |  Yes
Was consent obtained for any personal data used?                                                                                             |  Yes
How often is dataset reviewed?                                                                                                     |  Before Release
Is a mechanism in place to honor data subject right of access or deletion of personal data?                                        |  Yes
If personal data was collected for the development of the model, was it collected directly by NVIDIA?                                            |  Yes
If personal data was collected for the development of the model by NVIDIA, do you maintain or have access to disclosures made to data subjects?  |  Yes
If personal data was collected for the development of this AI model, was it minimized to only what was required?                                 |  Yes
Is there provenance for all datasets used in training?                                                                                |  Yes
Does data labeling (annotation, metadata) comply with privacy laws?                                                                |  Yes
Is data compliant with data subject requests for data correction or removal, if such a request was made?                           |  Yes
Applicable Privacy Policy        | https://www.nvidia.com/en-us/about-nvidia/privacy-policy/

## Safety & Security
Field                                               |  Response
:---------------------------------------------------|:----------------------------------
Model Application Field(s):                               |  Customer Service, Media & Entertainment
Describe the life critical impact (if present).   |  Not Applicable 
Use Case Restrictions:                              |  Abide by [NVIDIA Open Model License](https://www.nvidia.com/en-us/agreements/enterprise-software/nvidia-open-model-license/)
Model and dataset restrictions:            |  The Principle of least privilege (PoLP) is applied limiting access for dataset generation and model development.  Restrictions enforce dataset access during training, and dataset license constraints adhered to.

## Citation
```
@misc{nvidia2025audio2face3d,
      title={Audio2Face-3D: Audio-driven Realistic Facial Animation For Digital Avatars},
      author={Chaeyeon Chung and Ilya Fedorov and Michael Huang and Aleksey Karmanov and Dmitry Korobchenko and Roger Ribera and Yeongho Seol},
      year={2025},
      eprint={2508.16401},
      archivePrefix={arXiv},
      primaryClass={cs.GR},
      url={https://arxiv.org/abs/2508.16401},
      note={Authors listed in alphabetical order}
}
```