# From Registration Uncertainty to Segmentation Uncertainty
<a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg"></a> [![arXiv](https://img.shields.io/badge/arXiv-2403.05111-b31b1b.svg)](https://arxiv.org/abs/2403.05111)

keywords: image registration, registration uncertainty, segmentation uncertainty

This is an official **PyTorch** implementation of my paper:\
Chen, Junyu, et al. "From Registration Uncertainty to Segmentation Uncertainty." Accepted to ***ISBI 2024***.

Understanding the uncertainty inherent in deep learning-based image registration models has been an ongoing area of research. Existing methods have been developed to quantify both transformation and appearance uncertainties related to the registration process, elucidating areas where the model may exhibit ambiguity regarding the generated deformation. However, Our
study reveals that **neither transformation nor appearance uncertainty** effectively estimates the potential errors when the registration model is used for label propagation. Here, we propose a novel framework to concurrently estimate both the *epistemic* and *aleatoric* **segmentation uncertainties** for **image registration**.
## Network architecture:
<img src="https://github.com/junyuchen245/Registration_Uncertainty/blob/main/figs/framework.jpg" width="800"/>

## Segmentation and registration uncertainty estimates:
<img src="https://github.com/junyuchen245/Registration_Uncertainty/blob/main/figs/qualitative.jpg" width="800"/>

## Quantitative results:
<img src="https://github.com/junyuchen245/Registration_Uncertainty/blob/main/figs/table.jpg" width="800"/>


## Citation:
If you find this code is useful in your research, please consider to cite:
    
    @article{chen2024registration,
    title={From Registration Uncertainty to Segmentation Uncertainty},
    author={Chen, Junyu and Liu, Yihao and Wei, Shuwen and Bian, Zhangxing and Carass, Aaron and Du, Yong},
    journal={arXiv preprint arXiv:2403.05111},
    year={2024}
    }

