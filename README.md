<div align="center">

# [CVPR 2026] PureCC: Pure Learning for Text-to-Image Concept Customization

**[Zhichao Liao](https://lzc-sg.github.io/)**\*‡, **[Xiaole Xian](https://github.com/connorxian/)**\*‡, Qingyu Li, Wenyu Qin, Meng Wang, **[Weicheng Xie](https://wcxie.github.io/Weicheng-Xie/)** ✉️, Siyang Song, Pingfa Feng, **[Long Zeng](https://jackyzengl.github.io/)** ✉️, [Liang Pan](https://ethan7899.github.io/)  

*Tsinghua University · Shenzhen University  
Kling Team, Kuaishou Technology · University of Exeter · S-lab, Nanyang Technological University*

\* Equal contribution, ✉️ Corresponding author  
‡ Work conducted during an internship at Kling Team, Kuaishou Technology

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2603.07561)
[![Code](https://img.shields.io/badge/Code-GitHub-black)](https://github.com/lzc-sg/PureCC)
<!-- [![License](https://img.shields.io/badge/License-MIT-blue.svg)](./LICENSE)
[![Project Page](https://img.shields.io/badge/Project-Page-green)](https://your-project-page.github.io/) -->

</div>


##  🔥 News

- [x] **`2026/03/08`**: 🔥 We released the technical report on [arXiv](https://arxiv.org/abs/2603.07561).
- [x] **`2026/02/21`**: 🔥 PureCC was accepted by CVPR 2026.


## 🌏 Open Source
Thank you all for your attention! We are actively cleaning our technical report, models, and codes, and we will open source them soon.
- [x] Technical Paper on [arXiv](https://arxiv.org/abs/2603.07561)
- [ ] Training and Inference code on GitHub




## ✨ Highlight

### 🚀 Teaser

<p align="center">
<img src="assets/teaser.png" width="95%">
</p>

**PureCC enables high-fidelity personalized concept customization while better preserving the original model behavior and generation capability.**

### 💥 Motivation

<p align="center">
<img src="assets/insight.png" width="88%">
</p>

🔥🔥🔥 ***The goal of an I2I editing or inpainting task*** is to perform a one-time visual modification on a given image, with the ***focus on transforming that specific image into the desired result***. In contrast, ***PureCC*** aims to ***teach the model a new concept***. Moreover, compared with other concept customization methods, it not only emphasizes concept fidelity, but also highlights ***“pure learning”*** — learning only the target concept itself while ***minimizing disruption to the original model’s behavior, distribution, and capabilities***.

### 🥥 Pipeline

<p align="center">
<img src="assets/pipeline.png" width="95%">
</p>

## 💖 Results

## Concept Customization

<p align="center">
<img src="assets/sota.png" width="95%">
</p>

## Multi-Concept Customization

<p align="center">
<img src="assets/multi-ref.png" width="95%">
</p>

## Instance + Style Customization

<p align="center">
<img src="assets/stylized.png" width="95%">
</p>

## 🖊 Citation
If you find PureCC useful for your research, welcome to 🌟 this repo and cite our work using the following BibTeX:

```bibtex
@misc{liao2026purecc,
      title={PureCC: Pure Learning for Text-to-Image Concept Customization}, 
      author={Zhichao Liao and Xiaole Xian and Qingyu Li and Wenyu Qin and Meng Wang and Weicheng Xie and Siyang Song and Pingfa Feng and Long Zeng and Liang Pan},
      year={2026},
      eprint={2603.07561},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2603.07561}, 
}
```
## 🔗 Related Work (Pure Learning)
- [PuLID: Pure and Lightning ID Customization via Contrastive Alignment](https://github.com/ToTheBeginning/PuLID)
- [SPF-Portrait: Towards Pure Portrait Customization with Semantic Pollution-Free Fine-tuning](https://github.com/KlingAIResearch/SPF-Portrait)
