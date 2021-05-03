# Ambiguity of Objective Image Quality Metrics: A New Methodology for Performance Evaluation
This is an example code for measuring ambiguity intervals of an objective quality metric. All result data related to the paper will be added soon.

- Version: 0.1
- Modified date: May 4, 2021.
- Work done at MCML Group (http://mcml.yonsei.ac.kr)
- by Manri Cheon (manri.cheon@gmail.com)


## Abstract
Objective image quality metrics try to estimate the perceptual quality of the given image by considering the characteristics of the human visual system. However, it is possible that the metrics produce different quality scores even for two images that are perceptually indistinguishable by human viewers, which have not been considered in the existing studies related to objective quality assessment. In this paper, we address the issue of ambiguity of objective image quality assessment. We propose an approach to obtain an ambiguity interval of an objective metric, within which the quality score difference is not perceptually significant. In particular, we use the visual difference predictor, which can consider viewing conditions that are important for visual quality perception. In order to demonstrate the usefulness of the proposed approach, we conduct experiments with 33 state-of-the-art image quality metrics in the viewpoint of their accuracy and ambiguity for three image quality databases. The results show that the ambiguity intervals can be applied as an additional figure of merit when conventional performance measurement does not determine superiority between the metrics. The effect of the viewing distance on the ambiguity interval is also shown.



## Usage
- This is an example code for measuring ambiguity intervals of an objective quality metric. We only provide code to calculate the ambiguity interval. It is recommended to use the original author's implementation of the objective model and VDP model (HDR-VDP 2.2). Perceivableness map related information is saved in database_vdp_info_live.mat in this code. Simple example of vdp model (just difference!) is also included in the code.

- Download and unzip dataset_example.zip file to /dataset/ folder.
- Run
    ```bash
        python measure_ambiguity.py
    ```
    - Example code for measuring ambiguity intervals of SSIM for one content (ref img and dist imgs).

## Download
- Example dataset : https://drive.google.com/file/d/1ni2FC4GkkHBuAorJGoEqYZnVUw1-Gqoz/view?usp=sharing
- Databases and results : coming soon

## Citation
- Please use the following citation.

```
@article{cheon2021ambiguity,
  title={Ambiguity of objective image quality metrics: A new methodology for performance evaluation},
  author={Cheon, Manri and Vigier, Toinon and Krasula, Luk{\'a}{\v{s}} and Lee, Junghyuk and Le Callet, Patrick and Lee, Jong-Seok},
  journal={Signal Processing: Image Communication},
  volume={93},
  pages={116150},
  year={2021},
  publisher={Elsevier}
}

@inproceedings{cheon2016ambiguity,
  title={Ambiguity-based evaluation of objective quality metrics for image compression},
  author={Cheon, Manri and Lee, Jong-Seok},
  booktitle={2016 Eighth International Conference on Quality of Multimedia Experience (QoMEX)},
  pages={1--6},
  year={2016},
  organization={IEEE}
}
```



## Releated work
- HDR-VDP 2.2 - https://doi.org/10.1117/1.JEI.24.1.010501
- LIVE database - https://live.ece.utexas.edu/research/Quality/subjective.htm

