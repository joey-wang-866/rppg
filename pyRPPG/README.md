# pyRPPG

Python implementation of some remote Photoplethysmography (rPPG) algorithms (including our methods).

## Dependencies

See [requirements](./requirements.txt).

To install this package, please read [Installation guide](./installation.md).

## Features

### Reimplemented & proposed algorithms

Reimplemented methods:

- GREEN [2]
- ICA [4]
- CHROM [5]
- POS [7]
- SPH [8]

Our proposed methods:

- CSC

  _Combination of Simple Chrominance signals (CSC)_ that combines three simple chrominance signals through the alpha-tuning process proposed by [5].

- CHROM-PRN

      CHROM model with PRNet (CHROM-PRN) that

  combines multiple CHROM signals (De Haan and Jeanne, 2013) that are extracted from different facial regions tracked by the dense 3D face alignment algorithm [PRNet](https://github.com/YadiraF/PRNet) [9].

Utilities:

- Wrapper for face detector
- Wrapper for 2D face aligner
- Wrapper for 3D dense face aligner
- Wrapper for skin detector
- Pipeline for HR from ECG
- Pipeline for HR/SNR from rPPG
- Functions for rolling statistics

## To get results

`class HYBRID` includes GREEN, ICA, CHROM, POS, SPH, CSC and CHROM-PRN, so it can avoid processing videos and signals repeatedly.

1. Before using `class HYBRID`, please use following scripts to preprocess data

   - `export_bbox_colormap.py` to export preprocessed bounding box of face from face detection and color map from face alignment.

2. Use following scripts to obtain heart rate for each interval from rPPG and ECG signals.

   - `export_hr_gt.py` to export heart rate for every interval by spectral analysis or peak detection.
   - `export_hr_snr_rppg.py` to export heart rate and snr for every interval by spectral analysis.

3. Use following scripts to get statistics.

   - `export_statics.py` get the summary.

## Datasets

- Our own data (Nordling Lab)

  Not publish yet.

- Other datasets

  - [MAHNOB-HCI](https://mahnob-db.eu/hci-tagging/)
  - [OBF](https://ieeexplore.ieee.org/document/8373836)
  - [VIPL-HR](https://vipl.ict.ac.cn/view_database.php?id=15)
  - [UBFC-RPPG](https://sites.google.com/view/ybenezeth/ubfcrppg)
  - [Hoffman2020](https://data.4tu.nl/articles/dataset/Public_Benchmark_Dataset_for_Testing_rPPG_Algorithm_Performance/12684059)

## License

Apache License Version 2.0

See [LICENSE](./LICENSE).

## Reference

### If you use this package, please cite this paper

[1] ...

### In addition, please cite following

[2] Verkruysse, W., Svaasand, L. O., & Nelson, J. S. (2008). Remote plethysmographic imaging using ambient light. Optics express, 16(26), 21434-21445.

[3] Lewandowska, M., Rumiński, J., Kocejko, T., & Nowak, J. (2011, September). Measuring pulse rate with a webcam—a non-contact method for evaluating cardiac activity. In 2011 federated conference on computer science and information systems (FedCSIS) (pp. 405-410). IEEE.

[4] Poh, M. Z., McDuff, D. J., & Picard, R. W. (2010). Non-contact, automated cardiac pulse measurements using video imaging and blind source separation. Optics express, 18(10), 10762-10774.

[5] De Haan, G., & Jeanne, V. (2013). Robust pulse rate from chrominance-based rPPG. IEEE Transactions on Biomedical Engineering, 60(10), 2878-2886.

[6] Wang, W., Stuijk, S., & De Haan, G. (2015). A novel algorithm for remote photoplethysmography: Spatial subspace rotation. IEEE transactions on biomedical engineering, 63(9), 1974-1984.

[7] Wang, W., den Brinker, A. C., Stuijk, S., & de Haan, G. (2016). Algorithmic principles of remote PPG. IEEE Transactions on Biomedical Engineering, 64(7), 1479-1491.

[8] Pilz, C. (2019). On the vector space in photoplethysmography imaging. In Proceedings of the IEEE International Conference on Computer Vision Workshops (pp. 0-0).

[9] Feng, Y., Wu, F., Shao, X., Wang, Y., & Zhou, X. (2018). Joint 3d face reconstruction and dense alignment with position map regression network. In Proceedings of the European Conference on Computer Vision (ECCV) (pp. 534-551).

## Documentation

See [Documentations](./documentation.md).

## Contact

jack.wang@nordlinglab.org

ccwang.jack@gmail.com

<!--
https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html
https://www.datacamp.com/community/tutorials/docstrings-python
https://google.github.io/styleguide/pyguide.html
-->
