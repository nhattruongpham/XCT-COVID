<h1 align="center">
    XCT-COVID
    <br>
<h1>

<h4 align="center">Standalone program for the XCT-COVID paper</h4>

<p align="center">
<a href=""><img src="https://img.shields.io/github/stars/nhattruongpham/XCT-COVID?" alt="stars"></a>
<a href=""><img src="https://img.shields.io/github/forks/nhattruongpham/XCT-COVID?" alt="forks"></a>
<a href=""><img src="https://img.shields.io/github/license/nhattruongpham/XCT-COVID?" alt="license"></a>
<a href="https://doi.org/10.5281/zenodo.12772023">
    <img src="https://zenodo.org/badge/doi/10.5281/zenodo.12772023.svg" alt="DOI">
</a>
</p>

<p align="center">
  <a href="#introduction">Introduction</a> •
  <a href="#installation">Installation</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#citation">Citation</a> •
  <a href="#acknowledgements">Acknowledgements</a>
</p>

# Introduction
This repository provides the standalone program for XCT-COVID framework. The virtual environment, refined datasets, and final models are available via Zenodo at [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.12772023.svg)](https://doi.org/10.5281/zenodo.12772023)

# Installation
## Software requirements
* Ubuntu 20.04.6 LTS (This source code has been already tested on Ubuntu 20.04.6 LTS with NVIDIA RTX A5000)
* CUDA 11.7 (with GPU suport)
* cuDNN 8.6.0.163 (with GPU support)
* Python 3.10.14

## Cloning this repository
```
git clone https://github.com/nhattruongpham/XCT-COVID.git
```
```
cd XCT-COVID
```

## Creating virtual environment
* Please download the virtual environment (xct_covid.tar.gz) via Zenodo at [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.12772023.svg)](https://doi.org/10.5281/zenodo.12772023)
* Please extract it into the [xct_covid](https://github.com/nhattruongpham/XCT-COVID/tree/main/xct_covid) folder as below:
```
tar -xzf xct_covid.tar.gz -C xct_covid 
```
* Activate the virtual environment as below:
```
source xct_covid/bin/activate
```

# Getting started
## Downloading refined datasets
* Please download the refined datasets via Zenodo at [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.12772023.svg)](https://doi.org/10.5281/zenodo.12772023)
* For the refined **COVIDx_CT_3** independent dataset, please extract and put **COVID** and **non-COVID** folders into the [examples/COVIDx_CT_3](https://github.com/nhattruongpham/XCT-COVID/tree/main/examples/COVIDx_CT_3) folder.
* For the refined **COVID_CT** independent dataset, please extract and put **COVID** and **non-COVID** folders into the [examples/COVID_CT](https://github.com/nhattruongpham/XCT-COVID/tree/main/examples/COVID_CT) folder.
* For the refined **SARS_CoV_2_CT** independent dataset, please extract and put **COVID** and **non-COVID** folders into the [examples/SARS_CoV_2_CT](https://github.com/nhattruongpham/XCT-COVID/tree/main/examples/SARS_CoV_2_CT) folder.

## Downloading final models
* Please download the final models via Zenodo at [![DOI](https://zenodo.org/badge/doi/10.5281/zenodo.12772023.svg)](https://doi.org/10.5281/zenodo.12772023)
* For the **XCT_COVID_L** models, please extract and put all *.pt files into the [models/XCT_COVID_L](https://github.com/nhattruongpham/XCT-COVID/tree/main/models/XCT_COVID_L) folder.
* For the **XCT_COVID_S1** models, please extract and put all *.pt files into the [models/XCT_COVID_S1](https://github.com/nhattruongpham/XCT-COVID/tree/main/models/XCT_COVID_S1) folder.
* For the **XCT_COVID_S2** models, please extract and put all *.pt files into the [models/XCT_COVID_S2](https://github.com/nhattruongpham/XCT-COVID/tree/main/models/XCT_COVID_S2) folder.

## Running prediction
### Usage
```shell
CUDA_VISIBLE_DEVICES=<GPU_NUMBER> python predictor.py 
```
### Example
```shell
CUDA_VISIBLE_DEVICES=0 python predictor.py
```
### Note
* Please modify **dataset_dir**, **model_name**, and **model_path** in the [Configs.py](https://github.com/nhattruongpham/XCT-COVID/tree/main/Configs.py) file for the target model and its corresponding dataset!!!

# Citation
If you use this code or part of it as well as the refined datasets, please cite the following papers:
## Main
```
@article{,
  title={},
  author={},
  journal={},
  volume={},
  number={},
  pages={},
  year={},
  publisher={}
}
```

## References
[1] Gunraj, H., Sabri, A., Koff, D., Wong, A., 2022a. Covid-net ct-2: Enhanced deep neural networks for detection of covid-19 from chest ct images through bigger, more diverse learning. <i>Frontiers in Medicine</i> 8, 3126. <a href="https://doi.org/10.3389/fmed.2021.729287"><img src="https://zenodo.org/badge/doi/10.3389/fmed.2021.729287.svg" alt="DOI"> <br>
</a>
[2] Gunraj, H., Tuinstra, T., Wong, A., 2022b. Covidx ct-3: A large-scale, multinational, open-source benchmark dataset for computer-aided Covid-19 screening from chest CT images. arXiv preprint arXiv:2206.03043. <a href="
https://doi.org/10.48550/arXiv.2206.03043"><img src="https://zenodo.org/badge/doi/10.48550/arXiv.2206.03043.svg" alt="DOI"></a> <br>
[3] Gunraj, H., Wang, L., Wong, A., 2020. Covidnet-ct: A tailored deep convolutional neural network design for detection of covid-19 cases from chest ct images. <i>Frontiers in medicine</i> 7, 608525. <a href="
https://doi.org/10.3389/fmed.2020.608525"><img src="https://zenodo.org/badge/doi/10.3389/fmed.2020.608525.svg" alt="DOI"></a> <br>
[4] Zhang, K., Liu, X.H., Shen, J., Li, Z.H., Sang, Y., Wu, X.W., Zha, Y.F., Liang, W.H., Wang, C.D., Wang, K., Ye, L.S., Gao, M., Zhou, Z.G., Li, L., Wang, J., Yang, Z.H., Cai, H.M., Xu, J., Yang, L., Cai, W.J., Xu, W.Q., Wu, S.X., Zhang, W., Jiang, S.P., Zheng, L.H., Zhang, X., Wang, L., Lu, L., Li, J.M., Yin, H.P., Wang, W., Li, O., Zhang, C., Liang, L., Wu, T., Deng, R.Y., Wei, K., Zhou, Y., Chen, T., Lau, J.Y.N., Fok, M., He, J.X., Lin, T.X., Li, W.M., Wang, G.Y., 2020. Clinically Applicable AI System for Accurate Diagnosis, Quantitative Measurements, and Prognosis of COVID-19 Pneumonia Using Computed Tomography (vol 181, pg 1423, 2020). <i>Cell</i> 182, 1360-1360. <a href="
https://doi.org/10.1016/j.cell.2020.04.045"><img src="https://zenodo.org/badge/doi/10.1016/j.cell.2020.04.045.svg" alt="DOI"></a> <br>
[5] Revel, M.P., Boussouar, S., de Margerie-Mellon, C., Saab, I., Lapotre, T., Mompoint, D., Chassagnon, G., Milon, A., Lederlin, M., Bennani, S., Molière, S., Debray, M.P., Bompard, F., Dangeard, S., Hani, C., Ohana, M., Bommart, S., Jalaber, C., El Hajjam, M., Petit, I., Fournier, L., Khalil, A., Brillet, P.Y., Bellin, M.F., Redheuil, A., Rocher, L., Bousson, V., Rousset, P., Grégory, J., Deux, J.F., Dion, E., Valeyre, D., Porcher, R., Jilet, L., Abdoul, H., 2021. Study of Thoracic CT in COVID-19: The STOIC Project. <i>Radiology</i> 301, E361-E370. <a href="
https://doi.org/10.1148/radiol.2021210384"><img src="https://zenodo.org/badge/doi/10.1148/radiol.2021210384.svg" alt="DOI"></a> <br>
[6] Boulogne, L.H., Lorenz, J., Kienzle, D., Schön, R., Ludwig, K., Lienhart, R., Jegou, S., Li, G., Chen, C., Wang, Q., Shi, D., Maniparambil, M., Müller, D., Mertes, S., Schröter, N., Hellmann, F., Elia, M., Dirks, I., Bossa, M.N., Berenguer, A.D., Mukherjee, T., Vandemeulebroucke, J., Sahli, H., Deligiannis, N., Gonidakis, P., Huynh, N.D., Razzak, I., Bouadjenek, R., Verdicchio, M., Borrelli, P., Aiello, M., Meakin, J.A., Lemm, A., Russ, C., Ionasec, R., Paragios, N., van Ginneken, B., Revel, M.P., 2024. The STOIC2021 COVID-19 AI challenge: Applying reusable training methodologies to private data. <i>Med Image Anal</i> 97. <a href="
https://doi.org/10.1016/j.media.2024.103230"><img src="https://zenodo.org/badge/doi/10.1016/j.media.2024.103230.svg" alt="DOI"></a> <br>
[7] An, P., Xu, S., Harmon, S.A., Turkbey, E.B., Sanford, T.H., Amalou, A., Kassin, M., Varble, N., Blain, M., Anderson, V., Patella, F., Carrafiello, G., Turkbey, B.T., Wood, B.J., 2020. CT Images in COVID-19 [Data set]. The Cancer Imaging Archive. <a href="
https://doi.org/10.7937/tcia.2020.gqry-nc81"><img src="https://zenodo.org/badge/doi/10.7937/tcia.2020.gqry-nc81.svg" alt="DOI"></a> <br>
[8] Clark, K., Vendt, B., Smith, K., Freymann, J., Kirby, J., Koppel, P., Moore, S., Phillips, S., Maffitt, D., Pringle, M., Tarbox, L., Prior, F., 2013. The Cancer Imaging Archive (TCIA): Maintaining and Operating a Public Information Repository. <i>J Digit Imaging</i> 26, 1045-1057. <a href="
https://doi.org/10.1007/s10278-013-9622-7"><img src="https://zenodo.org/badge/doi/10.1007/s10278-013-9622-7.svg" alt="DOI"></a> <br>
[9] Harmon, S.A., Sanford, T.H., Xu, S., Turkbey, E.B., Roth, H., Xu, Z.Y., Yang, D., Myronenko, A., Anderson, V., Amalou, A., Blain, M., Kassin, M., Long, D., Varble, N., Walker, S.M., Bagci, U., Ierardi, A.M., Stellato, E., Plensich, G.G., Franceschelli, G., Girlando, C., Irmici, G., Labella, D., Hammoud, D., Malayeri, A., Jones, E., Summers, R.M., Choyke, P.L., Xu, D.G., Flores, M., Tamura, K., Obinata, H., Mori, H., Patella, F., Cariati, M., Carrafiello, G., An, P., Wood, B.J., Turkbey, B., 2020. Artificial intelligence for the detection of COVID-19 pneumonia on chest CT using multinational datasets. <i>Nat Commun</i> 11. <a href="
https://doi.org/10.1038/s41467-020-17971-2"><img src="https://zenodo.org/badge/doi/10.1038/s41467-020-17971-2.svg" alt="DOI"></a> <br>
[10] Kassin, M.T., Varble, N., Blain, M., Xu, S., Turkbey, E.B., Harmon, S., Yang, D., Xu, Z.Y., Roth, H., Xu, D.G., Flores, M., Amalou, A., Sun, K.Y., Kadri, S., Patella, F., Cariati, M., Scarabelli, A., Stellato, E., Ierardi, A.M., Carrafiello, G., An, P., Turkbey, B., Wood, B.J., 2021. Generalized chest CT and lab curves throughout the course of COVID-19. <i>Sci Rep-Uk</i> 11. <a href="
https://doi.org/10.1038/s41598-021-85694-5"><img src="https://zenodo.org/badge/doi/10.1038/s41598-021-85694-5.svg" alt="DOI"></a> <br>
[11] Jun, M., Cheng, G., Yixin, W., Xingle, A., Jiantao, G., Ziqi, Y., Minqing, Z., Xin, L., Xueyuan, D., Shucheng, C., Hao, W., Sen, M., Xiaoyu, Y., Ziwei, N., Chen, L., Lu, T., Yuntao, Z., Qiongjie, Z., Guoqiang, D., & Jian, H., 2020. COVID-19 CT Lung and Infection Segmentation Dataset (Verson 1.0) [Data set]. Zenodo. <a href="
https://doi.org/10.5281/zenodo.3757476"><img src="https://zenodo.org/badge/doi/10.5281/zenodo.3757476.svg" alt="DOI"></a> <br>
[12] Armato III, S.G., McLennan, G., Bidaut, L., McNitt-Gray, M.F., Meyer, C.R., Reeves, A.P., Zhao, B., Aberle, D.R., Henschke, C.I., Hoffman, E.A., Kazerooni, E.A., MacMahon, H., Van Beek, E.J.R., Yankelevitz, D., Biancardi, A.M., Bland, P.H., Brown, M.S., Engelmann, R.M., Laderach, G.E., Max, D., Pais, R.C., Qing, D.P.Y., Roberts, R.Y., Smith, A.R., Starkey, A., Batra, P., Caligiuri, P., Farooqi, A., Gladish, G.W., Jude, C.M., Munden, R.F., Petkovska, I., Quint, L.E., Schwartz, L.H., Sundaram, B., Dodd, L.E., Fenimore, C., Gur, D., Petrick, N., Freymann, J., Kirby, J., Hughes, B., Casteele, A.V., Gupte, S., Sallam, M., Heath, M.D., Kuhn, M.H., Dharaiya, E., Burns, R., Fryd, D.S., Salganicoff, M., Anand, V., Shreter, U., Vastagh, S., Croft, B.Y., Clarke, L.P., 2015. Data From LIDC-IDRI [Data set]. The Cancer Imaging Archive. <a href="
https://doi.org/10.7937/K9/TCIA.2015.LO9QL9SX"><img src="https://zenodo.org/badge/doi/10.7937/K9/TCIA.2015.LO9QL9SX.svg" alt="DOI"></a> <br>
[13] Armato, S.G., McLennan, G., Bidaut, L., McNitt-Gray, M.F., Meyer, C.R., Reeves, A.P., Zhao, B.S., Aberle, D.R., Henschke, C.I., Hoffman, E.A., Kazerooni, E.A., MacMahon, H., van Beek, E.J.R., Yankelevitz, D., Biancardi, A.M., Bland, P.H., Brown, M.S., Engelmann, R.M., Laderach, G.E., Max, D., Pais, R.C., Qing, D.P.Y., Roberts, R.Y., Smith, A.R., Starkey, A., Batra, P., Caligiuri, P., Farooqi, A., Gladish, G.W., Jude, C.M., Munden, R.F., Petkovska, I., Quint, L.E., Schwartz, L.H., Sundaram, B., Dodd, L.E., Fenimore, C., Gur, D., Petrick, N., Freymann, J., Kirby, J., Hughes, B., Casteele, A.V., Gupte, S., Sallam, M., Heath, M.D., Kuhn, M.H., Dharaiya, E., Burns, R., Fryd, D.S., Salganicoff, M., Anand, V., Shreter, U., Vastagh, S., Croft, B.Y., Clarke, L.P., 2011. The Lung Image Database Consortium, (LIDC) and Image Database Resource Initiative (IDRI): A Completed Reference Database of Lung Nodules on CT Scans. <i>Med Phys</i> 38, 915-931. <a href="
https://doi.org/10.1118/1.3528204"><img src="https://zenodo.org/badge/doi/10.1118/1.3528204.svg" alt="DOI"></a> <br>
[14] Bell, D., Campos, A., Sharma, R., 2020. COVID-19. Radiopaedia.org. <br>
[15] Rahimzadeh, M., Attar, A., Sakhaei, S.M., 2021. A fully automated deep learning-based network for detecting COVID-19 from a new and large lung CT scan dataset. <i>Biomed Signal Proces</i> 68. <a href="
https://doi.org/10.1016/j.bspc.2021.102588"><img src="https://zenodo.org/badge/doi/10.1016/j.bspc.2021.102588.svg" alt="DOI"></a> <br>
[16] Saltz, J., Saltz, M., Prasanna, P., Moffitt, R., Hajagos, J., Bremer, E., Balsamo, J., Kurc, T., 2021. Stony Brook University COVID-19 Positive Cases [Data set]. The Cancer Imaging Archive. <a href="
https://doi.org/10.7937/TCIA.BBAG-2923"><img src="https://zenodo.org/badge/doi/10.7937/TCIA.BBAG-2923.svg" alt="DOI"></a> <br>
[17] Ning, W.S., Lei, S.J., Yang, J.J., Cao, Y.K., Jiang, P.R., Yang, Q.Q., Zhang, J., Wang, X.B., Chen, F.H., Geng, Z., Xiong, L., Zhou, H.M., Guo, Y.P., Zeng, Y.L., Shi, H.S., Wang, L., Xue, Y., Wang, Z., 2020. Open resource of clinical data from patients with pneumonia for the prediction of COVID-19 outcomes via deep learning. <i>Nat Biomed Eng</i> 4, 1197-1207. <a href="
https://doi.org/10.1038/s41551-020-00633-5"><img src="https://zenodo.org/badge/doi/10.1038/s41551-020-00633-5.svg" alt="DOI"></a> <br>
[18] Morozov, S.P., Andreychenko, A.E., Blokhin, I.A., Gelezhe, P.B., Gonchar, A.P., Nikolaev, A.E., Pavlov, N.A., Chernina, V.Y., Gombolevskiy, V.A., 2020. MosMedData: data set of 1110 chest CT scans performed during the COVID-19 epidemic. <i>Digital Diagnostics</i> 1, 49-59. <a href="
https://doi.org/10.17816/dd46826"><img src="https://zenodo.org/badge/doi/10.17816/dd46826.svg" alt="DOI"></a> <br>
[19] Afshar, P., Heidarian, S., Enshaei, N., Naderkhani, F., Rafiee, M.J., Oikonomou, A., Fard, F.B., Samimi, K., Plataniotis, K.N., Mohammadi, A., 2021. COVID-CT-MD, COVID-19 computed tomography scan dataset applicable in machine learning and deep learning. <i>Sci Data</i> 8. <a href="
https://doi.org/10.1038/s41597-021-00900-3"><img src="https://zenodo.org/badge/doi/10.1038/s41597-021-00900-3.svg" alt="DOI"></a>   

# Acknowledgements
The authors also would like to thank the Multi-national NIH Consortium for CT AI in COVID-19. The authors acknowledge the National Cancer Institute and the Foundation for the National Institutes of Health, and their critical role in the creation of the free publicly available LIDC/IDRI Database used in this study.