# Point Set Matching/Registration Benchmark

A list of point set matching/registration resources collected by [Gang Wang](https://sites.google.com/site/2013gwang/). If you find that important resources are not included, please feel free to contact me.

#### Point Set Matching/Registration Methods

###### Point Matching/Registration Methods

- [ICP] A Method for Registration of 3-D Shapes, TPAMI'1992
- [RPM] New algorithms for 2d and 3d point matching: pose estimation and correspondence, PR'1998
- [SC] Shape matching and object recognition using shape contexts, TPAMI'2002
- [RPM-TPS] A new point matching algorithm for non-rigid registration, CVIU'2003
- [KC] A correlation-based approach to robust point set registration, ECCV'2004
- [RPM-PLNS] Robust point matching for nonrigid shapes by preserving local neighborhood structures, TPAMI'2006
- [GF] A new method for the registration of three-dimensional point-sets: The Gaussian fields framework, IVC'2010
- [QPCCP] A quadratic programming based cluster correspondence projection algorithm for fast point matching, CVIU'2010
- [CPD] Point set registration: Coherent point drift, TPAMI'2010
- [GMMReg/L2E-TPS] Robust point set registration using gaussian mixture models, TPAMI'2011
- [RPM-L2E] Robust estimation of nonrigid transformation for point set registration, CVPR'2013
- [GO-ICP] Go-ICP: Solving 3D Registration Efficiently and Globally Optimally, ICCV'2013
- [RPM-VFC] Robust Point Matching via Vector Field Consensus, TIP'2014
- [MoAGReg] A robust non-rigid point set registration method based on asymmetric gaussian representation, CVIU'2015
- [GLMD-TPS] A robust global and local mixture distance based non-rigid point set registration, PR'2015
- [GO-APM] An Efficient Globally Optimal Algorithm for Asymmetric Point Matching, TPAMI'2016
- [PR-GLS] Non-Rigid Point Set Registration by Preserving Global and Local Structures, TIP'2016
- [PM] Probabilistic Model for Robust Affine and Non-rigid Point Set Matching, TPAMI'2016
- [SCGF] Robust Non-rigid Point Set Registration Using Spatially Constrained Gaussian Fields, TIP'2017
- [LPM] Locality preserving matching, IJCV'2018

###### Mismatch Removal Methods

- [RANSAC] Random sample consensus: a paradigm for model fitting with applications to image analysis and automated cartography, 1981
- [MLESAC] MLESAC: A new robust estimator with application to estimating image geometry, CVIU'2000
- [PROSAC] Matching with PROSAC-progressive sample consensus, CVPR'2005
- [ICF] Rejecting mismatches by correspondence function, IJCV'2010
- [GS] Common visual pattern discovery via spatially coherent correspondences, CVPR'2010
- [VFC] A robust method for vector field learning with application to mismatch removing, CVPR'2011
- [DefRANSAC] In defence of RANSAC for outlier rejection in deformable registration, ECCV'2012 
- [TC] Epipolar geometry estimation for wide baseline stereo by Clustering Pairing Consensus, PRL'2014
- [WxBS] WxBS: Wide Baseline Stereo Generalizations, BMVC'2015
- [GFC] Gaussian Field Consensus: A Robust Nonparametric Matching Method for Outlier Rejection, PR'2018

###### Graph Matching Methods

- [SM] A spectral technique for correspondence problems using pairwise constraints, ICCV'2005 [[code]](https://sites.google.com/site/graphmatchingmethods/Code_including_Spectral_Matching.zip?attredirects=0)
- [SM-MAP] Efficient MAP approximation for dense energy functions, ICML'2006 [[code]](https://sites.google.com/site/graphmatchingmethods/Code_including_IPFP_and_L2QP_for_MAP_Inference.zip?attredirects=0)
- [SMAC] Balanced Graph Matching, NIPS'2006 [[code]](http://www.timotheecour.com/software/graph_matching/graph_matching.html)
- [FCGM] Feature correspondence via graph matching: Models and global optimization, ECCV'2008
- [PM] Probabilistic Graph and Hypergraph Matching, CVPR'2008
- [IPFP] An Integer Projected Fixed Point Method for Graph Matching and MAP Inference, NIPS'2009 [[code]](https://sites.google.com/site/graphmatchingmethods/Code_including_IPFP.zip?attredirects=0)
- [RRWM] Reweighted Random Walks for Graph Matching, ECCV'2010
- [FGM] Factorized graph matching, CVPR'2012 [[code]](http://www.f-zhou.com/gm_code.html)
- [DGM] Deformable Graph Matching, CVPR'2013 [[code]](https://github.com/zhfe99/fgm)
- [MS] Progressive mode-seeking on graphs for sparse feature matching, ECCV'2014

###### Other Related Works

- [DM-CNN] Descriptor Matching with Convolutional Neural Networks: a Comparison to SIFT, arXiv'2014
- [DASC] DASC: Robust Dense Descriptor for Multi-modal and Multi-spectral Correspondence Estimation, TPAMI'2017 [[project]](http://diml.yonsei.ac.kr/~srkim/DASC/)
- [MODS] MODS: Fast and Robust Method for Two-View Matching, CVIU'2015 [[project]](http://cmp.felk.cvut.cz/wbs/)
- [Elastic2D3D] Efficient Globally Optimal 2D-to-3D Deformable Shape Matching, CVPR'2016 [[project]](https://vision.in.tum.de/~laehner/Elastic2D3D/)
- [TCDCN] Facial Landmark Detection by Deep Multi-task Learning, ECCV'2014 [[project]](http://mmlab.ie.cuhk.edu.hk/projects/TCDCN.html)


#### Applications

###### Remote Sensing Image Registration
- [LLT] Robust Feature Matching for Remote Sensing Image Registration via Locally Linear Transforming, TGRS'2015
- [GLPM] Guided Locality Preserving Feature Matching for Remote Sensing Image Registration, TGRS'2018

###### Retinal Image Registration
- [DB-ICP] The dual-bootstrap iterative closest point algorithm with application to retinal image registration, TMI'2003
- [GDB-ICP] Registration of Challenging Image Pairs: Initialization, Estimation, and Decision, TPAMI'2007 [[project]](http://www.vision.cs.rpi.edu/gdbicp/)
- [ED-DB-ICP] The edge-driven dual-bootstrap iterative closest point algorithm for registration of multimodal fluorescein angiogram sequence, TMI'2010
- [RPM-SURFPIIFD] Robust point matching method for multimodal retinal image registration, BSPC'2015

###### Visual Homing Navigation
- [GLPM] Visual Homing via Guided Locality Preserving Matching, ICRA'2018

###### HDR Imaging
- [LNR] Locally non-rigid registration for mobile HDR photography, CVPRW'2015

#### Databases

###### Classical databases

- [2D Synthesized Chui-Rangarajan Dataset (deformation, noise, and outliers)](https://www.cise.ufl.edu/~anand/students/chui/research.html)
- [TOSCA](http://tosca.cs.technion.ac.il/book/resources_data.html)
- [Multi-View Stereo Dataset](http://vision.middlebury.edu/mview/data/)
- [Multi-View Stereo for Community Photo Collections](http://grail.cs.washington.edu/projects/mvscpc/)
- [VGG Affine Datasets](http://www.robots.ox.ac.uk/~vgg/data/data-aff.html)
- [Multi-view VGG's Dataset](http://www.robots.ox.ac.uk/~vgg/data1.html)
- [Oxford Building Reconstruction](http://www.robots.ox.ac.uk/~vgg/data2.html)
- [IMM Datasets](http://www.imm.dtu.dk/~aam/datasets/datasets.html)
- [MPEG7 CE Shape-1 Part B](http://www.imageprocessingplace.com/downloads_V3/root_downloads/image_databases/MPEG7_CE-Shape-1_Part_B.zip)
- [Leaf Shapes Database](http://www.imageprocessingplace.com/downloads_V3/root_downloads/image_databases/leaf%20shape%20database/leaf_shapes_downloads.htm)
- [CMU House/Hotel Sequence Images]
- [Generated Matching Dataset](https://lmb.informatik.uni-freiburg.de/resources/datasets/genmatch.en.html)
- [Image Sequences](https://lmb.informatik.uni-freiburg.de/resources/datasets/sequences.en.html) 
- [Mythological creatures 2D](http://tosca.cs.technion.ac.il/book/resources_data.html)
- [Tools 2D](http://tosca.cs.technion.ac.il/book/resources_data.html)
- [Human face](http://tosca.cs.technion.ac.il/book/resources_data.html)
- [Cars & Motorbikes](https://drive.google.com/drive/folders/0B7CshFGxfi_5RVoyYzFyMUhuZnM)
- [DIML Multimodal Benchmark](http://diml.yonsei.ac.kr/~srkim/DASC/DIMLmultimodal/)
- [Street View Dataset](http://3drepresentation.stanford.edu/) [[github]](https://github.com/amir32002/3D_Street_View) [[data]](https://console.cloud.google.com/storage/browser/streetview_image_pose_3d)
- [EVD: Extreme View Dataset](http://cmp.felk.cvut.cz/wbs/datasets/EVD.zip) [[EVD_tentatives]](http://cmp.felk.cvut.cz/wbs/datasets/EVD_tentatives.zip) [[EZD]](http://cmp.felk.cvut.cz/wbs/datasets/ExtremeZoomDataset.zip)
- [WxBS: Wide Baseline Dataset](http://cmp.felk.cvut.cz/wbs/datasets/WxBS-dataset.zip) [[W1BS]](http://cmp.felk.cvut.cz/wbs/datasets/W1BS.tar.gz)
- [Stanford 3D Scanning](http://graphics.stanford.edu/data/3Dscanrep/)
- [MPI FAUST Dataset](http://faust.is.tue.mpg.de/)
- [Mikolajczyk Database](http://lear.inrialpes.fr/people/mikolajczyk/Database/index.html)
- [Panoramic Image Database](http://www.ti.uni-bielefeld.de/html/research/avardy/index.html)
- [PS-Dataset (A Large Dataset for improving Patch Matching, PhotoSynth Dataset for improving local patch Descriptors)](https://github.com/rmitra/PS-Dataset)

###### Other databases

- [FIRE: Fundus Image Registration Dataset](https://www.ics.forth.gr/cvrl/fire/)
- [DRIVE (Retinal Images)](http://www.isi.uu.nl/Research/Databases/DRIVE/)
- [DRIONS-DB (Retinal Images)](http://www.ia.uned.es/~ejcarmona/DRIONS-DB.html)
- [STARE (Retinal Images)](http://cecas.clemson.edu/~ahoover/stare/)
- [Plant Images](https://www.plant-phenotyping.org/datasets-download)
- [MR Prostate Images](https://bigr-xnat.erasmusmc.nl/)
- [CV Images](http://www.cs.cmu.edu/afs/cs/project/cil/www/v-images.html)
- [ETHZ Datasets](http://www.vision.ee.ethz.ch/en/datasets/)
- [CVPapers](http://www.cvpapers.com/datasets.html)
- [YACVID](https://riemenschneider.hayko.at/vision/dataset)