# GCN-CMHG
This is an implementation of GCN-CMHG. 
# Introduction
GCN-CMHG is a recommendation algorithm that proposed in our paper "A Graph Convolutional Neural Network for Recommendation Based on Community Detection and the Combination of Multiple Heterogeneous Graphs".   
By combining the Graph Convolutional Neural Network (GCN) layer with the Heterogeneous Partial Adjacent Auxiliary (HPAA) layer constructed using community detection algorithm, GCN-CMHG can help any node in the user-iteminteraction heterogeneous graph obtain distant information with less time loss, and finally improve the recommendation effect.
# Statement
Our code is based on the public code provided in the following paper for academic communication only：  
• X. He, K. Deng, X. Wang, Y. Zhang, and M. Wang, “LightGCN: Simplifying and powering graph convolution network for recommendation,” in SIGKDD, 2020, pp. 639-648.  
• C. Mu, H. Huang, Y. Liu, and J. Luo, “Graph convolutional neural network based on the combination of multiple heterogeneous graphs,” in 2022 IEEE International Conference on Data Mining Workshops, 2022, pp. 724-731.
# Requirements
torch>=1.4.0  
tensorboardX==2.5.0  
scikit-learn==1.0.2  
# Some important parameters
--HFAA: Whether to use the HFAA layer. Optional [True,False]  
--HPAA: Whether to use the HPAA layer. Optional: [True,False]. Note that HPAA is based on HFAA. If you want to use HPAA, both HFAA and HPAA must be True  
--PLNS: Whether to use the PLNS policy. Optional [True,False]  
--AAlayer: Use HFAA layer or HPAA layer at the last k layer, default 1, that is, only after the last layer, the specific use of HFAA or HPAA depends on the --HFAA and --HPAA parameters  
--PLNSlayer: PLNS policy is used at the last k layer. The default is 1, that is, PLNS policy is used only after the last layer  
# How to start
We provide five datasets: [lastfm, douban, Flixster,gowalla, Movies_and_TV, yelp2018]. Specially, Movies_and_TV is the Amazon dataset mentioned in our paper. You can start model training by running main.py
