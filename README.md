# Image_Captioning
Image Caption, Image Description, From Text to Image
### Survey
- Automatic Description Generation from Images: A Survey of Models, Datasets, and Evaluation Measures.[[pdf]](https://www.jair.org/media/4900/live-4900-9139-jair.pdf)
- CONNECTING IMAGES AND NATURAL LANGUAGE.2016. [[pdf]](https://pdfs.semanticscholar.org/6271/07c02c2df1366965f11678dd3c4fb14ac9b3.pdf)

### Recurrent Neural Network
* UCLA + Baidu [[Web](http://www.stat.ucla.edu/~junhua.mao/m-RNN.html)] [[Paper-arXiv1](http://arxiv.org/pdf/1410.1090)], [[Paper-arXiv2](http://arxiv.org/pdf/1412.6632)]
  * Junhua Mao, Wei Xu, Yi Yang, Jiang Wang, and Alan L. Yuille, *Explain Images with Multimodal Recurrent Neural Networks*, arXiv:1410.1090
  * Junhua Mao, Wei Xu, Yi Yang, Jiang Wang, Zhiheng Huang, and Alan L. Yuille, *Deep Captioning with Multimodal Recurrent Neural Networks (m-RNN)*, arXiv:1412.6632 / ICLR 2015
* Univ. Toronto [[Paper](http://arxiv.org/pdf/1411.2539)] [[Web demo](http://deeplearning.cs.toronto.edu/i2t)]
  * Ryan Kiros, Ruslan Salakhutdinov, and Richard S. Zemel, *Unifying Visual-Semantic Embeddings with Multimodal Neural Language Models*, arXiv:1411.2539 / TACL 2015
* Berkeley [[Web](http://jeffdonahue.com/lrcn/)] [[Paper](http://arxiv.org/pdf/1411.4389)]
  * Jeff Donahue, Lisa Anne Hendricks, Sergio Guadarrama, Marcus Rohrbach, Subhashini Venugopalan, Kate Saenko, and Trevor Darrell, *Long-term Recurrent Convolutional Networks for Visual Recognition and Description*, arXiv:1411.4389 / CVPR 2015
* Google [[Paper](http://arxiv.org/pdf/1411.4555)]
  * Oriol Vinyals, Alexander Toshev, Samy Bengio, and Dumitru Erhan, *Show and Tell: A Neural Image Caption Generator*, arXiv:1411.4555 / CVPR 2015
* Stanford [[Web]](http://cs.stanford.edu/people/karpathy/deepimagesent/) [[Paper]](http://cs.stanford.edu/people/karpathy/cvpr2015.pdf)
  * Andrej Karpathy and Li Fei-Fei, *Deep Visual-Semantic Alignments for Generating Image Description*, CVPR 2015
* Microsoft [[Paper](http://arxiv.org/pdf/1411.4952)]
  * Hao Fang, Saurabh Gupta, Forrest Iandola, Rupesh Srivastava, Li Deng, Piotr Dollar, Jianfeng Gao, Xiaodong He, Margaret Mitchell, John C. Platt, Lawrence Zitnick, and Geoffrey Zweig, *From Captions to Visual Concepts and Back*, arXiv:1411.4952 / CVPR 2015
* CMU + Microsoft [[Paper-arXiv](http://arxiv.org/pdf/1411.5654)], [[Paper-CVPR](http://www.cs.cmu.edu/~xinleic/papers/cvpr15_rnn.pdf)]
  * Xinlei Chen, and C. Lawrence Zitnick, *Learning a Recurrent Visual Representation for Image Caption Generation*
  * Xinlei Chen, and C. Lawrence Zitnick, *Mind’s Eye: A Recurrent Visual Representation for Image Caption Generation*, CVPR 2015
* Univ. Montreal + Univ. Toronto [[Web](http://kelvinxu.github.io/projects/capgen.html)] [[Paper](http://www.cs.toronto.edu/~zemel/documents/captionAttn.pdf)]
  * Kelvin Xu, Jimmy Lei Ba, Ryan Kiros, Kyunghyun Cho, Aaron Courville, Ruslan Salakhutdinov, Richard S. Zemel, and Yoshua Bengio, *Show, Attend, and Tell: Neural Image Caption Generation with Visual Attention*, arXiv:1502.03044 / ICML 2015
* Idiap + EPFL + Facebook [[Paper](http://arxiv.org/pdf/1502.03671)]
  * Remi Lebret, Pedro O. Pinheiro, and Ronan Collobert, *Phrase-based Image Captioning*, arXiv:1502.03671 / ICML 2015
* UCLA + Baidu [[Paper](http://arxiv.org/pdf/1504.06692)]
  * Junhua Mao, Wei Xu, Yi Yang, Jiang Wang, Zhiheng Huang, and Alan L. Yuille, *Learning like a Child: Fast Novel Visual Concept Learning from Sentence Descriptions of Images*, arXiv:1504.06692
* MS + Berkeley
  * Jacob Devlin, Saurabh Gupta, Ross Girshick, Margaret Mitchell, and C. Lawrence Zitnick, *Exploring Nearest Neighbor Approaches for Image Captioning*, arXiv:1505.04467 (Note: technically not RNN) [[Paper](http://arxiv.org/pdf/1505.04467.pdf)]
  * Jacob Devlin, Hao Cheng, Hao Fang, Saurabh Gupta, Li Deng, Xiaodong He, Geoffrey Zweig, and Margaret Mitchell, *Language Models for Image Captioning: The Quirks and What Works*, arXiv:1505.01809 [[Paper](http://arxiv.org/pdf/1505.01809.pdf)]
* Adelaide [[Paper](http://arxiv.org/pdf/1506.01144.pdf)]
  * Qi Wu, Chunhua Shen, Anton van den Hengel, Lingqiao Liu, and Anthony Dick, *Image Captioning with an Intermediate Attributes Layer*, arXiv:1506.01144
* Tilburg [[Paper](http://arxiv.org/pdf/1506.03694.pdf)]
  * Grzegorz Chrupala, Akos Kadar, and Afra Alishahi, *Learning language through pictures*, arXiv:1506.03694
* Univ. Montreal [[Paper](http://arxiv.org/pdf/1507.01053.pdf)]
  * Kyunghyun Cho, Aaron Courville, and Yoshua Bengio, *Describing Multimedia Content using Attention-based Encoder-Decoder Networks*, arXiv:1507.01053
* Cornell [[Paper](http://arxiv.org/pdf/1508.02091.pdf)]
  * Jack Hessel, Nicolas Savva, and Michael J. Wilber, *Image Representations and New Domains in Neural Image Captioning*, arXiv:1508.02091

### Visual-semantic Embedding Based
- Unifying visual-semantic embeddings with multimodal neural language models. [[pdf](https://arxiv.org/pdf/1411.2539.pdf)]
- Deep visual-semantic alignments for generating image descriptions. CVPR 2015 [[pdf](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Karpathy_Deep_Visual-Semantic_Alignments_2015_CVPR_paper.pdf)]

### Encoder-Decoder
- Show and tell: A neural image caption generator. CVPR, 2015. [[pdf](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Vinyals_Show_and_Tell_2015_CVPR_paper.pdf)]
- Show, attend and tell: Neural image caption generation with visual attention. ICML, 2015. [[pdf](http://proceedings.mlr.press/v37/xuc15.pdf)]
- Deep visual-semantic alignments for generating image descriptions.CVPR 2015. [[pdf](http://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Karpathy_Deep_Visual-Semantic_Alignments_2015_CVPR_paper.pdf)]
- Bottom-up and top-down attention for image captioning and VQA.[[pdf]](https://arxiv.org/abs/1707.07998)
- Convolutional Image Captioning [[Paper]](https://arxiv.org/pdf/1711.09151)

### Reinforcement Learning
- Self-critical Sequence Training for Image Captioning. CVPR, 2017. [[pdf](http://openaccess.thecvf.com/content_cvpr_2017/papers/Rennie_Self-Critical_Sequence_Training_CVPR_2017_paper.pdf)]
- Improved Image Captioning via Policy Gradient optimization of SPIDEr. ICCV, 2017. [[pdf](http://openaccess.thecvf.com/content_ICCV_2017/papers/Liu_Improved_Image_Captioning_ICCV_2017_paper.pdf)] [[video](https://www.youtube.com/watch?v=PCGuC4M038E)]
- Deep Reinforcement Learning-based Image Captioning with Embedding Reward.  CVPR, 2017. [[pdf](http://openaccess.thecvf.com/content_cvpr_2017/papers/Ren_Deep_Reinforcement_Learning-Based_CVPR_2017_paper.pdf)] [[video](https://www.youtube.com/watch?v=iTpImCJRwks)]
- Ranzato, Marc’Aurelio, Sumit Chopra, Michael Auli, and Wojciech Zaremba. “Sequence level training with recurrent neural networks.” arXiv preprint arXiv:1511.06732 (2015).
- Rennie, Steven J., Etienne Marcheret, Youssef Mroueh, Jarret Ross, and Vaibhava Goel. “Self-critical Sequence Training for Image Captioning.” arXiv preprint arXiv:1612.00563 (2016).
- Yu, Lantao, Weinan Zhang, Jun Wang, and Yong Yu. “Seqgan: sequence generative adversarial nets with policy gradient.” arXiv preprint arXiv:1609.05473 (2016).
- Liu, Siqi, Zhenhai Zhu, Ning Ye, Sergio Guadarrama, and Kevin Murphy. “Optimization of image description metrics using policy gradient methods.” arXiv preprint arXiv:1612.00370 (2016).
- Bahdanau, Dzmitry, Philemon Brakel, Kelvin Xu, Anirudh Goyal, Ryan Lowe, Joelle Pineau, Aaron Courville, and Yoshua Bengio. “An actor-critic algorithm for sequence prediction.” arXiv preprint arXiv:1607.07086 (2016).
- Li, Jiwei, Will Monroe, Alan Ritter, Michel Galley, Jianfeng Gao, and Dan Jurafsky. “Deep reinforcement learning for dialogue generation.” arXiv preprint arXiv:1606.01541 (2016).

### Others
- ICCV-2017 Scene Graph Generation from Objects, Phrases and Caption Regions
- ICCV-2017 An Empirical Study of Language CNN for Image Captioning
- ICCV-2017 Show, Adapt and Tell: Adversarial Training of Cross-domain Image Captioner
- CVPR-2017 Skeleton Key: Image Captioning by Skeleton-Attribute Decomposition
- CVPR-2017 Knowing When to Look: Adaptive Attention via A Visual Sentinel for Image Captioning 
- CVPR-2017 SCA-CNN: Spatial and Channel-wise Attention in Convolutional Networks for Image Captioning 
- Boosting Image Captioning with Attributes
- What Value Do Explicit High Level Concepts Have in Vision to Language Problems?
- Image Caption Generation with Text-Conditional Semantic Attention
- Guiding Long-Short Term Memory for Image Caption Generation
- CVPR-2015 From captions to visual concepts and back
- Attend and Tell: Neural Image Caption Generation with Visual Attention
- Deep Captioning with Multimodal Recurrent Neural Networks
- Explain Images with Multimodal Recurrent Neural Networks 
- Deep Visual-Semantic Alignments for Generating Image Descriptions
-----
## Video Captioning
* Berkeley [[Web](http://jeffdonahue.com/lrcn/)] [[Paper](http://arxiv.org/pdf/1411.4389)]
  * Jeff Donahue, Lisa Anne Hendricks, Sergio Guadarrama, Marcus Rohrbach, Subhashini Venugopalan, Kate Saenko, and Trevor Darrell, *Long-term Recurrent Convolutional Networks for Visual Recognition and Description*, arXiv:1411.4389 / CVPR 2015
* UT Austin + UML + Berkeley [[Paper](http://arxiv.org/pdf/1412.4729)]
  * Subhashini Venugopalan, Huijuan Xu, Jeff Donahue, Marcus Rohrbach, Raymond Mooney, and Kate Saenko, *Translating Videos to Natural Language Using Deep Recurrent Neural Networks*, arXiv:1412.4729
* Microsoft [[Paper](http://arxiv.org/pdf/1505.01861)]
  * Yingwei Pan, Tao Mei, Ting Yao, Houqiang Li, and Yong Rui, *Joint Modeling Embedding and Translation to Bridge Video and Language*, arXiv:1505.01861
* UT Austin + Berkeley + UML [[Paper](http://arxiv.org/pdf/1505.00487)]
  * Subhashini Venugopalan, Marcus Rohrbach, Jeff Donahue, Raymond Mooney, Trevor Darrell, and Kate Saenko, *Sequence to Sequence--Video to Text*, arXiv:1505.00487
* Univ. Montreal + Univ. Sherbrooke [[Paper](http://arxiv.org/pdf/1502.08029.pdf)]
  * Li Yao, Atousa Torabi, Kyunghyun Cho, Nicolas Ballas, Christopher Pal, Hugo Larochelle, and Aaron Courville, *Describing Videos by Exploiting Temporal Structure*, arXiv:1502.08029
* MPI + Berkeley [[Paper](http://arxiv.org/pdf/1506.01698.pdf)]
  * Anna Rohrbach, Marcus Rohrbach, and Bernt Schiele, *The Long-Short Story of Movie Description*, arXiv:1506.01698
* Univ. Toronto + MIT [[Paper](http://arxiv.org/pdf/1506.06724.pdf)]
  * Yukun Zhu, Ryan Kiros, Richard Zemel, Ruslan Salakhutdinov, Raquel Urtasun, Antonio Torralba, and Sanja Fidler, *Aligning Books and Movies: Towards Story-like Visual Explanations by Watching Movies and Reading Books*, arXiv:1506.06724
* Univ. Montreal [[Paper](http://arxiv.org/pdf/1507.01053.pdf)]
  * Kyunghyun Cho, Aaron Courville, and Yoshua Bengio, *Describing Multimedia Content using Attention-based Encoder-Decoder Networks*, arXiv:1507.01053
* Zhejiang Univ. + UTS [[Paper](http://arxiv.org/abs/1511.03476)]
  * Pingbo Pan, Zhongwen Xu, Yi Yang, Fei Wu, Yueting Zhuang, *Hierarchical Recurrent Neural Encoder for Video Representation with Application to Captioning*, arXiv:1511.03476
* Univ. Montreal + NYU + IBM [[Paper](http://arxiv.org/pdf/1511.04590.pdf)]
  * Li Yao, Nicolas Ballas, Kyunghyun Cho, John R. Smith, and Yoshua Bengio, *Empirical performance upper bounds for image and video captioning*, arXiv:1511.04590

-----

## From Text to Image


### RNN

- Pixel recurrent neural networks. arXiv preprint arXiv:1601.06759 (2016). [[pdf](https://arxiv.org/pdf/1601.06759v3.pdf)]
- DRAW: A recurrent neural network for image generation. arXiv preprint arXiv:1502.04623 (2015). [[pdf](https://arxiv.org/pdf/1502.04623v2.pdf)]
- Generating images from captions with attention. arXiv preprint arXiv:1511.02793 (2015). [[pdf](https://arxiv.org/pdf/1511.02793v2.pdf)]


### GAN
- Conditional generative adversarial nets for convolutional face generation. Class Project for Stanford CS231N: Convolutional Neural Networks for Visual Recognition, Winter semester 2014.5 (2014): 2. [[pdf](https://pdfs.semanticscholar.org/42f6/f5454dda99d8989f9814989efd50fe807ee8.pdf)]

- Generative adversarial text to image synthesis. ICML, 2016. [[pdf](http://proceedings.mlr.press/v48/reed16.pdf)] [[Supplementary](http://proceedings.mlr.press/v48/reed16-supp.zip)]

- Learning what and where to draw. NIPS, 2016. [[pdf](http://papers.nips.cc/paper/6111-learning-what-and-where-to-draw.pdf)]

- StackGAN: Text to Photo-realistic Image Synthesis with Stacked Generative Adversarial Networks.    ICCV, 2017. [[pdf](http://openaccess.thecvf.com/content_ICCV_2017/papers/Zhang_StackGAN_Text_to_ICCV_2017_paper.pdf)] [[video](https://www.youtube.com/watch?v=crI5K4RCZws)]

- StackGAN++: Realistic Image Synthesis with Stacked Generative Adversarial Networks. arXiv preprint arXiv:1710.10916, 2017. [[pdf](https://arxiv.org/pdf/1710.10916v1.pdf)]

- AttnGAN: Fine-Grained Text to Image Generation with Attentional Generative Adversarial Networks. arXiv preprint arXiv:1711.10485 (2017). [[pdf](https://arxiv.org/abs/1711.10485)]

- Semantic Image Synthesis via Adversarial Learning. ICCV, 2017. [[pdf](http://openaccess.thecvf.com/content_ICCV_2017/papers/Dong_Semantic_Image_Synthesis_ICCV_2017_paper.pdf)] [[Supplementary](http://openaccess.thecvf.com/content_ICCV_2017/supplemental/Dong_Semantic_Image_Synthesis_ICCV_2017_supplemental.pdf)]
- TAC-GAN - Text Conditioned Auxiliary Classifier Generative Adversarial Network 2017. [[pdf](https://arxiv.org/abs/1703.06412)]
- Plug & play generative networks: Conditional iterative generation of images in latent space. CVPR, 2017. [[pdf](http://openaccess.thecvf.com/content_cvpr_2017/papers/Nguyen_Plug__Play_CVPR_2017_paper.pdf)] [[Supplementary](http://openaccess.thecvf.com/content_cvpr_2017/supplemental/Nguyen_Plug__Play_2017_CVPR_supplemental.pdf)]
