# Learning-Loop-ML-Edition
Welcome to "Learning Loop : ML Edition"!

In this repository, Each day, I'll be sharing my discoveries, insights, and lessons learned as I delve deeper into the fascinating realm of AI.

Happy Learning!

## Day 1 - 7/20/2023
- I got introduced to Multihead Attention, its mechanism in general. In short it is for loop over self attention mechanism ( which I learned yesterday ). Furthermore, I got intuition on Transformer Network in detail. The transformer network works by processing input data through layers of self-attention mechanisms and feed-forward neural networks. It utilizes self-attention to capture contextual information and dependencies within the input sequence, enabling efficient parallel processing. 

## Day 2 - 7/21/2023
- Today was the day of assignment. I worked on Transformer Architecture assignment where in the assignment I implemented the components of the transformers model in TensorFlow 2.4. The assignment was the great source to learn more indepth on Transformer where I implemented Position Encoding, Masking both padding and Look-ahead, Self- Attention, Encoder, Decoder and lastly prepared the model. 

## Day 3 - 7/22/2023
-  I deepened my knowledge on Transformer. Today I delved into the pre-processing methods applied to raw text before passing it to the encoder and decoder blocks of the transformer architecture. I learned that difference between two vectors seperated by k position is always constant and Positional Encoding (PE) can affect Word Embeddings if relative weights of PE is not small. 

## Day 4 - 7/23/2023
- Today I optimized Transformer model to perform Name Entity Recognition on dataset of resumes where I used tokenizers and pretrained model form Hugging Face Library and Fine-tuned a pre-trained transformer model for Named-Entity Recognition.

## Day 5 - 7/24/2023
- Today I was introducted to Generative Adversarial Networks (GANs). GANs are used to generate incredibly realistic images/ music. I learned what Generative models are and learned the difference of Discriminative and generative Model. Furthermore I got intuitions on types of Generative models and how they work: VAEs and GANs. Lastly I learned Real life GANs applications and big techs that are planning or using this cool technique to enhance.

## Day 6 - 7/25/2023
- Today I learned the intuitions behind GANs and learned about the goals of Generator, discriminator and competition between them. Furthermore I learned how discriminator classifier distinguish between classes of Real and Fake labels.

## Day 7 - 7/26/2023
-  Today I learned briefly about how Generator generates examples of the class with noise vector and input. The I learned about how Binary Cross Entropy funciton is used for training GANs and lastly Putting It All Together learned about the architecture of GANs as whole. 

## Day 8 - 7/27/2023
- I was introducted to functionality of Pytorch and cleared few question like why pytorch. Next, I implemeted tensor's operations like addition, multiplication using pytorch. Furthermore I implemeted simple Neural Network using pytorch's nn library. This was all done in guided assignment which introduced pytorch. 

## Day 9 - 7/28/2023
- Today I worked on a guided assignment where I created my first generative adversarial network (GAN) for the course! Specifically, I built and trained a GAN that can generate hand-written images of digits (0-9).

## Day 10 - 7/29/2023
- Today I learned about Activation functions (I knew them already so more like a revision), how they work, common activation functions and also about Batch Normalization and how it works.

## Day 11 - 7/30/2023
- Today I learned more about GANs. I was able to revise things on convolutions like pooling, strides, padding etc and futher more learned about Devonvolution which was the highlight of the day. I also dived deeped on Checkerboard pattern problems while upscaling.

## Day 12 - 7/31/2023
- Today I read research paper on Deep Convolutional GANs (DCGAN). I created another GANs (DCGAN) using the MNIST dataset. I also implemented a DCGAN in the course.

## Day 13 - 8/1/2023
- Today I checked out the research paper of DCGAN. It was one of the first research paper I completed. I was not very easy for me the grasp everything but I sure did learned alot on Research paper in general and more about DCGAN in general. I also implemented the code of TGAN which is used for Video Generation.

## Day 14 - 8/2/2023
- Today I hopped on the Problems in BCE Loss like Mode Collapse, Vanishing Gradient and how it impacts on instability of the GANs model. Not only did I learned about the problem, I got intuition on the cost function which were more stable like Earth Mover's Distance (EMD) and Wassertein Loss (W- loss).

## Day 15 - 8/3/2023
- Today I continued on yesterday's topic. W - Loss which was one of the cost function which was less prone to Vanishing Gradient and Mode Collapse was also not perfect like everything else. For it to work well it had few condtions ie 1-L continuity Enforcement. Today I learned about it and lastly code implemented SNGAN.

## Day 16 - 8/4/2023
- Yeahh! Exam is finished and I will be learning alot next two day. Talking about learning, I slept and rest alot today but I sure did learned few things. I learned about WGAN and code implemeted it. I learned about W-Loss few days ago. I guess it was extention to it. WGAN solves instability issues. I implemented it with Gradient Penalty which prevents Mode Collapse better. 

## Day 17 - 8/5/2023
- After implementing the WGAN yesterday, I checked out the research paper of the WGAN which was pretty hard for me to grasp. The maths there we too much so instead learned in detail using youtube. Also I worked on chatbot project using RASA.

## Day 18 - 8/6/2023
- Today I got intuitions on Conditional Generation which is a technique to ask model which data to generate. Here we concatenate the One-Hot encoded vector of the class labels with noise vector and fed to generate as input.

## Day 19 - 8/7/2023
- Today worked on programming assignment. The assignment was of Conditional GAN where I made a conditional GAN in order to generate hand-writtern images of digits, conditioned on the digit to be generated (the class vector).

## Day 20 - 8/8/2023
- Today's day was more focused on theory. I learned about Contrallable Generation, Vector Algebra in the Z-space, Challenges with controllable Generation, Classifier Gradients, Disentanglement.

## Day 21 - 8/9/2023
- Hectic day. Today I worked on programming exercise where I implemented the concept of Controllable Generation. It was a guided assignment.

## Day 22 - 8/10/2023
- I overviewed the course 2 of new course (Buikd Better GANs) where I also learned why evaluating the performance can be hard. I understood about the challenges of evaluating GANs.

## Day 23 - 8/11/2023
- Today I learned on Comparing Images where I learned about getting the distance between images which adds feature level. Also learned how pixel distance is simple and too sensitive and unreliable. Furthermore to it, I learned more reliable technique which is Feature distance.

## Day 24 - 8/12/2023
- Extending the yesterday's topic of Feature distance, I learned about Feature Extraction where I learned how to extract features using classifiers and compute feature distance between those images.

## Day 25 - 8/13/2023
- I learned about Inception- v3 network which was  trained on image.net. I learned how to extract feature embeddings from it and then compare those embedding.

## Day 26 - 8/14/2023
- Today I learned about Frechet Inception Distance (FID) which is a metric for measuring the feature distance between real and generated images. FID uses inception model and miltivariate normal frechet distance.

## Day 27 - 8/15/2023
- Today I learned anohter way to calculate the distance between reals and fakes using the inception v3 model i.e. Inception Score. Inception score tries to capture fidelity and diversity and not as good as FID.

## Day 28 - 8/16/2023
- Today I deepened my knowledge about why FID has overtaken Inception Score with the help of Research paper by Barrat and Sharma. I learned that FID is more sensitive to the similarity between the generated and real iamges.

## Day 29 - 8/17/2023
- Today I learned about the main evaluation methods of GANS. I learned few tricks in evaluation GANs. I learned Truncation Trick where we truncate the tail to tune for each fidelity and diversity trade off. Furthermore I learned about HYPE and Human Evaluation.

## Day 30 - 8/18/2023
- Diving deep into the world of HYPE, I learned about HYPE-Time which measures how long a human need to see an image before classifying real or fake and HYPE-Infinity which measure how often human make mistake judging images. 

## Day 31 - 8/19/2023
- Into the Evaluation of GANs as topic, I learned about precision and Recall in GANs where I learned about Precision being fidelity (How well the model classifies fake over all fakes) and Recall being diversity (how well the model is at classifying real over all reals). I learned how truncation trick help to improve precision.

## Day 32 - 8/20/2023
- Today I worked on programming assignment about Frechet Inception Distance (FID) which is a metric for measuring feature distance between real and generated images.

## Day 33 - 8/21/2023
- Diving deep into GANS. I learned about the disadvantages of GANS like lack of good metric, No density estimation, and not having straight forward inverting from generated image to noise. And furthermore I learned about alternatives to GANs like VAEs and learned surfacely how they work.

## Day 34 - 8/22/2023
- I checked out the alternative to GANS called VAEs which has exact opposite disadvanteges and advantages as GANs. I understood how they work and basic component of VAEs.

## Day 35 - 8/23/2023
- I read an article titled Machine Bias by ProPublica which talks about a software to predict future criminals and its biased against black.

## Day 36 - 8/24/2023
- Today I learned about Fairness in Machine learning model. Starting with its definitoin which I found it has no concrete definition and is still a vast area of research. Furthermore I learned about many ways how bias can enter in model. Continuing this knowledge I saw an example article where PULSE which is am image upscaling model learned bias and upsampled the obama as white.
  
## Day 37 - 8/25/2023
- Today I explored a way to identifying some baises of a GAN using a classifier, in a way that's well suited for attempting to make a model independent of an input. It was a programming assignment,

## Day 38 - 8/26/2023
- Today I worked on a/ project: Movie recommendation system which takes input movie and recommend similar movies based on content. Today I preprocessed the data and created model using nltk library and bagofwords as vectorization technique.

## Day 39 - 8/27/2023
- Working further on the same project I was able to built a webapp using streamlit. The web app takes the input as the movie name and outputs similar movies with its poster. 

## Day 40 - 8/28/2023
- Learned about Improvements in GANs and I am learned about state of art GAN ie StyleGAN which produce high quality images, increased diversity and gives more control over features. I got overview of StyleGAN about its architecture and it's working in short.

## Day 41 - 8/29/2023
- Diving deep into the StyleGAN architecture, under stood how Progressive Growing lets GAN generate image from lower resolution to higher. Secondly I learned about Noise Mapping network which is used to make the noise vector less entangled and Finally AdaIN which is Normalization technique based on Adaptive Style necessarily scaling and shifting. 

## Day 42 - 8/30/2023
- Today I worked on programming assignment where I implemented various components of StyleGAN, including truncation trick, the mapping layer, noise injection, adaptive instance normalization (AdaIn( and progressive growing.

## Day 43 - 8/31/2023
- Today I got overview of the components of StyleGAN2 like Demodulation which is the modification of AdaIn normalization, saw how Progressive Growing was replaced with better neural network architecture with skip and residual connetion and Path length regularization which is for the smooth mapping.

## Day 44 - 9/1/2023
- Worked on project of Facial recognition with Siamese Network. I completed the first part where I created a conda environment, installed and imported required packagaes, setup the GPU for the jupyter network by downloading cuba toolkit.

## Day 45 - 9/2/2023/
- Furthermore on Facial recognition Project, today I completed the part where I collected anchor, positive and negative image for the model. Next I preprocess the images created positve and negative samples and load those data into TF Dataloader.

## Day 46 - 9/3/2023
- Sticking with the project I was able to build a Siamese Neural Network with similar architecture in the research paper. After building the model I trained and test the model using 2 way pipeline structure.

## Day 47 - 9/4/2023
- Finally it was time when we make facial recognition predictions. I used open cv to make the prediction real-time. The result was pretty bad but it is the journey which matters than the destination. I learned alot working in the project and from now on I will be doing these project quite often.

## Day 48 - 9/5/2023
- I worked on project to train YOLOv8 object detection on custom data. I used simple image downloader to download the data. After collecting the data I labelled each image using labelImg and used them to train the yolov8 model on commannd Interface. The model could not label maybe the problem was on the data or else. But I sure learned alot of process and will be training the model again.

## Day 49 - 9/6/2023
-- Hectic day but still manage to work on a simple project ie Text Generation using Hugging Face Transformer Model. I used GPT-2 for the blog post generation which takes short sentence as input and outputs a small paragraph of text related to that topic. 

## Day 50 - 9/7/2023
- Working further more on YOLOv8 object detection today I used Oidv4 toolkit to download images from Open Image Dataset of Google, used CVAT to annotate the data and train the model in local environment.

## Day 51 - 9/8/2023
- Today I looked at different applications of GANs mostly focusing on the topic of Data Augmentation. In Data Augmentation I learned how it helps us to use more realistic fake data for the ml model to improve its accuracy.

## Day 52 - 9/9/2023
- Futhermore into the Data Augmentation using GANs, I learned Pros and Cons of Data Augmentation like pros being it better than hand crafted syntetic examples, Generate more labeled examples and Improve the model generalization. Cons being limited Diversity, not useful when overfit to real data.

## Day 53 - 9/10/2023
- I surfacely explored the RandAugment paper which shows that data augmentation has the potential t osignificantly improve the generalization of deep learning model. 

## Day 54 - 9/11/2023
- Explored the fundamental of Matrix Factorization as matrix are everywhere in machine learning. I learned about LR, QR and Cholesky method and importance of Matrix Factorization, where it is used in machine learning surfacely.

## Day 55 - 9/12/2023
- Today I implemented a Generative Teaching Network (GTN) which was first introduced in 2019. Essentially GTN is composed of a generator (ie teacher which produces synthetic data, and a student which is trained on this data for some task. The main difference with GANs being GTN working cooperatively opposed to adversarilly.

## Day 56 - 9/13/2023
- Learned what Image to Image Translation is. Got overview of Pix2Pix GAN and its components. I learned that Pix2Pix Generator is based on U-Net Architecture which takes Image as input and give Image as output. Then comes Discriminator (PatchGAN) which takes the concatenated output from U-Net and the Real Input and output classification matrix and compares it with Real/ fake matrix with additional pixel distance loss term.

## Day 57 - 9/14/2023
- I summarized the learning of Pix2Pix then learned improvements and extension of Pix2Pix for image to image translation like Higher resolution images and image editing. Pix2PixHD, GauGAN, are successors of Pix2Pix.

## Day 58 - 9/15/2023
- Worked in the project where I learned how to use Langchain. It was a beginner-friendly project where I built a webapp using streamlit whuc turn as image into an audio story. I used Langchain framework, and Hugging face models.

## Day 59 - 9/16/2023
- I completed my programming assignment of U-net. I implemeted  a U-Net for a biomedical imaging segmentation task.

## Day 60 - 9/17/2023
- Today I worked on programming assignment of a generative model based on the paper Image-to-Image translation with conditional adversarial netowrks by Isola also know as Pix2Pix and trained a mo/del that can convert aerial satellite imagery to map routes.
## Day 61 - 9/18/2023
- Learned about Super-Resolution GAN (SRGAN), a GAN that enhances the resolution of images by 4x, proposed in Photo-Realistic Single Image Super-Resolution Using a Generative Adversarial Network (Ledig et al. 2017). Also implemented the architecture and training in full and be able to train it on the CIFAR dataset.

## Day 62 - 9/19/2023
- Understood how unpaired image-to-image translation differs from paired translation and took a surface overview of how CycleGAN work and its components like two GANs, cycle consistency.

## Day 63 - 9/20/2023
- Understood the concept of Least Squares Loss and Identity loss in GANs and learned putting every thing all together. Alos learned how and where we can apply Cycle GAN and its variants.

## Day 64 - 9/21/2023
- Worked on Programming Assignment where I wirite generative model based on the paper Unpaired Image to Image Translation using Cycle Consistent Adversarial Networks ie CycleGAN. 

## Day 65 - 9/22/2023
- Covered a quick introduction to Machine Learning Production system focusing on their requirements and challenges by studying a Case Study of speech recognition project.

## Day 66- 9/23/2023
- I learned Key challenges while deployment mainly being data and concept drift, Stats issues as well and software issues. Next I learned few common Deployment patterrns.

## Day 67 - 9/24/2023
- Learned how to monitor a system mostly focusing on software metrics, input metrics and output metrics. And futher more I learned about pipeline monitoring.
  
## Day 68 - 9/25/2023
- While learning about model strategies and ey challenges in model deployment, I learned about why low average error isn't good enough, and how to establish a baseline and got few tips for getting started with selection and training of a model.

## Day 69 - 9/26/2023
- Covered Error analysis and strategies to work with different data types. I learned how to handle Skewed datasets and audit performance with example. 

## Day 70 - 9/27/2023
- Learned about Data centric AI development and how data augmentation can be used and which type of data to focus on more with example and using analogy of rubber band. Next I learned about Checklist of being a data that can be added to dataset.

## Day 71 - 9/28/2023
- Learned about a rare case of model performance being hurt by adding data. Then I learned more on structured data, adding feature in it which is similar to data augmentation when it comes to structured data.

## Day 72 - 9/29/2023
- Learned about how we can track the experiments and its importance. Then I learned about the importance of good data and criteria of good data. 

## Day - 73 - 9/30/2023
- Worked on lab and observed how data intricacies affect the outcome of models, applied data changes without addressing the model where I used a single model throughout: a simple Convolutional Neural Network (CNN). 

## Day 74 - 10/1/2023
- Understood why data definition is quite harder than anyone might think then learned about ambiguity while labeling data with examples.

## Day 75 - 10/2/2023
- Learned the major types of data problems with respect to different types of data interms of sizes and nature. Then I learned how much clean data is important futhermore I learned how to handle unlabeled data for structured and unstructured data.
