
# Ladies and Gentlemen of GitHub, welcome.

###### `MAIS202` is an *introductory bootcamp to ML* that taught me the basics such as CNNs, Linear Regression, Naive Bayes and many more.

As a complementary to the bootcamp, here is a project I wanted to tackle: Emotion Detection in Faces using ML.

The goal is to identify what emotion a person is feeling by looking at a static pitcure or a real-time video of them. 

###### STEPS:
1) Started by reading the academic paper [DeepEmotion2019](https://arxiv.org/pdf/1902.01019.pdf),
2) Created my CNN and got stuck on how to convert numbers to images, how to localize faces, and how to make a live webcame demo.
3) Found out two implementation that went over the correct code
- [OmarSayed7's implementation of DeepEmotion2019](https://github.com/omarsayed7/Deep-Emotion)
- [DeepLearning_by_PhDScholar](https://www.youtube.com/watch?v=yN7qfBhfGqs&ab_channel=DeepLearning_by_PhDScholar)
4) Implemented and merged the two after understanding every class and function used. 

I followed [OmarSayed7's implementation of DeepEmotion2019](https://github.com/omarsayed7/Deep-Emotion) and built my understanding of this whole project thanks to him.
Then integrated the live demo feature from [DeepLearning_by_PhDScholar](https://www.youtube.com/watch?v=yN7qfBhfGqs&ab_channel=DeepLearning_by_PhDScholar), however when I test it on, the face detection square doesn't localize my face.

__*This project is for educational purposes only.*__

###### Results:

![Alt text](/Users/kassa/Desktop/Poster_MAIS202.png?raw=true "Descriptive Poster With Results")

###### Code:
Database: *[FER-2013](https://www.kaggle.com/msambare/fer2013)*
Main: *Dataset setup, training loop, and section to test on any image you input.* 
Deep_emotion: *CNN structure with localization function.*
Data_loaders: *DataLoader PlainDataset*
Generate_data: *Convert data from numbers to images using PIL* 
Visualize: *Test data and live demo on webcam.*


