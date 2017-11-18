# Face Recognition For The Happy House

* Papers
  * [FaceNet](https://arxiv.org/pdf/1503.03832.pdf)
    * It Learns  to encode a face into a vector of 128 numbers
    * Comparing 2 such vectors with a distance `d`, `d(img1, img2)`
  * [DeepFace](https://research.fb.com/wp-content/uploads/2016/11/deepface-closing-the-gap-to-human-level-performance-in-face-verification.pdf)
* **Face Verification**
  * Is this the claimed person?
  * Eg. Airport passport scanning systems
  * 1:1 matching problem
* **Face Recognition**
  * Who is this person?
  * Eg. Baidu Employees Recognition system
  * 1:K matching problem

## Objectives

* Implement the triplet loss function
* Use a pretrained model to map face images into 128-dimensional encodings
* Use these encodings to perform face verification and face recognition

## Improvements

* Put more images of each person (under different lighting conditions, taken on different days, etc.) into the database. Then given a new image, compare the new face to multiple pictures of the person. This would increase accuracy.
* Crop the images to just contain the face, and less of the "border" region around the face. This preprocessing removes some of the irrelevant pixels around the face, and also makes the algorithm more robust.
