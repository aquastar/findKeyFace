#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml/ml.hpp"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <stdio.h>
#include <dirent.h>

using namespace std;
using namespace cv;

void detectAndDraw(string path, string img,
        CascadeClassifier& cascade, CascadeClassifier& nestedCascade, CascadeClassifier& noseCas, CascadeClassifier& mouthCas,
        double scale);

String cascadeName = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"; //face
String eyeCasName = "/usr/local/share/OpenCV/haarcascades/haarcascade_eye.xml"; //eyes
String noseCasName = "/usr/local/share/OpenCV/haarcascades/haarcascade_mcs_nose.xml"; //nose
String mouthCasName = "/usr/local/share/OpenCV/haarcascades/haarcascade_mcs_mouth.xml"; //mouth

int main(int argc, const char** argv) {
    CascadeClassifier cascade, eyeCas, noseCas, mouthCas;
    double scale = 1.5;

    if (!cascade.load(cascadeName))//从指定的文件目录中加载级联分类器
    {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        return 0;
    }

    if (!eyeCas.load(eyeCasName)) {
        cerr << "WARNING: Could not load classifier cascade for nested objects" << endl;
        return 0;
    }

    if (!noseCas.load(noseCasName))//从指定的文件目录中加载级联分类器
    {
        cerr << "ERROR: Could not load nose classifier cascade" << endl;
        return 0;
    }

    if (!mouthCas.load(mouthCasName)) {
        cerr << "WARNING: Could not load mouth classifier cascade for nested objects" << endl;
        return 0;
    }

    //    if (!image.empty())//读取图片数据不能为空
    //    {
    //        detectAndDraw(image, cascade, nestedCascade, scale);
    //        waitKey(0);
    //    }

    //    Mat image;
    //    image = imread("lena.jpg", 1); //读入lena图片
    //    //image = imread("people_with_hands.png",1);
    //    namedWindow("result", 1); //opencv2.0以后用namedWindow函数会自动销毁窗口


    DIR *dir;
    struct dirent *ent;
    string path = "/home/danny/Documents/jaffe/";
    if ((dir = opendir("/home/danny/Documents/jaffe")) != NULL) {
        clock_t begin1 = clock();
        while ((ent = readdir(dir)) != NULL) {
            if (strcmp(ent->d_name, ".") && strcmp(ent->d_name, "..") && strstr(ent->d_name, "tiff")) {
                printf("Loading image: %s\n", ent->d_name);
                detectAndDraw(path, ent->d_name, cascade, eyeCas, noseCas, mouthCas, scale);
            }
        }
        clock_t end1 = clock();
        double elapsed_secs1 = double(end1 - begin1) / CLOCKS_PER_SEC;
        cout << "time:" << elapsed_secs1 << endl;
        closedir(dir);
    } else {
        perror("");
        return EXIT_FAILURE;
    }

    return 0;
}

void detectAndDraw(Mat& img,
        CascadeClassifier& cascade, CascadeClassifier& nestedCascade,
        double scale) {
    int i = 0;
    double t = 0;
    vector<Rect> faces;
    const static Scalar colors[] = {CV_RGB(0, 0, 255),
        CV_RGB(0, 128, 255),
        CV_RGB(0, 255, 255),
        CV_RGB(0, 255, 0),
        CV_RGB(255, 128, 0),
        CV_RGB(255, 255, 0),
        CV_RGB(255, 0, 0),
        CV_RGB(255, 0, 255)}; //用不同的颜色表示不同的人脸

    Mat gray, smallImg(cvRound(img.rows / scale), cvRound(img.cols / scale), CV_8UC1); //将图片缩小，加快检测速度

    cvtColor(img, gray, CV_BGR2GRAY); //因为用的是类haar特征，所以都是基于灰度图像的，这里要转换成灰度图像
    resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR); //将尺寸缩小到1/scale,用线性插值
    equalizeHist(smallImg, smallImg); //直方图均衡

    t = (double) cvGetTickCount(); //用来计算算法执行时间


    //检测人脸
    //detectMultiScale函数中smallImg表示的是要检测的输入图像为smallImg，faces表示检测到的人脸目标序列，1.1表示
    //每次图像尺寸减小的比例为1.1，2表示每一个目标至少要被检测到3次才算是真的目标(因为周围的像素和不同的窗口大
    //小都可以检测到人脸),CV_HAAR_SCALE_IMAGE表示不是缩放分类器来检测，而是缩放图像，Size(30, 30)为目标的
    //最小最大尺寸
    cascade.detectMultiScale(smallImg, faces,
            1.1, 2, 0
            //|CV_HAAR_FIND_BIGGEST_OBJECT
            //|CV_HAAR_DO_ROUGH_SEARCH
            | CV_HAAR_SCALE_IMAGE
            ,
            Size(30, 30));

    t = (double) cvGetTickCount() - t; //相减为算法执行的时间
    printf("detection time = %g ms\n", t / ((double) cvGetTickFrequency()*1000.));
    for (vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++) {
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        Scalar color = colors[i % 8];
        int radius;
        center.x = cvRound((r->x + r->width * 0.5) * scale); //还原成原来的大小
        center.y = cvRound((r->y + r->height * 0.5) * scale);
        radius = cvRound((r->width + r->height)*0.25 * scale);
        circle(img, center, radius, color, 3, 8, 0);

        //检测人眼，在每幅人脸图上画出人眼
        if (nestedCascade.empty())
            continue;
        smallImgROI = smallImg(*r);

        //和上面的函数功能一样
        nestedCascade.detectMultiScale(smallImgROI, nestedObjects,
                1.1, 2, 0
                //|CV_HAAR_FIND_BIGGEST_OBJECT
                //|CV_HAAR_DO_ROUGH_SEARCH
                //|CV_HAAR_DO_CANNY_PRUNING
                | CV_HAAR_SCALE_IMAGE
                ,
                Size(30, 30));
        for (vector<Rect>::const_iterator nr = nestedObjects.begin(); nr != nestedObjects.end(); nr++) {
            center.x = cvRound((r->x + nr->x + nr->width * 0.5) * scale);
            center.y = cvRound((r->y + nr->y + nr->height * 0.5) * scale);
            radius = cvRound((nr->width + nr->height)*0.25 * scale);
            circle(img, center, radius, color, 3, 8, 0); //将眼睛也画出来，和对应人脸的图形是一样的
        }
    }
    cv::imshow("result", img);
}

void detectAndDraw(string path, string imgname, CascadeClassifier& cascade, CascadeClassifier& eyeCas, CascadeClassifier& noseCas, CascadeClassifier& mouthCas,
        double scale) {
    int i = 0;
    double t = 0;
    vector<Rect> faces;

    // Color set to use
    const static Scalar colors[] = {
        CV_RGB(0, 0, 255),
        CV_RGB(0, 128, 255),
        CV_RGB(0, 255, 255),
        CV_RGB(0, 255, 0),
        CV_RGB(255, 128, 0),
        CV_RGB(255, 255, 0),
        CV_RGB(255, 0, 0),
        CV_RGB(255, 0, 255)
    };

    // Load image as gray scale
    Mat img = imread(path + imgname, CV_LOAD_IMAGE_COLOR);
    Mat gray, smallImg(cvRound(img.rows / scale), cvRound(img.cols / scale), CV_8UC1);
    cvtColor(img, gray, CV_BGR2GRAY);
    
    // Shrink size to speed up
    resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
    // Equalizes the histogram of a gray scale image.
    equalizeHist(smallImg, smallImg);

    // Detect face and statistics performance
    t = (double) cvGetTickCount();
    cascade.detectMultiScale(smallImg, faces,
            1.1, 2, 0
            |CV_HAAR_FIND_BIGGEST_OBJECT
            //|CV_HAAR_DO_ROUGH_SEARCH
            | CV_HAAR_SCALE_IMAGE
            ,
            Size(20, 20));
    t = (double) cvGetTickCount() - t;
    printf("detection time = %g ms\n", t / ((double) cvGetTickFrequency()*1000.));

    // Traverse detected faces in variable faces
    for (vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++) {
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        Scalar color = colors[i % 8];
        int radius;

        // Recover to original size
        center.x = cvRound((r->x + r->width * 0.5) * scale);
        center.y = cvRound((r->y + r->height * 0.5) * scale);
        radius = cvRound((r->width + r->height)*0.25 * scale);

        // Tag face
        circle(img, center, radius, colors[0], 3, 8, 0);
        int face_x = center.x;


        //check eyes
        if (eyeCas.empty())
            continue;
        smallImgROI = smallImg(*r);

        eyeCas.detectMultiScale(smallImgROI, nestedObjects,
                1.1, 1, 0
                //|CV_HAAR_FIND_BIGGEST_OBJECT
                //|CV_HAAR_DO_ROUGH_SEARCH
                //|CV_HAAR_DO_CANNY_PRUNING
                | CV_HAAR_SCALE_IMAGE
                ,
                Size(10, 10));
        int eye_y = 0;
        for (vector<Rect>::const_iterator nr = nestedObjects.begin(); nr != nestedObjects.end(); nr++) {
            center.x = cvRound((r->x + nr->x + nr->width * 0.5) * scale);
            center.y = cvRound((r->y + nr->y + nr->height * 0.5) * scale);
            radius = cvRound((nr->width + nr->height)*0.25 * scale);
            // tag eyes
            circle(img, center, radius, colors[1], 3, 8, 0); //将眼睛也画出来，和对应人脸的图形是一样的
            eye_y += center.y;
        }
        eye_y /= 2;

        //check nose
        if (noseCas.empty())
            continue;

        noseCas.detectMultiScale(smallImgROI, nestedObjects,
                1.1, 1, 0
                //|CV_HAAR_FIND_BIGGEST_OBJECT
                //|CV_HAAR_DO_ROUGH_SEARCH
                //|CV_HAAR_DO_CANNY_PRUNING
                | CV_HAAR_SCALE_IMAGE
                ,
                Size(15, 15));
        int nose_y = 0;
        for (vector<Rect>::const_iterator nr = nestedObjects.begin(); nr != nestedObjects.end(); nr++) {
            if (nr != nestedObjects.begin()) break;
            center.x = cvRound((r->x + nr->x + nr->width * 0.5) * scale);
            center.y = cvRound((r->y + nr->y + nr->height * 0.5) * scale);
            radius = cvRound((nr->width + nr->height)*0.25 * scale);
            circle(img, center, radius, colors[2], 3, 8, 0); //将眼睛也画出来，和对应人脸的图形是一样的
            nose_y = center.y;
        }

        //check mouth
        if (mouthCas.empty())
            continue;
        //        smallImgROI = smallImg(*r);

        mouthCas.detectMultiScale(smallImgROI, nestedObjects,
                1.1, 4, 0
                //|CV_HAAR_FIND_BIGGEST_OBJECT
                //|CV_HAAR_DO_ROUGH_SEARCH
                //|CV_HAAR_DO_CANNY_PRUNING
                | CV_HAAR_SCALE_IMAGE
                ,
                Size(15, 15));
        int mouth_y = 0;
        cout << "nestedObjects:" << nestedObjects.size() << endl;
        for (vector<Rect>::const_iterator nr = nestedObjects.begin(); nr != nestedObjects.end(); nr++) {
            if (nr != nestedObjects.begin()) break;
            center.x = cvRound((r->x + nr->x + nr->width * 0.5) * scale);
            center.y = cvRound((r->y + nr->y + nr->height * 0.5) * scale);
            radius = cvRound((nr->width + nr->height)*0.25 * scale);
            circle(img, center, radius, colors[7], 3, 8, 0); //将眼睛也画出来，和对应人脸的图形是一样的
            mouth_y = center.y;
        }

        nose_y = 165;
        int avg_mouth_nose = nose_y + 7;
        int avg_nose_eye = nose_y - 22;
        line(img, Point(0, avg_mouth_nose), Point(img.cols, avg_mouth_nose), colors[4]);
        line(img, Point(0, avg_nose_eye), Point(img.cols, avg_nose_eye), colors[5]);


        Mat src;
        src = img;
        string str = "~/";
        imwrite(str + "ori.jpg", src);
        imshow("src", src);

        Rect up_rect(0, 90, src.cols, 50);
        Mat up_image = src(up_rect);
        imwrite(str + "corp_up.jpg", up_image);
        imshow("image", up_image);

        Rect down_rect(0, 180, src.cols, 50);
        Mat down_image = src(down_rect);
        imwrite(str + "corp_down.jpg", down_image);
        imshow("image", down_image);

        Mat combine;
        vconcat(up_image, down_image, combine);
        namedWindow("combine", CV_WINDOW_AUTOSIZE);
        imshow("combine", combine);
        imwrite(str + "cc.jpg", combine);


        int limit_x = 128; //int face_x=128;
        for (int row = 0; row < combine.rows; ++row) {
            for (int col = 0; col < combine.cols + limit_x; col++) {
                combine.at<uchar>(row, col + face_x + combine.cols) = (combine.at<uchar>(row, col + face_x + combine.cols) + combine.at<uchar>(row, face_x + combine.cols - col)) / 2;
            }
        }

        Rect final_rect(face_x, 0, combine.cols - face_x, combine.rows);
        Mat final = combine(final_rect);
        string path_half = "/home/danny/Documents/test_half/";
        imwrite(path_half + imgname, final);

    }
    cv::imshow("result", img);
    waitKey(0);
}