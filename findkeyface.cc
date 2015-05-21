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

// output base directory and input image folder
const string to_write_dir = "/home/danny/Documents/";
char* input_img_dir = "/home/danny/Documents/jaffe/";

//cascade file by OpenCV
String cascadeName = "/usr/local/share/OpenCV/haarcascades/haarcascade_frontalface_alt.xml"; //face
String eyeCasName = "/usr/local/share/OpenCV/haarcascades/haarcascade_eye.xml"; //eyes
String noseCasName = "/usr/local/share/OpenCV/haarcascades/haarcascade_mcs_nose.xml"; //nose
String mouthCasName = "/usr/local/share/OpenCV/haarcascades/haarcascade_mcs_mouth.xml"; //mouth

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

enum cutpos {
    FACE, RM_NOSE, RM_EYES, RM_MOUTH, RM_NOSE_EYES, RM_NOSE_MOUTH, RM_EYES_MOUTH
};

string cutpos_str[] = {"face", "rm_nose", "rm_eyes", "rm_mouth", "rm_nose_eyes", "rm_nose_mouth", "rm_eyes_mouth"};


void detectAndDraw(string path, string img,
        CascadeClassifier& cascade, CascadeClassifier& nestedCascade, CascadeClassifier& noseCas, CascadeClassifier& mouthCas,
        double scale);
void cut(string path, string img, cutpos cp, string towrite);
string cleandir(string path);


int face_top = 0;
int face_bottom = 0;
int face_center = 0;
int face_cnt = 0;
int eye_y = 0;
int eye_cnt = 0;
int eye_height = 0;
int nose_y = 0;
int nose_cnt = 0;
int nose_height = 0;

int line_nose_eye = 0;
int line_nose_mouth = 0;

int main(int argc, const char** argv) {

    CascadeClassifier cascade, eyeCas, noseCas, mouthCas;
    double scale = 1.5;

    if (!cascade.load(cascadeName) || !eyeCas.load(eyeCasName) || !noseCas.load(noseCasName) || !mouthCas.load(mouthCasName)) {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        return 0;
    }

    DIR *dir, *dird;
    struct dirent *ent, *cutd;
    if ((dir = opendir(input_img_dir)) != NULL) {

        // Get all size info from whole set
        clock_t begin1 = clock();
        while ((ent = readdir(dir)) != NULL) {
            if (strcmp(ent->d_name, ".") && strcmp(ent->d_name, "..") && strstr(ent->d_name, "tiff")) {
                //                printf("Calculating image: %s\n", ent->d_name);
                detectAndDraw(input_img_dir, ent->d_name, cascade, eyeCas, noseCas, mouthCas, scale);
            }
        }
        clock_t end1 = clock();
        double elapsed_secs1 = double(end1 - begin1) / CLOCKS_PER_SEC;
        cout << "time:" << elapsed_secs1 << endl;

        // Calculate proper cut boundary by face, nose and eye sub component size
        face_top /= face_cnt;
        face_bottom /= face_cnt;
        face_center /= face_cnt;

        eye_y /= eye_cnt;
        eye_height /= eye_cnt;

        nose_y /= nose_cnt;
        nose_height /= nose_cnt;

        // FACE, RM_NOSE, RM_EYES, RM_MOUTH, RM_NOSE_EYES, RM_NOSE_MOUTH, RM_EYES_MOUTH
        cutpos cp = RM_NOSE_EYES;
        string towrite = cleandir(cutpos_str[cp]);
        if ((dird = opendir(input_img_dir)) != NULL) {
            // Cut by computed boundary
            while ((cutd = readdir(dird)) != NULL) {
                if (strcmp(cutd->d_name, ".") && strcmp(cutd->d_name, "..") && strstr(cutd->d_name, "tiff")) {
                    printf("Cutting image: %s\n", cutd->d_name);
                    cut(input_img_dir, cutd->d_name, cp, towrite);
                }
            }

            return 0;
        }
        closedir(dir);
    } else {
        perror("");
        return EXIT_FAILURE;
    }

}

string cleandir(string path) {
    int ret = system(("test -d " + to_write_dir + path).c_str());
    if (ret) {
        system(("mkdir " + to_write_dir + path).c_str());
    } else {
        system(("rm -rf " + to_write_dir + path).c_str());
        system(("mkdir " + to_write_dir + path).c_str());
    }
    return (to_write_dir + path + "/").c_str();
}

void detectAndDraw(string path, string imgname, CascadeClassifier& cascade, CascadeClassifier& eyeCas, CascadeClassifier& noseCas, CascadeClassifier& mouthCas,
        double scale) {
    int i = 0;
    double t = 0;
    vector<Rect> faces;

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
            | CV_HAAR_FIND_BIGGEST_OBJECT
            //|CV_HAAR_DO_ROUGH_SEARCH
            | CV_HAAR_SCALE_IMAGE
            ,
            Size(20, 20));
    t = (double) cvGetTickCount() - t;
    //    printf("detection time = %g ms\n", t / ((double) cvGetTickFrequency()*1000.));

    // Traverse detected faces in variable faces
    for (vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++) {
        Mat smallImgROI;
        vector<Rect> nestedObjects;
        Point center;
        int radius;

        // circle on face
        center.x = cvRound((r->x + r->width * 0.5) * scale);
        center.y = cvRound((r->y + r->height * 0.5) * scale);

        face_top += cvRound(r->y * scale);
        face_bottom += cvRound((r->y + r->height) * scale);
        face_center += center.y;
        face_cnt += 1;

        // Tag face using rectangle
        rectangle(img, Point(r->x, r->y), Point(cvRound((r->x + r->width) * scale), cvRound((r->y + r->height) * scale)), colors[0], 2, 8);
        int face_x = center.x;

        // find within found faces
        smallImgROI = smallImg(*r);

        // find nose
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

        for (vector<Rect>::const_iterator nr = nestedObjects.begin(); nr != nestedObjects.end(); nr++) {
            if (nr != nestedObjects.begin()) break;
            center.x = cvRound((r->x + nr->x + nr->width * 0.5) * scale);
            center.y = cvRound((r->y + nr->y + nr->height * 0.5) * scale);
            radius = cvRound((nr->width + nr->height)*0.25 * scale);
            rectangle(img, Point(cvRound((r->x + nr->x) * scale), cvRound((r->y + nr->y) * scale)), Point(cvRound((r->x + nr->x + nr->width) * scale), cvRound((r->y + nr->y + nr->height) * scale)), colors[2], 2, 8);
            nose_y += center.y;
            nose_cnt += 1;
            nose_height += cvRound(nr->height * 0.5 * scale);
        }

        // find eyes
        if (eyeCas.empty())
            continue;

        eyeCas.detectMultiScale(smallImgROI, nestedObjects,
                1.1, 1, 0
                //|CV_HAAR_FIND_BIGGEST_OBJECT
                | CV_HAAR_DO_ROUGH_SEARCH
                //|CV_HAAR_DO_CANNY_PRUNING
                | CV_HAAR_SCALE_IMAGE
                ,
                Size(1, 1));

        int eye_sum = 0;
        for (vector<Rect>::const_iterator nr = nestedObjects.begin(); nr != nestedObjects.end(); nr++) {
            center.x = cvRound((r->x + nr->x + nr->width * 0.5) * scale);
            center.y = cvRound((r->y + nr->y + nr->height * 0.5) * scale);
            radius = cvRound((nr->width + nr->height)*0.25 * scale);
            // tag eyes
            if (center.y < nose_y and eye_sum < 2) {
                rectangle(img, Point(cvRound((r->x + nr->x) * scale), cvRound((r->y + nr->y) * scale)), Point(cvRound((r->x + nr->x + nr->width) * scale), cvRound((r->y + nr->y + nr->height) * scale)), colors[7], 2, 8);
                eye_y += center.y;
                eye_height += cvRound(nr->height * 0.5 * scale);
                eye_cnt += 1;
                eye_sum += 1;
            }
        }

        //        cv::imshow("result", img);
        //        waitKey(0);
    }
}

void cut(string path, string imgname, cutpos cp, string towrite) {
    Mat img = imread(path + imgname, CV_LOAD_IMAGE_COLOR);
    Mat output = img;

    switch (cp) {
        case FACE:
        {
            Rect face_rec(0, face_top, output.cols, face_bottom - face_top);
            Mat face = output(face_rec);
            //            imshow("output", face);
            //            waitKey(0);
            imwrite(towrite + imgname, face);
            break;
        }
        case RM_NOSE:
        {
            Rect up_rect(0, face_top, output.cols, face_center - face_top);
            Mat up_image = output(up_rect);

            Rect down_rect(0, nose_y + nose_height / 2, output.cols, face_bottom - nose_y - nose_height / 2);
            Mat down_image = output(down_rect);

            Mat combine;
            vconcat(up_image, down_image, combine);
            imwrite(towrite + imgname, combine);

            //            imshow("combine", combine);
            //            waitKey(0);
            break;
        }
        case RM_EYES:
        {
            Rect up_rect(0, face_top, output.cols, eye_y - eye_height / 2 - face_top);
            Mat up_image = output(up_rect);

            Rect down_rect(0, eye_y + eye_height / 2, output.cols, face_bottom - eye_y - eye_height / 2);
            Mat down_image = output(down_rect);

            Mat combine;
            vconcat(up_image, down_image, combine);
            imwrite(towrite + imgname, combine);

            //            imshow("combine", combine);
            //            waitKey(0);
            break;
        }
        case RM_MOUTH:
        {
            Rect mouth_rec(0, face_top, output.cols, nose_y + nose_height / 2 - face_top);
            Mat mouth = output(mouth_rec);

            imwrite(towrite + imgname, mouth);

            //            imshow("combine", combine);
            //            waitKey(0);
            break;
        }
        case RM_NOSE_EYES:
        {
            Rect up_rect(0, face_top, output.cols, eye_y - eye_height / 2 - face_top);
            Mat up_image = output(up_rect);

            Rect down_rect(0, nose_y + nose_height / 2, output.cols, face_bottom - nose_y - nose_height / 2);
            Mat down_image = output(down_rect);

            Mat combine;
            vconcat(up_image, down_image, combine);
            imwrite(towrite + imgname, combine);

            //            imshow("combine", combine);
            //            waitKey(0);
            break;
        }
        case RM_NOSE_MOUTH:
        {
            Rect eyes_rec(0, face_top, output.cols, face_center - face_top);
            Mat eyes = output(eyes_rec);

            imwrite(towrite + imgname, eyes);

            //            imshow("combine", combine);
            //            waitKey(0);
            break;
        }
        case RM_EYES_MOUTH:
        {
            Rect nose_rec(0, face_center, output.cols, nose_y + nose_height / 2 - face_center);
            Mat nose = output(nose_rec);

            imwrite(towrite + imgname, nose);

            //            imshow("combine", combine);
            //            waitKey(0);
            break;
        }
        default:
            cout << "Nothing to do" << endl;
            break;
    }

    line(img, Point(0, face_top), Point(img.cols, face_top), colors[1]);
    line(img, Point(0, eye_y), Point(img.cols, eye_y), colors[2]);
    line(img, Point(0, nose_y), Point(img.cols, nose_y), colors[3]);
    line(img, Point(0, face_center), Point(img.cols, face_center), colors[4]);
    line(img, Point(0, face_bottom), Point(img.cols, face_bottom), colors[5]);
    //    cv::imshow("result", img);
    //    waitKey(0);
}