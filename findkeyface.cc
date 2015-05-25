#include "opencv2/objdetect/objdetect.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/ml/ml.hpp"
#include <iostream>
#include <opencv2/core/core.hpp>
#include <stdio.h>
#include <dirent.h>
#include <string>
#include <sstream>

using namespace std;
using namespace cv;

// Output base directory and input image folder
const string to_write_dir = "/home/danny/Downloads/";
const char* input_img_dir = "/home/danny/Downloads/verdata/";

#define IMG_SUFFIX "png"

// Cascade file by OpenCV
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

// Cutting style

enum cutpos {
    FACE, RM_NOSE, RM_NOSE_EYES, RM_NOSE_MOUTH, RM_EYES, RM_MOUTH, RM_EYES_MOUTH
};
string cutpos_str[] = {"pure_face", "rm_nose", "rm_nose_eyes", "rm_nose_mouth", "rm_eyes", "rm_mouth", "rm_eyes_mouth"};

// cutting parameters control
cutpos cp = RM_EYES_MOUTH;
bool isHalf = true;
bool traverseAll = true;

void detectAndStat(string path, string img,
        CascadeClassifier& cascade, CascadeClassifier& eyeCas, CascadeClassifier& noseCas, CascadeClassifier& mouthCas,
        double scale);
void statAndNormal(string path, string imgname, CascadeClassifier& cascade, double scale);
void normalizeFaceCut(string path, string imgname, string towrite);
void cut(string path, string img, cutpos cp, string towrite);
string cleandir(string path);
void averageFace(string towrite, Mat face);

vector<string> split(string str, char delimiter);

// record of each face size
static map<string, string> face_size;

// global vars
int face_top = 0;
int face_bottom = 0;
int face_center_y = 0;
int face_center_x = 0;
int face_width = 0;
int face_height = 0;
int face_cnt = 0;

int eye_y = 0;
int eye_cnt = 0;
int eye_height = 0;
int nose_y = 0;
int nose_cnt = 0;
int nose_height = 0;

namespace patch {

    template < typename T > std::string to_string(const T& n) {
        std::ostringstream stm;
        stm << n;
        return stm.str();
    }
}

vector<string> split(string str, char delimiter) {
    vector<string> internal;
    stringstream ss(str);
    string tok;
    while (getline(ss, tok, delimiter)) {
        internal.push_back(tok);
    }
    return internal;
}

int main(int argc, const char** argv) {

    CascadeClassifier cascade, eyeCas, noseCas, mouthCas;
    double scale = 1.5;

    if (!cascade.load(cascadeName) || !eyeCas.load(eyeCasName) || !noseCas.load(noseCasName) || !mouthCas.load(mouthCasName)) {
        cerr << "ERROR: Could not load classifier cascade" << endl;
        return 1;
    }

    DIR *dir_face_stat, *dir_face_cut, *dir_other_stat, *dir_other_cut;
    struct dirent *ent_face_stat, *ent_face_cut, *ent_other_stat, *ent_other_cut;

    if ((dir_face_stat = opendir(input_img_dir)) != NULL && (dir_face_cut = opendir(input_img_dir)) != NULL) {

        // Stat for normalizing face size
        clock_t begin1 = clock();
        cout << "Start stat face..." << endl;
        while ((ent_face_stat = readdir(dir_face_stat)) != NULL) {
            if (strcmp(ent_face_stat->d_name, ".") && strcmp(ent_face_stat->d_name, "..")
                    //                    && strstr(ent_face_stat->d_name, IMG_SUFFIX)
                    ) {
                statAndNormal(input_img_dir, ent_face_stat->d_name, cascade, scale);
            }
        }
        clock_t end1 = clock();
        double elapsed_secs1 = double(end1 - begin1) / CLOCKS_PER_SEC;
        //        cout << "time:" << elapsed_secs1 << endl;
        closedir(dir_face_stat);

        // Stat result about face size
        face_top /= face_cnt;
        face_bottom /= face_cnt;
        face_center_y /= face_cnt;
        face_center_x /= face_cnt;
        face_width /= face_cnt;
        face_height /= face_cnt;

        // Cut out normalized faces
        string face_path = cleandir("face");
        cout << "Start normalizing face..." << endl;
        while ((ent_face_cut = readdir(dir_face_cut)) != NULL) {
            if (strcmp(ent_face_cut->d_name, ".") && strcmp(ent_face_cut->d_name, "..")
                    //                    && strstr(ent_face_cut->d_name, IMG_SUFFIX)
                    ) {
                normalizeFaceCut(input_img_dir, ent_face_cut->d_name, face_path);
            }
        }
        closedir(dir_face_cut);

        face_top = 0;
        face_bottom = 0;
        face_center_x = 0;
        face_center_y = 0;
        face_width = 0;
        face_height = 0;
        face_cnt = 0;

        if ((dir_other_stat = opendir(face_path.c_str())) != NULL && (dir_other_cut = opendir(face_path.c_str())) != NULL) {

            cout << "Start calculating other positions..." << endl;
            while ((ent_face_stat = readdir(dir_other_stat)) != NULL) {
                if (strcmp(ent_face_stat->d_name, ".") && strcmp(ent_face_stat->d_name, "..")
                        //                        && strstr(ent_face_stat->d_name, IMG_SUFFIX)
                        ) {
                    detectAndStat(face_path, ent_face_stat->d_name, cascade, eyeCas, noseCas, mouthCas, scale);
                }
            }
            closedir(dir_other_stat);


            // Based on normalized face, stat about others: like nose and eyes
            face_top /= face_cnt;
            face_bottom /= face_cnt;
            face_center_y /= face_cnt;
            face_center_x /= face_cnt;
            face_width /= face_cnt;
            face_height /= face_cnt;

            eye_y /= eye_cnt;
            eye_height /= eye_cnt;

            nose_y /= nose_cnt;
            nose_height /= nose_cnt;

            if (traverseAll) {
                for (int i = 0; i < sizeof (cutpos_str) / sizeof (cutpos_str[0]); i++) {
                    closedir(dir_other_cut);
                    dir_other_cut = opendir(face_path.c_str());

                    string towrite = cleandir(cutpos_str[i]);
                    cutpos icp = (cutpos) (FACE + i);
                    cout << "Start cutting image..." << cutpos_str[i] << endl;
                    // Cut by computed boundary
                    while ((ent_other_cut = readdir(dir_other_cut)) != NULL) {
                        if (strcmp(ent_other_cut->d_name, ".") && strcmp(ent_other_cut->d_name, "..")
                                //                                && strstr(ent_other_cut->d_name, IMG_SUFFIX)
                                ) {
                            cut(face_path, ent_other_cut->d_name, icp, towrite);
                        }
                    }
                }
            } else {
                string towrite = cleandir(cutpos_str[cp]);
                cout << "Start cutting image..." << endl;
                // Cut by computed boundary
                while ((ent_other_cut = readdir(dir_other_cut)) != NULL) {
                    if (strcmp(ent_other_cut->d_name, ".") && strcmp(ent_other_cut->d_name, "..")
                            //                            && strstr(ent_other_cut->d_name, IMG_SUFFIX)
                            ) {
                        cut(face_path, ent_other_cut->d_name, cp, towrite);
                    }
                }
            }
            closedir(dir_other_cut);

        } else {
            perror("");
            return EXIT_FAILURE;
        }

    } else {
        perror("");
        return EXIT_FAILURE;
    }


    return 0;
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

void statAndNormal(string path, string imgname, CascadeClassifier& cascade, double scale) {

    // Load image as gray scale
    Mat img = imread(path + imgname, CV_LOAD_IMAGE_COLOR);
    Mat gray, smallImg(cvRound(img.rows / scale), cvRound(img.cols / scale), CV_8UC1);
    cvtColor(img, gray, CV_BGR2GRAY);

    // Shrink size to speed up
    resize(gray, smallImg, smallImg.size(), 0, 0, INTER_LINEAR);
    // Equalizes the histogram of a gray scale image.
    equalizeHist(smallImg, smallImg);

    // Detect face and statistics performance
    double t = 0;
    t = (double) cvGetTickCount();
    vector<Rect> faces;
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
    int i = 0;
    for (vector<Rect>::const_iterator r = faces.begin(); r != faces.end(); r++, i++) {
        // stat about face
        face_top += cvRound(r->y * scale);
        face_bottom += cvRound((r->y + r->height) * scale);
        face_center_x += cvRound((r->y + r->width * 0.5) * scale);
        face_center_y += cvRound((r->x + r->height * 0.5) * scale);
        face_width += cvRound(r->width * scale);
        face_height += cvRound(r->height * scale);
        face_cnt += 1;

        // record face: start  point(x,y) and width / height
        face_size[imgname] = patch::to_string(cvRound(r->x * scale)) + "_" + patch::to_string(cvRound(r->y * scale)) + "_" + patch::to_string(cvRound(r->width * scale)) + "_" + patch::to_string(cvRound(r->height * scale));

        // Tag face using rectangle
        //        rectangle(img, Point(r->x, r->y), Point(cvRound((r->x + r->width) * scale), cvRound((r->y + r->height) * scale)), colors[0], 2, 8);
        //        int face_x = center.x;
        //        cv::imshow("result", img);
        //        waitKey(0);
    }
}

void detectAndStat(string path, string imgname, CascadeClassifier& cascade, CascadeClassifier& eyeCas, CascadeClassifier& noseCas, CascadeClassifier& mouthCas,
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

        // circle on face
        center.x = cvRound((r->x + r->width * 0.5) * scale);
        center.y = cvRound((r->y + r->height * 0.5) * scale);

        face_top += cvRound(r->y * scale);
        face_bottom += cvRound((r->y + r->height) * scale);
        face_center_y += center.y;
        face_center_x += center.x;
        face_width += r->width * scale;
        face_cnt += 1;

        face_size[imgname] = patch::to_string(cvRound(r->x * scale)) + "_" + patch::to_string(cvRound(r->y * scale)) + "_" + patch::to_string(cvRound(r->width * scale)) + "_" + patch::to_string(cvRound(r->height * scale));

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

void averageFace(string towrite, Mat face) {
    double mid = cvCeil(face.cols / 2);
    Mat gray;
    cvtColor(face, gray, CV_BGR2GRAY);

    if (isHalf) {
        for (int row = 0; row < gray.rows; row++) {
            for (int col = mid; col < 2 * mid; col++) {
                gray.at<uchar>(row, col) = (gray.at<uchar>(row, col) + gray.at<uchar>(row, gray.cols - 1 - col)) / 2;
            }
        }

        Rect average(mid, 0, mid, gray.rows);
        //    imshow("test", gray(average));
        //    waitKey(0);
        gray = gray(average);
        equalizeHist(gray, gray);
        imwrite(towrite, gray);
    } else {
        equalizeHist(gray, gray);
        imwrite(towrite, gray);
    }

}

void cut(string path, string imgname, cutpos cp, string towrite) {
    Mat img = imread(path + imgname, CV_LOAD_IMAGE_COLOR);
    Mat output = img;

    switch (cp) {
        case FACE:
        {

            averageFace(towrite + imgname, output);

            //            imshow("combine", combine);
            //            waitKey(0);
            break;
        }
        case RM_NOSE:
        {
            Rect up_rect(0, 0, output.cols, eye_y + eye_height / 2);
            Mat up_image = output(up_rect);

            Rect down_rect(0, nose_y + nose_height / 2, output.cols, face_bottom - nose_y - nose_height / 2);
            Mat down_image = output(down_rect);

            Mat combine;
            vconcat(up_image, down_image, combine);
            averageFace(towrite + imgname, combine);

            //            imshow("combine", combine);
            //            waitKey(0);
            break;
        }
        case RM_EYES:
        {
            Rect up_rect(0, 0, output.cols, eye_y - eye_height);
            Mat up_image = output(up_rect);

            Rect down_rect(0, eye_y + eye_height, output.cols, output.rows - eye_y - eye_height);
            Mat down_image = output(down_rect);

            Mat combine;
            vconcat(up_image, down_image, combine);
            averageFace(towrite + imgname, combine);

            //            imshow("combine", combine);
            //            waitKey(0);
            break;
        }
        case RM_MOUTH:
        {
            Rect mouth_rec(0, face_top, output.cols, nose_y + nose_height / 2 - face_top);
            Mat mouth = output(mouth_rec);

            averageFace(towrite + imgname, mouth);

            //            imshow("combine", combine);
            //            waitKey(0);
            break;
        }
        case RM_NOSE_EYES:
        {
            Rect up_rect(0, 0, output.cols, eye_y - eye_height);
            Mat up_image = output(up_rect);

            Rect down_rect(0, nose_y + nose_height / 2, output.cols, face_bottom - nose_y - nose_height / 2);
            Mat down_image = output(down_rect);

            Mat combine;
            vconcat(up_image, down_image, combine);
            averageFace(towrite + imgname, combine);

            //            imshow("combine", combine);
            //            waitKey(0);
            break;
        }
        case RM_NOSE_MOUTH:
        {
            Rect eyes_rec(0, 0, output.cols, face_center_y);
            Mat eyes = output(eyes_rec);
            averageFace(towrite + imgname, eyes);
            //            imshow("combine", combine);
            //            waitKey(0);
            break;
        }
        case RM_EYES_MOUTH:
        {
            Rect nose_rec(0, face_center_y, output.cols, nose_y + nose_height / 2 - face_center_y);
            Mat nose = output(nose_rec);
            averageFace(towrite + imgname, nose);
            //            imshow("combine", combine);
            //            waitKey(0);
            break;
        }
        default:
            cout << "Nothing to do" << endl;
            break;
    }

    // For Cutting Visualization, you will see more about the cutting line when uncomment the following codes
    //    line(img, Point(0, face_top), Point(img.cols, face_top), colors[1]);
    //    line(img, Point(0, eye_y), Point(img.cols, eye_y), colors[2]);
    //    line(img, Point(0, nose_y), Point(img.cols, nose_y), colors[3]);
    //    line(img, Point(0, face_center), Point(img.cols, face_center), colors[4]);
    //    line(img, Point(0, face_bottom), Point(img.cols, face_bottom), colors[5]);
    //    cv::imshow("result", img);
    //    waitKey(0);
}

void normalizeFaceCut(string path, string imgname, string towrite) {
    //    int face_left = face_center_x - face_width / 2;
    Mat img = imread(path + imgname, CV_LOAD_IMAGE_COLOR);

    vector<string> sep = split(face_size[imgname], '_');
    int self_face_x = atoi(sep[0].c_str());
    int self_face_y = atoi(sep[1].c_str());
    int self_face_width = atoi(sep[2].c_str());
    int self_face_height = atoi(sep[3].c_str());

    Rect face_rec(self_face_x, self_face_y, self_face_width, self_face_height);
    Mat face = img(face_rec);
    //            imshow("output", face);
    //            waitKey(0);
    Size size(face_width, face_bottom - face_top);
    Mat dst;
    resize(face, dst, size);
    imwrite(towrite + imgname, dst);
}