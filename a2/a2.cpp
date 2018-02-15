#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

void dftQuadSwap (Mat& img)  {
    // rearrange the quadrants of Fourier image  so that the origin is at the image center
    int cx = img.cols/2;
    int cy = img.rows/2;

    Mat q0(img, Rect(0, 0, cx, cy));   // Top-Left - Create a ROI per quadrant
    Mat q1(img, Rect(cx, 0, cx, cy));  // Top-Right
    Mat q2(img, Rect(0, cy, cx, cy));  // Bottom-Left
    Mat q3(img, Rect(cx, cy, cx, cy)); // Bottom-Right

    Mat tmp;                           // swap quadrants (Top-Left with Bottom-Right)
    q0.copyTo(tmp);
    q3.copyTo(q0);
    tmp.copyTo(q3);

    q1.copyTo(tmp);                    // swap quadrant (Top-Right with Bottom-Left)
    q2.copyTo(q1);
    tmp.copyTo(q2);
}

void switchLogScale(Mat& img) {
    img += Scalar::all(1);                    // switch to logarithmic scale
    log(img, img);
}

Mat doHighPass(Mat img) {
    Mat imgPlanes[] = {Mat_<float>(img),    Mat::zeros(img.size(),    CV_32F)};
    Mat imgRI, prdRI;
    merge(imgPlanes, 2, imgRI);
    prdRI = imgRI.clone();
    dft(imgRI, imgRI, DFT_COMPLEX_OUTPUT);
	dftQuadSwap(imgRI);
    Point2i center(imgRI.rows/2,imgRI.cols/2);
    circle(imgRI, center, 50, Scalar(0,0,0), -1);
	/*vector<Mat> channels(2);
	split(imgRI, channels);
    imshow("HP imgR", channels[0]);
    imshow("HP imgI", channels[1]);*/
	dftQuadSwap(imgRI);
    Mat inverseTransform; 
    dft(imgRI, inverseTransform, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
    normalize(inverseTransform, inverseTransform, 0, 1, CV_MINMAX);
	return inverseTransform;
}

Mat doLowPass(Mat img) {
	// Now construct a Gaussian kernel
    float sigma = 8.0;
    Mat kernelX   = getGaussianKernel(img.rows, sigma, CV_32FC1);
    Mat kernelY   = getGaussianKernel(img.cols, sigma, CV_32FC1);
    Mat kernel  = kernelX * kernelY.t();
    Mat kernel_d = kernel.clone();
    normalize(kernel_d, kernel_d, 0, 1, CV_MINMAX);
	//imshow("Spatial Domain", kernel_d);
    // Build complex images for both the source image and the Gaussian kernel
    Mat imgPlanes[] = {Mat_<float>(img),    Mat::zeros(img.size(),    CV_32F)};
    Mat kerPlanes[] = {Mat_<float>(kernel), Mat::zeros(kernel.size(), CV_32F)};
    Mat imgRI, kerRI, prdRI;
    merge(imgPlanes, 2, imgRI);
    merge(kerPlanes, 2, kerRI);
    prdRI = imgRI.clone();
    dft(imgRI, imgRI, DFT_COMPLEX_OUTPUT);
    dft(kerRI, kerRI, DFT_COMPLEX_OUTPUT);
	/*dftQuadSwap(imgRI);
	vector<Mat> channels(2);
	split(imgRI, channels);
    imshow("LP imgR", channels[0]);
    imshow("LP imgI", channels[1]);
	dftQuadSwap(imgRI);*/
    mulSpectrums(imgRI, kerRI, prdRI, DFT_COMPLEX_OUTPUT);
    Mat inverseTransform; // broken because it takes inverse of original image
    dft(imgRI, inverseTransform, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
    dft(prdRI, inverseTransform, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
    dftQuadSwap(inverseTransform);
    normalize(inverseTransform, inverseTransform, 0, 1, CV_MINMAX);
	return inverseTransform;
}

Mat doSomethingCool(Mat img) {
	vector<Mat> channels(3);
	split(img, channels);
	Mat B = doLowPass(channels[0]);
	Mat G = doHighPass(channels[1]);
	vector<Mat> input = {B, G, channels[2]};
	Mat output;
	merge(input, output);
	return output;
}

Mat doSomethingCool2(Mat img, Mat edges) {
	// BGR to HSV
	cvtColor(img, img, CV_BGR2HSV);    
	for (int i=0; i < img.rows ; i++)
	{
		for(int j=0; j < img.cols; j++)
		{
			int b = edges.at<cv::Vec3b>(i,j)[0];
			int g = edges.at<cv::Vec3b>(i,j)[1];
			int r = edges.at<cv::Vec3b>(i,j)[2];
			float ratio = float(b + g + r) / float(3 * 255) + 1;
			int s = img.at<cv::Vec3b>(i,j)[1];
			int s2 = s * ratio;
			img.at<cv::Vec3b>(i,j)[1] = s2;
		}
	}
	// HSV back to BGR
	cvtColor(img, img, CV_HSV2BGR);
	return img;
}

int main(int argc, char ** argv)
{
    //  Start by loading the image to be smoothed
	const char* filename = argc >= 3 ? argv[2] : "colostate_quad_bw_512.png";
	bool isCool = argc >= 2 ? string(argv[1]) == "cool" : false;
    Mat inimg;                            //expand input image to optimal size
    Mat inimggray;                            //expand input image to optimal size
	inimg = imread(filename, CV_LOAD_IMAGE_COLOR);
	inimggray = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	Mat img;
	Mat imggray;
	int m = getOptimalDFTSize( img.rows );
	int n = getOptimalDFTSize( img.cols ); // on the border add zero values
	int mg = getOptimalDFTSize( imggray.rows );
	int ng = getOptimalDFTSize( imggray.cols ); // on the border add zero values
	copyMakeBorder(inimg, img, 0, m - img.rows, 0, n - img.cols, BORDER_CONSTANT, Scalar::all(0));
	copyMakeBorder(inimggray, imggray, 0, m - imggray.rows, 0, n - imggray.cols, BORDER_CONSTANT, Scalar::all(0));
    if( inimg.empty())
        return -1;
	if (isCool) {
		Mat im2 = doHighPass(imggray);
		imshow("original", img);
		imshow("high-pass", im2);
		imshow("cool", doSomethingCool2(img, im2));
	} else {
		imshow("High Pass", doHighPass(imggray));
		imshow("Low Pass", doLowPass(imggray));
	}
	waitKey();
    return 0;
}
