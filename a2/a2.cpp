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

Mat getGaussian(int rows, int cols, float sigma, bool isFourier) {
	Mat kerRI;
	Mat kernelX   = getGaussianKernel(rows, sigma, CV_32FC1);
	Mat kernelY   = getGaussianKernel(cols, sigma, CV_32FC1);
	Mat kernel  = kernelX * kernelY.t();
	Mat kernel_d = kernel.clone();
	normalize(kernel_d, kernel_d, 0, 1, CV_MINMAX);
	if (isFourier) {
		Mat kerPlanes[] = {Mat_<float>(kernel), Mat::zeros(kernel.size(), CV_32F)};
		merge(kerPlanes, 2, kerRI);
		dft(kerRI, kerRI, DFT_COMPLEX_OUTPUT);
		return kerRI;
	} else {
		return kernel_d;
	}
}

//uses inverse gaussian
Mat doHighPass(Mat img, int scale) {
    Mat imgPlanes[] = {Mat_<float>(img),    Mat::zeros(img.size(),    CV_32F)};
    Mat imgRI, prdRI;
    merge(imgPlanes, 2, imgRI);
    prdRI = imgRI.clone();
    dft(imgRI, imgRI, DFT_COMPLEX_OUTPUT);
	dftQuadSwap(imgRI);
    Point2i center(imgRI.rows/2,imgRI.cols/2);
    circle(imgRI, center, imgRI.rows/scale, Scalar(0,0,0), -1);
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

//use this to display fourier domains to the user
Mat getMag(Mat complex) {
	Mat magI;
	Mat planes[] = {
			Mat::zeros(complex.size(), CV_32F),
			Mat::zeros(complex.size(), CV_32F)
	};
	split(complex, planes); // planes[0] = Re(DFT(I)), planes[1] = Im(DFT(I))
	magnitude(planes[0], planes[1], magI); // sqrt(Re(DFT(I))^2 + Im(DFT(I))^2)
	// switch to logarithmic scale: log(1 + magnitude)
	magI += Scalar::all(1);
	log(magI, magI);
	dftQuadSwap(magI); // rearrage quadrants
	// Transform the magnitude matrix into a viewable image (float values 0-1)
	normalize(magI, magI, 1, 0, NORM_INF);
	return magI;
}

Mat medianToZero(Mat img) {
	Mat copy = img.clone();
	copy = copy.reshape(0,1);
	vector<double> vecFromMat;
	copy.copyTo(vecFromMat);
	std::nth_element(vecFromMat.begin(), vecFromMat.begin() + vecFromMat.size() / 2, vecFromMat.end());
	double median = vecFromMat[vecFromMat.size() / 2];
	//cout << "median is " << median << endl;
	//cout << "mean is " << mean(img) << endl;
	return img - median;
}

Mat meanToZero(Mat img) {
	Scalar ave = mean(img);
	return img - ave;
}

//using inverse gaussian instead of circle
Mat doHighPass2(Mat img, int sigma) {
    Mat imgPlanes[] = {Mat_<float>(img),    Mat::zeros(img.size(),    CV_32F)};
    Mat imgRI, inverseTransform, invkerRI, invgauss;
    merge(imgPlanes, 2, imgRI);
    dft(imgRI, imgRI, DFT_COMPLEX_OUTPUT);
	invgauss = 1 - getGaussian(imgRI.rows, imgRI.cols, sigma, false);
	Mat kerPlanes[] = {invgauss, Mat::zeros(invgauss.size(), CV_32F)};
	merge(kerPlanes, 2, invkerRI);
	dftQuadSwap(invkerRI);
    mulSpectrums(imgRI, invkerRI, imgRI, DFT_COMPLEX_OUTPUT);
    dft(imgRI, inverseTransform, cv::DFT_INVERSE|cv::DFT_REAL_OUTPUT);
    normalize(inverseTransform, inverseTransform, 0, 1, CV_MINMAX);
	return inverseTransform;
}

Mat doLowPass(Mat img, float sigma) {
    Mat imgPlanes[] = {Mat_<float>(img),    Mat::zeros(img.size(),    CV_32F)};
    Mat imgRI, kerRI, prdRI;
	kerRI = getGaussian(img.rows, img.cols, sigma, true);
    merge(imgPlanes, 2, imgRI);
    prdRI = imgRI.clone();
    dft(imgRI, imgRI, DFT_COMPLEX_OUTPUT);
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
	Mat B = doLowPass(channels[0],1.0);
	//Mat B = doHighPass(channels[0],40);
	//Mat G = doLowPass(channels[1],0.01);
	Mat R = doLowPass(channels[2],8.0);
	Mat G = doHighPass(channels[1],40);
    //Mat R = doHighPass(channels[2],40);
	vector<Mat> input = {B, G, R};
	Mat output;
	merge(input, output);
	return output;
}

Mat doSomethingCool2(Mat img, Mat edges) {
	// BGR to HSV
	cvtColor(img, img, CV_BGR2HSV);for (int i=0; i < img.rows ; i++)
	{
		for(int j=0; j < img.cols; j++)
		{
			int b = edges.at<cv::Vec3b>(i,j)[0];
			int g = edges.at<cv::Vec3b>(i,j)[1];
			int r = edges.at<cv::Vec3b>(i,j)[2];
			float ratio = float(b+g+r) / float(3);
            //cout << float(img.at<cv::Vec3b>(i,j)[1]) << " ratio: " << int(ratio) << endl;
            img.at<cv::Vec3b>(i,j)[2] = char(ratio);
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
		Mat im2 = doHighPass2(imggray,20) - 0.5;
		imshow("original", img);
		imshow("high-pass", im2);
        imshow("Selective low pass", doSomethingCool(img));
		imshow("cool", doSomethingCool2(img, im2));
	} else {
		imshow("High Pass", doHighPass2(imggray,40));
		imshow("Low Pass", doLowPass(imggray, 4.0));
	}
	waitKey();
    return 0;
}
