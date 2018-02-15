#include "opencv2/core/core.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/highgui/highgui.hpp"
#include <iostream>
#include <vector>
#include <string>

using namespace cv;
using namespace std;

/*
 * In this example we do smoothing with a Gaussian, but in the
 * Frequency domain.
 *
 * Ross Beveridge
 *
 */

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
    circle(imgRI, center, 30, Scalar(0,0,0), -1);
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

Mat doLowPass(Mat img, float sigma) {
	// Now construct a Gaussian kernel
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
	Mat B = doLowPass(channels[0],0.01);
	Mat G = doLowPass(channels[1],0.01);
	Mat R = doLowPass(channels[2],16.0);
	//Mat G = doHighPass(channels[1]);
    //Mat R = doHighPass(channels[2]);
	vector<Mat> input = {B, G, R};
    cout << "B: " << B.size() << "G: " << G.size() << "channels[2]: " << channels[2].size() << endl;
    cout << "B: " << B.depth() << "G: " << G.depth() << "channels[2]: " << channels[2].depth() << endl;
	Mat output;
	merge(input, output);
	return output;
}


int main(int argc, char ** argv)
{
    //  Start by loading the image to be smoothed
	const char* filename = argc >= 3 ? argv[2] : "colostate_quad_bw_512.png";
	bool isColor = argc >= 2 ? string(argv[1]) == "color" : false;
    Mat inimg;                            //expand input image to optimal size
	if (isColor) 
		inimg = imread(filename, CV_LOAD_IMAGE_COLOR);
	else
		inimg = imread(filename, CV_LOAD_IMAGE_GRAYSCALE);
	Mat img;
	int m = getOptimalDFTSize( img.rows );
	int n = getOptimalDFTSize( img.cols ); // on the border add zero values
	copyMakeBorder(inimg, img, 0, m - img.rows, 0, n - img.cols, BORDER_CONSTANT, Scalar::all(0));
    if( inimg.empty())
        return -1;
	int nChannels = img.channels();
	if (isColor)
		imshow("cool", doSomethingCool(img));
	else {
		imshow("High Pass", doHighPass(img));
		imshow("Low Pass", doLowPass(img, 4.0));
	}
	waitKey();
    return 0;
}
