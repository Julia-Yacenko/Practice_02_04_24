#include <opencv2/core.hpp>
#include <opencv2/videoio.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/features2d.hpp>
#include <iostream>
#include <vector>

using namespace std;
using namespace cv;

void collectCards(vector<Mat>& cardsImages, vector< string>& cardsNames, vector<Mat>& cardsDescriptors, vector<vector<KeyPoint>>& cardsKeypoints) 
{
	Mat card;

	card = imread("D:/Camera/6_piki.jpg");
	resize(card, card, Size(145, 220));
	cardsImages.push_back(card);
	cardsNames.push_back("6_PIKY");

	card = imread("D:/Camera/8_bubi.jpg");
	resize(card, card, Size(145, 220));
	cardsImages.push_back(card);
	cardsNames.push_back("8_BUBY");

	card = imread("D:/Camera/king_chervi.jpg");
	resize(card, card, Size(150, 220));
	cardsImages.push_back(card);
	cardsNames.push_back("KOROL_CHERVY");

	card = imread("D:/Camera/tuz_cresti.jpg");
	resize(card, card, Size(150, 220));
	cardsImages.push_back(card);
	cardsNames.push_back("TUZ_KRESTY");


	Ptr<ORB> detector = ORB::create();

	for (int i = 0; i < cardsImages.size(); i++) {
		Mat dis;
		vector<KeyPoint> keys;
		detector->detectAndCompute(cardsImages[i], noArray(), keys, dis);
		cardsKeypoints.push_back(keys);
		cardsDescriptors.push_back(dis);
	}
}

void detectCard(string& cardName, Mat& card, vector<Mat>& cardsImages, vector<string>& cardsNames, vector<Mat>& cardsDescriptors, vector<vector<KeyPoint>>& cardsKeypoints) 
{
	Mat cardDescriptors;
	vector<KeyPoint> cardKeypoints;
	Ptr<ORB> detector = ORB::create();
	Ptr<BFMatcher> matcher = BFMatcher::create();
	detector->detectAndCompute(card, noArray(), cardKeypoints, cardDescriptors);

	if (cardDescriptors.empty()) {
		cardName = "";
		return;
	}

	int maxI = -1;
	int maxCount = 0;

	for (int i = 0; i < cardsImages.size(); i++) {

		if (cardsDescriptors[i].empty()) {
			continue;
		}

		vector<vector<DMatch>> knn_matches;

		matcher->knnMatch(cardsDescriptors[i], cardDescriptors, knn_matches, 3);

		vector<DMatch> correct;

		for (size_t i = 0; i < knn_matches.size(); i++) {
			if (knn_matches[i][0].distance < 0.75 * knn_matches[i][1].distance) {
				correct.push_back(knn_matches[i][0]);
			}
		}

		if (maxCount < correct.size()) {
			maxCount = static_cast<int>(correct.size());
			maxI = i;
		}
	}

	if (maxI == -1) {
		cardName = "";
	}
	else {
		cardName = cardsNames[maxI];
	}
}

int main()
{
    vector<Mat> cardsImages;
    vector<string> cardsNames;
    vector<Mat> cardsDescriptors;
    vector<vector<KeyPoint>> keypoints;

    collectCards(cardsImages, cardsNames, cardsDescriptors, keypoints);
    
    const Mat input = imread("D:/Camera/carta.jpg");
    if (!input.data)
    {
        printf("Error loading image \n"); return -1;
    }
	cv::Mat img, greyImg, image1, image2;
    resize(input, img, Size(500, 500), INTER_LINEAR);
    cvtColor(img, greyImg, COLOR_BGR2GRAY);
    GaussianBlur(greyImg, image1, Size(3, 3), 0);
    threshold(image1, image2, 215, 255, THRESH_BINARY);

    std::vector<std::vector<Point>> contours;
    std::vector<Vec4i> hierarchy;
    findContours(image2, contours, hierarchy, RETR_EXTERNAL,
        CHAIN_APPROX_SIMPLE);

	for (const auto& contour : contours) {
		vector<Point> contoursPoly;

		approxPolyDP(contour, contoursPoly, 1, true);

		RotatedRect cardRect = minAreaRect(contoursPoly);

		if (cardRect.size.width < 100 || cardRect.size.height < 100) {
			continue;
		}

		Mat card, rotatedMatrix, rotatedImage;
		string cardName;

		rotatedMatrix = getRotationMatrix2D(cardRect.center, cardRect.angle, 1.0);
		warpAffine(img, rotatedImage, rotatedMatrix, img.size(), INTER_CUBIC);
		getRectSubPix(rotatedImage, cardRect.size, cardRect.center, card);

		rotate(card, card, ROTATE_180);

		if (card.size[0] < card.size[1]) {
			rotate(card, card, ROTATE_90_CLOCKWISE);
		}

		detectCard(cardName, card, cardsImages, cardsNames, cardsDescriptors, keypoints);

		if (cardName != "") {
			Point2f boxPoints[4];
			cardRect.points(boxPoints);

			for (int j = 0; j < 4; j++) {
				line(img, boxPoints[j], boxPoints[(j + 1) % 4], Scalar(0, 255, 0), 2, LINE_AA);
			}
			putText(img, cardName, cardRect.center, FONT_HERSHEY_DUPLEX, 0.5, Scalar(0, 255, 0), 2);
		}
	}

    imshow("Output", img);
    waitKey(0);

    return 0;
}