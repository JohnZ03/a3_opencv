#include <iostream>
#include "opencv2/opencv.hpp"

using namespace cv;
using namespace std;
using namespace std::chrono;

int main() {
	auto start = high_resolution_clock::now();

//    std::cout << "OpenCV config sample." << std::endl;
//    std::cout<< cv::getBuildInformation() <<std::endl;
	Mat Ihack = imread("../yonge_dundas_square.jpg", IMREAD_COLOR);
	Mat Ist = imread("../uoft_soldiers_tower_dark.png", IMREAD_GRAYSCALE);

	for (int n = 0; n < 100; n++) {
		int32_t bbox_array[2][4] = {{404, 490, 404, 490},
		                            {38,  38,  354, 354}};
		int bbox_array_len = sizeof(bbox_array[0]) / sizeof(bbox_array[0][0]);
		int32_t i_range[2] = {*std::min_element(bbox_array[0], bbox_array[0] + bbox_array_len),
		                      *std::max_element(bbox_array[0], bbox_array[0] + bbox_array_len)};
		int32_t j_range[2] = {*std::min_element(bbox_array[1], bbox_array[1] + bbox_array_len),
		                      *std::max_element(bbox_array[1], bbox_array[1] + bbox_array_len)};

		int32_t Iyd_pts_array[4][2] = {{416, 40},
		                               {485, 61},
		                               {488, 353},
		                               {410, 349}};
		Mat Iyd_pts = Mat(4, 1, CV_32SC2, &Iyd_pts_array);
		std::vector<Point> Iyd_pts_contours = {Point(416, 40), Point(485, 61), Point(488, 353), Point(410, 349)};

		int32_t Ist_pts_array[4][2] = {{2,   2},
		                               {218, 2},
		                               {218, 409},
		                               {2,   409}};
		Mat Ist_pts = Mat(4, 1, CV_32SC2, &Ist_pts_array);

		Mat I_equ;
		equalizeHist(Ist, I_equ);
		I_equ.convertTo(I_equ, CV_32F);

		//    std::cout << Iyd_pts.at<int32_t>(0,0) << std::endl;
		//    std::cout << Ist_pts << std::endl;

		Mat H = findHomography(Iyd_pts, Ist_pts);
		H.convertTo(H, CV_32F);

		//    std::cout << H << std::endl;

		for (int32_t i = i_range[0]; i <= i_range[1]; i++) {
			for (int32_t j = j_range[0]; j <= j_range[1]; j++) {
				Point2f point_inspected(i, j);
				if (pointPolygonTest(Iyd_pts, point_inspected, 0) >= 0) {
					float x_array[3][1] = {{static_cast<float>(i)},
					                       {static_cast<float>(j)},
					                       {1}};
					Mat x = Mat(3, 1, CV_32F, &x_array);
					Mat x_prime = H * x;

					// Normalization
					x_prime = x_prime / x_prime.at<float>(2, 0);

					// Interpolation
					int x1 = ceil(x_prime.at<float>(0, 0)) - 1;
					int x2 = floor(x_prime.at<float>(0, 0)) + 1;
					int y1 = ceil(x_prime.at<float>(1, 0)) - 1;
					int y2 = floor(x_prime.at<float>(1, 0)) + 1;

					// Clip into range
					x1 = std::max(0, std::min(x1, 219));
					x2 = std::max(0, std::min(x2, 219));
					y1 = std::max(0, std::min(y1, 410));
					y2 = std::max(0, std::min(y2, 410));

					float x_temp = x_prime.at<float>(0, 0);
					float y_temp = x_prime.at<float>(1, 0);

					float temp_1 = ((x2 - x_temp) / (x2 - x1)) * float(I_equ.at<float>(y1, x1)) +
					               ((x_temp - x1) / (x2 - x1)) * float(I_equ.at<float>(y1, x2));
					float temp_2 = ((x2 - x_temp) / (x2 - x1)) * float(I_equ.at<float>(y2, x1)) +
					               ((x_temp - x1) / (x2 - x1)) * float(I_equ.at<float>(y2, x2));

					int b = round(((y2 - y_temp) / (y2 - y1)) * temp_1 + ((y_temp - y1) / (y2 - y1)) * temp_2);

					// Mat b_in = Mat(1, 1, CV_32FC3, &b);
					Ihack.at<Vec3b>(j, i)[0] = b;
					Ihack.at<Vec3b>(j, i)[1] = b;
					Ihack.at<Vec3b>(j, i)[2] = b;
				}
			}
		}


	}

	imwrite("../billboard_hacked_cpp.png", Ihack);
//    namedWindow("result",cv::WINDOW_AUTOSIZE);
//    imshow("result", Ihack);
//
//    cv::waitKey(0);
//    cv::destroyWindow("result");

	auto stop = high_resolution_clock::now();
	auto duration = duration_cast<microseconds>(stop - start);
	cout << duration.count() << endl;

	return 0;
}
