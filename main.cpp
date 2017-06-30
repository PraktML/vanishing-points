#include <math.h>
#include <armadillo>
#include <ctime>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "opencv2/core/core.hpp"
#include "opencv2/core/utility.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgcodecs.hpp"
#include "opencv2/imgproc/imgproc.hpp"

using namespace cv;
using namespace std;
using namespace arma;

struct MatchPathSeparator {
  bool operator()(char ch) const { return ch == '/'; }
};

class VanishingPointDetector {
 public:
  cv::Mat image, img, gray;
  cv::Mat frame;
  vector<vector<int>> points;
  mat A, b, prevRes;
  mat Atemp, btemp, res, aug, error, soln;
  // ofstream out1, out2;
  float epsilon;

  // store slope (m) and y-intercept (c) of each lines
  float m, c;

  // store minimum length for lines to be considered while estimating
  // vanishing point
  int minlength;

  // temporary vector for intermediate storage
  vector<int> temp;

  // store (x1, y1) and (x2, y2) endpoints for each line segment
  vector<cv::Vec4i> lines_std;

  // video capture object from OpenCV
  cv::VideoCapture cap;

  // to store intermediate errors
  double temperr;

  // Bounding box to be excluded for vanishing point detection
  cv::Point bb_top_left;
  cv::Point bb_bottom_right;

  // Flag whether a bounding box was set
  bool has_bounding_box;

  // constructor to set video/webcam and find vanishing point
  VanishingPointDetector() {
    cv::namedWindow("win", 2);
    cv::namedWindow("Lines", 2);

    // read from video file on disk
    // to read from webcam initialize as: cap = VideoCapture(int device_id);
    // cap = VideoCapture(1);
    // cap = VideoCapture("road.m4v");

    // if (cap.isOpened()) {
    // get first frame to intialize the values

    // for (auto &imgPath : imagePaths) {
    // frame = cv::imread(input_path, CV_LOAD_IMAGE_COLOR);
    // image = cv::Mat(cv::Size(frame.rows, frame.cols), CV_8UC1, 0.0);
    ////}

    //// define minimum length requirement for any line
    // minlength = image.cols * image.cols * 0.001;

    // cout << minlength << endl;

    // int flag = 0;
    //// while (cap.isOpened())  // check if camera/ video stream is
    //// available
    ////{
    //// if (!cap.grab()) continue;

    //// if (!cap.retrieve(img)) continue;

    //// it's advised not to modify image stored in the buffer structure
    //// of the opencv.
    //// frame = img.clone();

    //// to calculate fps
    //// begin = clock();

    // cv::cvtColor(frame, image, cv::COLOR_BGR2GRAY);

    //// resize frame to 480x320
    // cv::resize(image, image, cv::Size(480, 320));

    //// equalize histogram
    // cv::equalizeHist(image, image);

    //// initialize the line segment matrix in format y = m*x + c
    // init(image, prevRes);

    //// draw lines on image and display
    // makeLines(flag);

    //// approximate vanishing point
    // eval(output_path);
  }

  void find_contours(cv::Mat img, std::string output_path) {
    cv::RNG rng(12345);
    cv::Mat canny_output;
    vector<vector<Point>> contours;
    vector<Vec4i> hierarchy;
    int thresh = 100;

    cv::Rect roi(bb_top_left.x - 10, bb_top_left.y - 10, bb_bottom_right.x - bb_top_left.x + 20, bb_bottom_right.y - bb_top_left.y + 20);
    cv::Mat image_roi = img(roi);

    cv::Canny(image_roi, canny_output, thresh, thresh * 2, 3);

    findContours(canny_output, contours, hierarchy, CV_RETR_TREE,
		 CV_CHAIN_APPROX_SIMPLE, Point(0, 0));

    cv::Mat drawing = cv::Mat::zeros(canny_output.size(), CV_8UC3);
    for (int i = 0; i < contours.size(); i++) {
      Scalar color =
	  Scalar(rng.uniform(0, 255), rng.uniform(0, 255), rng.uniform(0, 255));
      drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
    }
    cv::imwrite(output_path + "_contours.bmp", drawing);
  }

  void process_image(std::string input_path, std::string output_path) {
    frame = cv::imread(input_path, CV_LOAD_IMAGE_COLOR);
    image = cv::Mat(cv::Size(frame.rows, frame.cols), CV_8UC1, 0.0);
    int flag = 0;

    minlength = image.cols * image.cols * 0.01;
    cv::cvtColor(frame, image, cv::COLOR_BGR2GRAY);

    find_contours(image, output_path);

    // cv::resize(image, image, cv::Size(480, 320));
    cv::equalizeHist(image, image);
    init(image, prevRes);
    makeLines(flag);
    eval(output_path);
  }

  void set_bounding_box(cv::Point top_left, cv::Point bottom_right) {
    bb_top_left = top_left;
    bb_bottom_right = bottom_right;

    has_bounding_box = true;
  }

  inline bool is_in_bounding_box(mat point) {
    double x = soln(0, 0);
    double y = soln(1, 0);

    return x >= bb_top_left.x && x <= bb_bottom_right.x &&
	   y >= bb_top_left.y && y <= bb_bottom_right.y;
  }

  void init(cv::Mat image, mat prevRes) {
    // create OpenCV object for line segment detection
    cv::Ptr<cv::LineSegmentDetector> ls =
	cv::createLineSegmentDetector(cv::LSD_REFINE_STD);

    // initialize
    lines_std.clear();

    // detect lines in image and store in linse_std
    // store (x1, y1) and (x2, y2) endpoints for each line segment
    ls->detect(image, lines_std);

    // Show found lines
    cv::Mat drawnLines(image);
    // cv::Mat drawnLines(cv::Size(image.cols, image.rows), CV_8UC1,
    // 100.0);

    for (int i = 0; i < lines_std.size(); i++) {
      // ignore if almost vertical
      if (abs(lines_std[i][0] - lines_std[i][2]) < 10 ||
	  abs(lines_std[i][1] - lines_std[i][3]) <
	      10)  // check if almost vertical
	continue;
      // ignore shorter lines (x1-x2)^2 + (y2-y1)^2 < minlength
      if (((lines_std[i][0] - lines_std[i][2]) *
	       (lines_std[i][0] - lines_std[i][2]) +
	   (lines_std[i][1] - lines_std[i][3]) *
	       (lines_std[i][1] - lines_std[i][3])) < minlength)
	continue;

      // store valid lines' endpoints for calculations
      for (int j = 0; j < 4; j++) {
	temp.push_back(lines_std[i][j]);
      }

      points.push_back(temp);
      temp.clear();
    }

    for (int i = 0; i < lines_std.size(); ++i) {
      int x1 = lines_std[i][0];
      int x2 = lines_std[i][2];
      int y1 = lines_std[i][1];
      int y2 = lines_std[i][3];
      cv::line(drawnLines, cv::Point(x1, y1), cv::Point(x2, y2),
	       CV_RGB(255, 0, 0), 1, CV_AA);
    }
    // ls->drawSegments(drawnLines, lines_std);
    // cv::imwrite("/Users/kolja/Projects/ml-prakt/data/test" + ".png",
    // drawnLines);
    // cv::imshow("Lines", drawnLines);
    cout << "Detected:" << lines_std.size() << endl;
    cout << "Filtered:" << points.size() << endl;
  }
  void makeLines(int flag) {
    // to solve Ax = b for x
    A = zeros<mat>(points.size(), 2);
    b = zeros<mat>(points.size(), 1);

    // convert given end-points of line segment into a*x + b*y = c format
    // for calculations  do for each line segment detected
    for (int i = 0; i < points.size(); i++) {
      A(i, 0) = -(points[i][3] - points[i][1]);  //-(y2-y1)
      A(i, 1) = (points[i][2] - points[i][0]);   // x2-x1
      b(i, 0) = A(i, 0) * points[i][0] +
		A(i, 1) * points[i][1];  //-(y2-y1)*x1 + (x2-x1)*y1
    }
  }

  // estimate the vanishing point
  void eval(std::string output_path) {
    // stores the estimated co-ordinates of the vanishing point with respect
    // to the image
    soln = zeros<mat>(2, 1);

    // initialize error
    double err = 9999999999;

    // calculate point of intersection of every pair of lines and
    // find the sum of distance from all other lines
    // select the point which has the minimum sum of distance
    for (int i = 0; i < points.size(); i++) {
      for (int j = 0; j < points.size(); j++) {
	if (i >= j) continue;

	// armadillo vector
	uvec indices;

	// store indices of lines to be used for calculation
	indices << i << j;

	// extract the rows with indices specified in uvec indices
	// stores the ith and jth row of matrices A and b into Atemp and
	// btemp respectively  hence creating a 2x2 matrix for
	// calculating point of intersection
	Atemp = A.rows(indices);
	btemp = b.rows(indices);

	// if lines are parallel then skip
	if (arma::rank(Atemp) != 2) continue;

	// solves for 'x' in A*x = b
	res = calc(Atemp, btemp);

	if (res.n_rows == 0 || res.n_cols == 0) continue;

	// calculate error assuming perfect intersection is
	error = A * res - b;

	// reduce size of error
	error = error / 1000;

	// to store intermediate error values
	temperr = 0;
	// summation of errors
	for (int i = 0; i < error.n_rows; i++)
	  temperr += (error(i, 0) * error(i, 0)) / 1000;

	// scale errors to prevent any overflows
	temperr /= 1000000;

	// if current error is smaller than previous min error then
	// update the solution (point)
	if (err > temperr && (!has_bounding_box || !is_in_bounding_box(soln))) {
	  soln = res;
	  err = temperr;
	}
      }
    }

    cout << "\n\nResult:\n"
	 << soln(0, 0) << "," << soln(1, 0) << "\nError:" << err << "\n\n";

    double buffer = 15.f;

    int left = abs(std::min(double(0), soln(0, 0) - buffer));
    int right =
	std::max(double(image.cols), soln(0, 0) + buffer) - double(image.cols);
    int top = abs(std::min(double(0), soln(1, 0) - buffer));
    int bottom =
	std::max(double(image.rows), soln(1, 0) + buffer) - double(image.rows);

    cout << "Deltas: " << left << " " << right << " " << top << " " << bottom
	 << endl;

    cv::Mat testimg(image.rows + bottom + top, image.cols + left + right,
		    CV_8UC1);
    cv::copyMakeBorder(image, testimg, top, bottom, left, right,
		       BORDER_CONSTANT, Scalar(255));

    cv::circle(testimg, Point(soln(0, 0) + left, soln(1, 0) + top), 10,
	       cv::Scalar(0, 0, 0), 5);

    cv::imwrite(output_path, testimg);

    // flush the vector
    points.clear();

    // toDo: use previous frame's result to reduce calculations and
    // stabilize the region of vanishing point
    prevRes = soln;
  }

  // function to calculate and return the intersection point
  mat calc(mat A, mat b) {
    mat x = zeros<mat>(2, 1);
    solve(x, A, b);
    return x;
  }
};

std::vector<std::string> tokenize(std::string input) {
  std::stringstream ss(input);
  std::vector<string> output;
  std::string token;

  while (std::getline(ss, token, ',')) {
    output.push_back(token);
  }

  return output;
}

std::string extract_filename(const std::string &path) {
  return std::string(
      std::find_if(path.rbegin(), path.rend(), MatchPathSeparator()).base(),
      path.end());
}

std::vector<std::tuple<std::string, int, int, int, int>> read_config(
    std::string path) {
  std::vector<std::tuple<std::string, int, int, int, int>> result;
  ifstream f(path);
  std::string line;

  if (f.is_open()) {
    while (std::getline(f, line)) {
      std::vector<std::string> tokens = tokenize(line);
      result.push_back(
	  std::make_tuple(tokens[0], std::stoi(tokens[1]), std::stoi(tokens[2]),
			  std::stoi(tokens[3]), std::stoi(tokens[4])));
    }
    f.close();
  }

  return result;
}

int main(int argc, char *argv[]) {
  if (argc < 2) {
    cerr << "Specify path to images folder." << endl;
    return -1;
  }
  std::string images_path = argv[1];

  std::vector<std::tuple<std::string, int, int, int, int>> cfg =
      read_config(images_path + "/config.txt");

  VanishingPointDetector vp_detector;
  for (auto &entry : cfg) {
    cv::Point p1(std::get<1>(entry), std::get<2>(entry));
    cv::Point p2(std::get<3>(entry), std::get<4>(entry));

    vp_detector.set_bounding_box(p1, p2);
    vp_detector.process_image(images_path + "/" + std::get<0>(entry),
			      images_path + "/out/" + std::get<0>(entry));
  }

  cv::destroyAllWindows();
  return 0;
}
