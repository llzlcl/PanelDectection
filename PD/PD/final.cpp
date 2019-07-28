//#include <vector>
//#include "stdlib.h"
#include <iostream>  
#include <opencv2/core/core.hpp>  
#include <opencv2/highgui/highgui.hpp>  
#include <opencv2/imgproc/imgproc.hpp>


using namespace cv;
using namespace std;

void readPic();
void readVid();
Mat showline(vector<Vec4i> lines, Mat &img, String s);
Mat showline(vector<Vec4i> lines, Mat &img, Mat &bg, String s);
Mat showline(Vec4i lines, Mat &img, String s);
Mat showpoint(Point p, Mat &img, String s);
Mat showpoint(vector<Point> p, Mat &img, String s);
void postest(vector<Vec4i> pos, Mat bg);

void normal(Mat &img0);
bool lightspot(Mat &img0);
bool ground(Mat &img0);
void skel(Mat &srcimage);
void show(Mat &img, const String &str);
void minBlur(Mat &src, int size);
double getAngle(Vec4i p, Size size);
double dstns(Vec4i l1, Vec4i l2, Vec4i lv);
vector<Vec4i> combine(vector<Vec4i> lines, int thresh);
Point intersection(Vec4i l1, Vec4i l2);
int* intersection(Vec4i l1, Vec4i l2, Size size);
vector<Vec4i> accuracy(vector<Vec4i> lines, Mat &bg, Size size); 
vector<Vec4i> select(vector<Vec4i> lines, double angle); 
Vec4i pointAngle(Point p, double angle, Size size); 
Vec4i extension(Vec4i lines, Size size);
vector<Vec4i> extension(vector<Vec4i> lines, Size size);
void cont(Mat &img);
vector<Vec4i> hLines(Mat &img, Size size);
vector<Vec4i> fill(vector<Vec4i> lines, Vec4i lv, Size size, Mat &bg);
vector<Point> selectp(vector<Point> ep, vector<Point>ps, vector<Point>nps, int thresh); //重复点消除
vector<Point> selectp(vector<Point> ep, int thresh);
void savePic(Mat &img);
vector<Vec4i> getLlines(Vec4i l1, Vec4i l2, Size size);
vector<Vec4i> sortLines(vector<Vec4i> lines, Vec4i lv, double angle);
vector<vector<Vec4i>> fitLines(vector<Vec4i> lines, vector<Vec4i> hlines, int thresh);
vector<Vec4i> getPos(vector<vector<Vec4i>> lLines, vector<vector<Vec4i>> sLines);
vector<Vec4i> creatMap(Size size, Vec2i ab);
Mat dataMat(vector<Vec4i> pos, Vec2i ab);
vector<Vec4i> trans(vector<Vec4i> pos, Size size);
Mat xyMat(vector<Vec4i> pos);
Mat tMat(vector<Vec4i> pos, Vec2i ab);
Mat draw(Mat map, vector<Vec4i> pos, Vec2i ab);
Mat draw(Mat map, vector<Vec4i> pos);
vector<Vec4i> getPos(vector<Vec4i> lLines, vector<Vec4i> sLines);
vector<Vec4i> transLines(vector<Vec4i> map, Mat transMat);
bool ifErr(vector<Vec4i> lines);
bool flag = false;
int num = 0;
int main()
{	
	if (0) readPic();
	else readVid();
}

Mat iimg;
void normal(Mat &img0)
{	
	Mat img;
	resize(img0, img, Size((int)img0.cols / 4, (int)img0.rows / 4));
	String s = to_string(num);
	iimg = img.clone();
	Mat ch2 = img.clone();
	Size size = img.size();
	Mat ch[3], hsv[3];
	split(img, ch);
	cvtColor(img, img, CV_BGR2HSV);
	split(img, hsv);
	Mat map = Mat::zeros(size, CV_8UC1);

	if (lightspot(img0))
	{
		for (int i = 0; i < size.height; i++)
		{
			uchar* h = hsv[0].ptr(i);
			uchar* s = hsv[1].ptr(i);
			uchar* v = hsv[2].ptr(i);
			uchar* mptr = map.ptr(i);
			//cout << hsv[1] << endl;
			for (int j = 0; j < size.width; j++)
			{
				if ((h[j] > 90 & h[j] <120 & s[j] >30 & s[j]<200 & v[j] >172) | (s[j] <10 & v[j] >250))
					mptr[j] = 255;
			}
		}
		Mat element = getStructuringElement(MORPH_ELLIPSE, Size(6, 6));
		dilate(map, map, element);
		element = getStructuringElement(MORPH_ELLIPSE, Size(8, 8));
		erode(map, map, element);
	}
	else
	{
		for (int i = 0; i < size.height; i++)
		{
			uchar* h = hsv[0].ptr(i);
			uchar* s = hsv[1].ptr(i);
			uchar* v = hsv[2].ptr(i);
			uchar* mptr = map.ptr(i);
			//cout << hsv[1] << endl;
			for (int j = 0; j < size.width; j++)
			{
				if ((h[j] > 90 & h[j] <120 & s[j] >30 & v[j] >130) | (s[j] <10 & v[j] >250))
					mptr[j] = 255;
			}
		}
		Mat element = getStructuringElement(MORPH_ELLIPSE, Size(4, 4));
		dilate(map, map, element);
		element = getStructuringElement(MORPH_ELLIPSE, Size(14, 14));
		erode(map, map, element);
		element = getStructuringElement(MORPH_ELLIPSE, Size(10, 10));
		dilate(map, map, element);
	}
	
	vector<vector<Point>> contours;
	findContours(map, contours, 0, CV_RETR_LIST);
	Mat b = Mat::zeros(img.size(), CV_8UC1);
	for (int t = 0; t < contours.size(); t++)
	{
		Mat white(img.size(), CV_8UC1, Scalar(255));
		Mat black = Mat::zeros(img.size(), CV_8UC1);
		if (contourArea(contours[t]) < 9000) continue;
		Mat bg = Mat::zeros(size, CV_8UC1); 
		drawContours(bg, contours, t, Scalar(255), 1);
		vector<Vec4i> cLines;
		HoughLinesP(bg, cLines, 1, CV_PI / 180, 10, 40, 40);
		cLines = combine(cLines, 20);
		if (cLines.size() == 0) continue;
		drawContours(bg, contours, t, Scalar(255), -1);
		Mat mPanel;
		ch[2].copyTo(mPanel, bg);
		for (int i = cLines.size() - 1; i >= 0; i--)
		{
			double angle = getAngle(cLines[i], size);
			if (angle == 201 | angle == 202 | angle == 203 | angle == 204)
				cLines.erase(cLines.begin() + i);
		}
		vector<Vec4i> resultLines;
		if (cLines.size() == 2)
		{
			double a1 = getAngle(cLines[0], size), a2 = getAngle(cLines[1], size);
			if (abs(a1 - a2) < 20| abs(a1 - a2) >160)
			{
				vector<Vec4i> lLines = getLlines(cLines[0], cLines[1], size);
				vector<Vec4i> hLines1 = hLines(mPanel, size);
				vector<Vec4i> hLines2 = combine(hLines1, 16);
				hLines2 = extension(hLines2, size);
				hLines2 = select(hLines2, (a1 + a2) / 2);
				hLines2 = accuracy(hLines2, ch[2], size);
				hLines2 = fill(hLines2, lLines[2], size, ch[2]);
				hLines2 = accuracy(hLines2, ch[2], size);
				lLines = accuracy(lLines, ch[2], size);
				vector<Vec4i> pos = getPos(lLines, hLines2);
				if (pos.size() > 3)
				{
					resultLines = trans(pos, size);
				}
			}
		}
		else if (cLines.size() == 3)
		{
			vector<Vec4i> lLines;
			double a1 = getAngle(cLines[0], size), a2 = getAngle(cLines[1], size), a3 = getAngle(cLines[2], size);
			int a;
			if (abs(a1 - a2) < 20| abs(a1 - a2) >160)
			{
				lLines = getLlines(cLines[0], cLines[1], size);
				a = (a1 + a2) / 2;
			}
			else if (abs(a1 - a3) < 20| abs(a1 - a3) >160)
			{
				lLines = getLlines(cLines[0], cLines[2], size);
				a = (a1 + a3) / 2;
			}
			else if (abs(a2 - a3) < 20| abs(a2 - a3)>160)
			{
				lLines = getLlines(cLines[1], cLines[2], size);
				a = (a2 + a3) / 2;
			}
			else continue;
			vector<Vec4i> hLines1 = hLines(mPanel, size);
			vector<Vec4i> hLines2 = combine(hLines1, 16);
			hLines2 = extension(hLines2, size);
			hLines2 = select(hLines2, a);
			hLines2 = accuracy(hLines2, ch[2], size);
			hLines2 = fill(hLines2, lLines[2], size, ch[2]);
			hLines2 = accuracy(hLines2, ch[2], size);
			lLines = accuracy(lLines, ch[2], size);
			vector<Vec4i> pos = getPos(lLines, hLines2);
			if (pos.size() > 3)
			{
				resultLines = trans(pos, size);
			}
		}
		if (resultLines.size() == 0) continue;
		if (ifErr(resultLines))
		{
			cout << num << endl;
			continue;
		}
		flag = true;
		for (int i = 0; i < resultLines.size(); i++)
		{
			Vec4i l = resultLines[i];
			line(black, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255), 2, LINE_AA);
		}
		Mat bg2 = bg.clone();
		Mat element = getStructuringElement(MORPH_ELLIPSE, Size(20, 20));
		dilate(bg2, bg2, element);
		showline(resultLines, ch2, bg, "result");
		black = black & bg2;
		white.copyTo(b, black);
		show(b, "bg2");
	}
	if (flag)
	{
		String name = "C:\\Users\\97306\\Desktop\\原图\\" + s + " 原图.jpg";
		imwrite(name, iimg);
		name = "C:\\Users\\97306\\Desktop\\原图\\" + s + " 标注.jpg";
		threshold(b, b, 150, 255, 1);
		imwrite(name, b);
		name = "C:\\Users\\97306\\Desktop\\原图\\" + s + " 结果.jpg";
		imwrite(name, ch2);
	}
	else
	{
		String name = "C:\\Users\\97306\\Desktop\\原图2\\" + s + " 原图.jpg";
		imwrite(name, iimg);
		name = "C:\\Users\\97306\\Desktop\\原图2\\" + s + " 标注.jpg";
		threshold(b, b, 150, 255, 1);
		imwrite(name, b);
		name = "C:\\Users\\97306\\Desktop\\原图2\\" + s + " 结果.jpg";
		imwrite(name, ch2);
	}
	resize(ch2, ch2, img0.size());
	img0 = ch2;
}
bool ground(Mat &img0)
{
	Mat img, channel[3];
	resize(img0, img, Size((int)img0.cols / 5, (int)img0.rows / 5));
	split(img, channel);
	Size size = img.size();
	Mat element, bg, bg1;
	bg = channel[1] - channel[2];
	threshold(bg, bg1, 0, 255, THRESH_OTSU);
	threshold(bg, bg, 35, 255, THRESH_BINARY);
	bg = bg&bg1;
	element = getStructuringElement(MORPH_RECT, Size(3, 3));
	erode(bg, bg, element);
	for (int i = 0; i < size.height; i++)
	{
		uchar* p = bg.ptr<uchar>(i);
		for (int j = 0; j < size.width; j++)
			if (p[j]>0) return false;
	}
	return true;
}
bool lightspot(Mat &img0)
{
	Mat img, channel[3];
	split(img0, channel);
	Mat element, bg, bg1;
	resize(img0, img, Size((int)img0.cols / 5, (int)img0.rows / 5));
	Size size = img.size();
	Mat gray;
	cvtColor(img, gray, CV_RGB2GRAY);
	threshold(gray, gray, 254, 255, THRESH_BINARY);
	element = getStructuringElement(MORPH_RECT, Size(8, 8));
	erode(gray, gray, element);
	for (int i = 0; i < size.height; i++)
	{
		uchar* p = gray.ptr<uchar>(i);
		for (int j = 0; j < size.width; j++)
			if (p[j]>0) return true;
	}
	return false;
}
void show(Mat &img, const String &str)
{
	namedWindow(str, CV_WINDOW_NORMAL);
	imshow(str, img);
	//imwrite("C:\\Users\\97306\\Desktop\\"+str + ".jpg", img);
	if (waitKey(1) == 32)
	{
		if (waitKey(0) == 32);
	}
}
void minBlur(Mat &src, int size)
{
	int rowNumber = src.rows;
	int colNumber = src.cols;
	Mat dst = src.clone();
	for (int i = 1; i < rowNumber - 1; i++)
	{
		for (int j = 1; j < colNumber - 1; j++)
		{
			int i0 = size, j0 = size;
			if (i0 + i >= rowNumber) i0 = rowNumber - i - 1;
			if (j0 + j >= colNumber) j0 = colNumber - j - 1;
			Rect r(j, i, j0, i0);
			double min;
			minMaxLoc(src(r), &min, NULL, NULL, NULL);
			dst.at<uchar>(i, j) = (uchar)min;
		}
	}
	src = dst;
}
void skel(Mat &srcimage)
{
	vector<Point> deletelist1;
	int Zhangmude[9];
	int nl = srcimage.rows;
	int nc = srcimage.cols;
	while (true)
	{

		for (int j = 1; j < (nl - 1); j++)
		{
			uchar* data_last = srcimage.ptr<uchar>(j - 1);
			uchar* data = srcimage.ptr<uchar>(j);
			uchar* data_next = srcimage.ptr<uchar>(j + 1);

			for (int i = 1; i < (nc - 1); i++)
			{
				if (data[i] == 255)
				{
					Zhangmude[0] = 1;
					if (data_last[i] == 255) Zhangmude[1] = 1;
					else  Zhangmude[1] = 0;
					if (data_last[i + 1] == 255) Zhangmude[2] = 1;
					else  Zhangmude[2] = 0;
					if (data[i + 1] == 255) Zhangmude[3] = 1;
					else  Zhangmude[3] = 0;
					if (data_next[i + 1] == 255) Zhangmude[4] = 1;
					else  Zhangmude[4] = 0;
					if (data_next[i] == 255) Zhangmude[5] = 1;
					else  Zhangmude[5] = 0;
					if (data_next[i - 1] == 255) Zhangmude[6] = 1;
					else  Zhangmude[6] = 0;
					if (data[i - 1] == 255) Zhangmude[7] = 1;
					else  Zhangmude[7] = 0;
					if (data_last[i - 1] == 255) Zhangmude[8] = 1;
					else  Zhangmude[8] = 0;
					int whitepointtotal = 0;
					for (int k = 1; k < 9; k++)
					{
						whitepointtotal = whitepointtotal + Zhangmude[k];
					}
					if ((whitepointtotal >= 2) && (whitepointtotal <= 6))
					{
						int ap = 0;
						if ((Zhangmude[1] == 0) && (Zhangmude[2] == 1)) ap++;
						if ((Zhangmude[2] == 0) && (Zhangmude[3] == 1)) ap++;
						if ((Zhangmude[3] == 0) && (Zhangmude[4] == 1)) ap++;
						if ((Zhangmude[4] == 0) && (Zhangmude[5] == 1)) ap++;
						if ((Zhangmude[5] == 0) && (Zhangmude[6] == 1)) ap++;
						if ((Zhangmude[6] == 0) && (Zhangmude[7] == 1)) ap++;
						if ((Zhangmude[7] == 0) && (Zhangmude[8] == 1)) ap++;
						if ((Zhangmude[8] == 0) && (Zhangmude[1] == 1)) ap++;
						if (ap == 1)
						{
							if ((Zhangmude[1] * Zhangmude[7] * Zhangmude[5] == 0) && (Zhangmude[3] * Zhangmude[5] * Zhangmude[7] == 0))
							{
								deletelist1.push_back(Point(i, j));
							}
						}
					}
				}
			}
		}
		if (deletelist1.size() == 0) break;
		for (size_t i = 0; i < deletelist1.size(); i++)
		{
			Point tem;
			tem = deletelist1[i];
			uchar* data = srcimage.ptr<uchar>(tem.y);
			data[tem.x] = 0;
		}
		deletelist1.clear();

		for (int j = 1; j < (nl - 1); j++)
		{
			uchar* data_last = srcimage.ptr<uchar>(j - 1);
			uchar* data = srcimage.ptr<uchar>(j);
			uchar* data_next = srcimage.ptr<uchar>(j + 1);

			for (int i = 1; i < (nc - 1); i++)
			{
				if (data[i] == 255)
				{
					Zhangmude[0] = 1;
					if (data_last[i] == 255) Zhangmude[1] = 1;
					else  Zhangmude[1] = 0;
					if (data_last[i + 1] == 255) Zhangmude[2] = 1;
					else  Zhangmude[2] = 0;
					if (data[i + 1] == 255) Zhangmude[3] = 1;
					else  Zhangmude[3] = 0;
					if (data_next[i + 1] == 255) Zhangmude[4] = 1;
					else  Zhangmude[4] = 0;
					if (data_next[i] == 255) Zhangmude[5] = 1;
					else  Zhangmude[5] = 0;
					if (data_next[i - 1] == 255) Zhangmude[6] = 1;
					else  Zhangmude[6] = 0;
					if (data[i - 1] == 255) Zhangmude[7] = 1;
					else  Zhangmude[7] = 0;
					if (data_last[i - 1] == 255) Zhangmude[8] = 1;
					else  Zhangmude[8] = 0;
					int whitepointtotal = 0;
					for (int k = 1; k < 9; k++)
					{
						whitepointtotal = whitepointtotal + Zhangmude[k];
					}
					if ((whitepointtotal >= 2) && (whitepointtotal <= 6))
					{
						int ap = 0;
						if ((Zhangmude[1] == 0) && (Zhangmude[2] == 1)) ap++;
						if ((Zhangmude[2] == 0) && (Zhangmude[3] == 1)) ap++;
						if ((Zhangmude[3] == 0) && (Zhangmude[4] == 1)) ap++;
						if ((Zhangmude[4] == 0) && (Zhangmude[5] == 1)) ap++;
						if ((Zhangmude[5] == 0) && (Zhangmude[6] == 1)) ap++;
						if ((Zhangmude[6] == 0) && (Zhangmude[7] == 1)) ap++;
						if ((Zhangmude[7] == 0) && (Zhangmude[8] == 1)) ap++;
						if ((Zhangmude[8] == 0) && (Zhangmude[1] == 1)) ap++;
						if (ap == 1)
						{
							if ((Zhangmude[1] * Zhangmude[3] * Zhangmude[5] == 0) && (Zhangmude[3] * Zhangmude[1] * Zhangmude[7] == 0))
							{
								deletelist1.push_back(Point(i, j));
							}
						}
					}
				}
			}
		}
		if (deletelist1.size() == 0) break;
		for (size_t i = 0; i < deletelist1.size(); i++)
		{
			Point tem;
			tem = deletelist1[i];
			uchar* data = srcimage.ptr<uchar>(tem.y);
			data[tem.x] = 0;
		}
		deletelist1.clear();
	}
}
int getAngle(Vec4i p)
{
	if ((p[0] - p[2]) == 0)
		return 90;
	else
	{
		double result = atan2((double)(p[1] - p[3]), (double)(p[0] - p[2])) * 180.0 / 3.14159;
		if (result > 0) return (int)result;
		else return (int)result + 180;
	}
}
double getAngle(Vec4i p, Size size)
{
	//垂直90 左201 下2 右3 上4
	if (p[0] <= 10 & p[2] <= 10)
		return 201;
	else if (p[0] >= size.width - 100 & p[2] >= size.width - 10)
		return 203;
	else if ((p[0] - p[2]) == 0)
		return 90;
	else if (p[1] <= 10 & p[3] <= 10)
		return 204;
	else if (p[1] >= size.height - 10 & p[3] >= size.height - 10)
		return 202;
	else
	{
		double result = atan2((double)(p[1] - p[3]), (double)(p[0] - p[2])) * 180.0 / 3.14159;
		if (result > 0)return result;
		else return result + 180;
	}
}
Point intersection(Vec4i l1, Vec4i l2)
{
	int x1 = l1[0], x2 = l1[2], x3 = l2[0], x4 = l2[2], y1 = l1[1], y2 = l1[3], y3 = l2[1], y4 = l2[3];
	if ((-(x3*y1) + x4*y1 + x3*y2 - x4*y2 + x1*y3 - x2*y3 - x1*y4 + x2*y4) == 0 |
		(x3*y1 - x4*y1 - x3*y2 + x4*y2 - x1*y3 + x2*y3 + x1*y4 - x2*y4) == 0)
		return Point(99999, 99999);
	int x = -(x2*x3*y1 - x2*x4*y1 - x1*x3*y2 + x1*x4*y2 - x1*x4*y3 + x2*x4*y3 +
		x1*x3*y4 - x2*x3*y4) / (-x3*y1 + x4*y1 + x3*y2 - x4*y2 + x1*y3 -
			x2*y3 - x1*y4 + x2*y4);
	int y = -(-x2*y1*y3 + x4*y1*y3 + x1*y2*y3 - x4*y2*y3 + x2*y1*y4 -
		x3*y1*y4 - x1*y2*y4 + x3*y2*y4) / (x3*y1 - x4*y1 - x3*y2 + x4*y2 -
			x1*y3 + x2*y3 + x1*y4 - x2*y4);
	Point result(x, y);
	return result;
}
int* intersection(Vec4i l1, Vec4i l2, Size size)
{
	int x1 = l1[0], x2 = l1[2], x3 = l2[0], x4 = l2[2], y1 = l1[1], y2 = l1[3], y3 = l2[1], y4 = l2[3];
	if ((-(x3*y1) + x4*y1 + x3*y2 - x4*y2 + x1*y3 - x2*y3 - x1*y4 + x2*y4) == 0 |
		(x3*y1 - x4*y1 - x3*y2 + x4*y2 - x1*y3 + x2*y3 + x1*y4 - x2*y4) == 0)
	{
		int result[] = { 0,0,0 };
		return result;
	}
	int x = -(x2*x3*y1 - x2*x4*y1 - x1*x3*y2 + x1*x4*y2 - x1*x4*y3 + x2*x4*y3 +
		x1*x3*y4 - x2*x3*y4) / (-(x3*y1) + x4*y1 + x3*y2 - x4*y2 + x1*y3 -
			x2*y3 - x1*y4 + x2*y4);
	int y = -(-(x2*y1*y3) + x4*y1*y3 + x1*y2*y3 - x4*y2*y3 + x2*y1*y4 -
		x3*y1*y4 - x1*y2*y4 + x3*y2*y4) / (x3*y1 - x4*y1 - x3*y2 + x4*y2 -
			x1*y3 + x2*y3 + x1*y4 - x2*y4);
	int z; //左1 下2 右3 上4
	if (x < 0) z = 1;
	else if (y > size.height - 1) z = 2;
	else if (x > size.width - 1) z = 3;
	else if (y < 0)z = 4;
	else z = 5;
	int result[] = { x,y,z };
	return result;
}
vector<Vec4i> extension(vector<Vec4i> lines, Size size)
{
	Vec4i up = { 1,1,size.width - 2,1 }, down = { 1,size.height - 2,size.width - 2 ,size.height - 2 },
		left = { 1,1,1,size.height - 2 }, right = { size.width - 2,1,size.width - 2 ,size.height - 2 };
	vector<Vec4i> edges = { up,left,down,right };
	vector<Vec4i> nl;
	for (int i = 0; i < lines.size(); i++)
	{	
		int* p;
		vector<int> np;
		for (int j = 0; j < edges.size(); j++)
		{
			p = intersection(lines[i], edges[j], size);
			if (p[2] == 5)
			{
				Point q = intersection(lines[i], edges[j]);
				np.push_back(q.x);
				np.push_back(q.y);
			}
		}
		if (np.size() == 4)
		{
			Vec4i n = { np[0], np[1], np[2], np[3] };
			nl.push_back(n);
		}
	}
	return nl;
}
Vec4i extension(Vec4i lines, Size size)
{
	Vec4i up = { 1,1,size.width - 2,1 }, down = { 1,size.height - 2,size.width - 2 ,size.height - 2 },
		left = { 1,1,1,size.height - 2 }, right = { size.width - 2,1,size.width - 2 ,size.height - 2 };
	vector<Vec4i> edges = { up,left,down,right };
	int* p;
	vector<int> np;
	vector<Point> q;
	for (int j = 0; j < edges.size(); j++)
	{
		p = intersection(lines, edges[j], size);
		if (p[2] == 5)
		{
			q.push_back(intersection(lines, edges[j]));
		}
	}
	q = selectp(q, 10);
	if (q.size() >= 2)
	{
		Vec4i n = { q[0].x, q[0].y, q[1].x, q[1].y };
		return n;
	}
}
Vec4i pointAngle(Point p, double angle, Size size)
{
	int x = p.x + 100, y = p.y + tan(angle / 180.0* 3.14159) * 100;
	Vec4i l = { p.x,p.y,x,y };
	return extension(l, size);
}
vector<Vec4i> accuracy(vector<Vec4i> lines, Mat &bg0, Size size)
{
	vector<Vec4i> ls;
	for (int i = 0; i < lines.size(); i++)
	{
		double a = getAngle(lines[i], size);
		Point p = { (lines[i][0] + lines[i][2]) / 2,(lines[i][1] + lines[i][3]) / 2 };
		if ((a >= 45 & a <= 88) | (a >= 92 & a <= 135))
		{ 
			double f1[9];
			int temp = p.x;
			for (int j = -4; j < 5; j++)
			{
				Mat bg = bg0.clone();
				p.x = temp + j;
				Vec4i l = pointAngle(p, a, size);
				//lookinto(bg, l);
				int y1 = l[1], y2 = l[3], x1 = l[0];
				if (y1 > y2)
				{
					y1 = l[3];
					y2 = l[1];
					x1 = l[2];
				}
				double sum = 0;
				int num = 0;
				for (int y = y1; y <= y2; y++)
				{
					int x = (y - y1) / tan(a / 180.0* 3.14159) + x1;
					if (x >= size.width) x = size.width - 1;
					if (x <= 0) x = 1;
					const uchar* ptr = bg0.ptr<uchar>(y);
					if (ptr[x] != 0)
					{
						sum += ptr[x];
						num++;
					}
				}
				f1[j + 4] = sum / num;
			}
			int maxp = 0;
			for (int i = 0; i < 9; i++)
			{
				if (f1[maxp] < f1[i])
					maxp = i;
			}
			p.x = temp + maxp - 4;
			ls.push_back(pointAngle(p, a, size));

		}
		else if (a > 88 & a < 92)
		{
			double f1[9];
			for (int j = -4; j < 5; j++)
			{
				Mat bg = bg0.clone();
				double sum = 0;
				int num = 0;
				for (int y = min(lines[i][1], lines[i][3]); y <= max(lines[i][1], lines[i][3]); y++)
				{
					const uchar* ptr = bg0.ptr<uchar>(y);
					if (ptr[lines[i][0] + j] != 0)
					{
						sum += ptr[lines[i][0] + j];
						num++;
					}
				}
				f1[j + 4] = sum / num;
			}
			int maxp = 0;
			for (int i = 0; i < 9; i++)
			{
				if (f1[maxp] < f1[i])
					maxp = i;
			}
			lines[i][0] = lines[i][0] + maxp - 4;
			lines[i][2] = lines[i][2] + maxp - 4;
			ls.push_back(lines[i]);

		}
		else if ((a >= 0 & a < 45) | (a > 135 & a < 181))
		{
			double f1[9];
			int temp = p.y;
			for (int j = -4; j < 5; j++)
			{
				Mat bg = bg0.clone();
				p.y = temp + j;
				Vec4i l = pointAngle(p, a, size);
				//lookinto(bg, l);
				int x1 = l[0], x2 = l[2], y1 = l[1];
				if (x1 > x2)
				{
					x1 = l[2];
					x2 = l[0];
					y1 = l[3];
				}
				double sum = 0;
				int num = 0;
				for (int x = x1; x <= x2; x++)
				{
					int y = (x - x1)*tan(a / 180.0* 3.14159) + y1;
					if (y >= size.height) y = size.height - 1;
					if (y <= 0) y = 1;
					const uchar* ptr = bg0.ptr<uchar>(y);
					if (ptr[x] != 0)
					{
						sum += ptr[x];
						num++;
					}
				}
				f1[j + 4] = sum / num;
			}
			int maxp = 0;
			for (int i = 0; i < 9; i++)
			{
				if (f1[maxp] < f1[i])
					maxp = i;
			}
			p.y = temp + maxp - 4;
			ls.push_back(pointAngle(p, a, size));
		}
	}
	return ls;
 }
void readVid()
{
	string filename = "RGB.mov";
	VideoCapture capture;
	capture.open(filename);
	double rate = capture.get(CV_CAP_PROP_FPS);
	//VideoWriter writer("VideoTest11.avi", CV_FOURCC('M', 'J', 'P', 'G'), rate, Size(1920, 1080));
	if (!capture.isOpened())
	{
		return;
	}
	else
	{
		while (true)
		{
			Mat frame;
			//for (int i = 0; i < 20; i++)  //为了实现实时处理，每隔几帧处理一帧。如果要每一帧都处理，可以把这一行注释掉
			capture >> frame;
			if (frame.empty()) break;
			num++;
			if (!ground(frame))
				normal(frame);
			//cont(frame);
			//writer << frame;
			//show(frame, "处理后视频");
			savePic(frame);
		}
	}
}
void readPic()
{
	String s = to_string(num);
	string filename = "C:\\Users\\97306\\Desktop\\英特尔比赛\\264.jpg";
	filename = "4.jpg";
	Mat img0 = imread(filename);
	normal(img0);
	show(img0, "处理后图像");
	waitKey(0);
}
void cont(Mat &img)
{
	num++;
	char s[5];
	_itoa_s(num, s, 10);
	putText(img, s, Point(300, 100), CV_FONT_HERSHEY_PLAIN, 10, CV_RGB(255, 0, 0), 5);
}
void savePic(Mat &img)
{
	//num++;
	String s = to_string(num);
	String name = s + ".jpg";
	cout << name << endl;
	imwrite(name, img);
}
Mat showline(vector<Vec4i> lines, Mat &img, String s)
{
	Mat img0 = img.clone();
	for (int i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];
		line(img0, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255,255,255), 2, LINE_AA);
	}
	show(img0, s);
	imwrite("C:\\Users\\97306\\Desktop\\h2.jpg", img0);

	return img0;
}
Mat showline(vector<Vec4i> lines, Mat &img, Mat &bg, String s)
{
	Mat img0 = Mat::zeros(img.size(), CV_8UC3);
	Mat red = Mat::zeros(img.size(), CV_8UC3);
	Mat element = getStructuringElement(MORPH_ELLIPSE, Size(20, 20));
	dilate(bg, bg, element);
	cvtColor(bg, bg, CV_GRAY2BGR);
	for (int i = 0; i < lines.size(); i++)
	{
		Vec4i l = lines[i];
		line(img0, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255, 255, 255), 2, LINE_AA);
		line(red, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(0, 0, 255), 2, LINE_AA);
	}
	bg = img0&bg;
	red.copyTo(img, bg);
	show(img, s);
	return img;
}
Mat showline(Vec4i lines, Mat &img, String s)
{
	Mat img0 = img.clone();
	Vec4i l = lines;
	line(img0, Point(l[0], l[1]), Point(l[2], l[3]), Scalar(255), 1, LINE_AA);
	//show(img0, s);
	return img0;
}
Mat showpoint(Point p, Mat &img,String s)
{
	Mat img0 = img.clone();
	circle(img, p, 2, Scalar(0,0,255), 2);
	show(img, s);
	return img;
}
Mat showpoint(vector<Point> p, Mat &img, String s)
{
	Mat img0 = img.clone();
	for (int i = 0; i < p.size(); i++)
		circle(img0, p[i], 2, Scalar(0), 2);
	show(img0, s);
	return img0;
}
vector<Point> selectp(vector<Point> ep, vector<Point>ps, vector<Point>nps, int thresh)
{
	for (int i = 0; i < ep.size(); i++)
	{
		bool flag = true;
		for (int j = 0; j < ps.size(); j++)
		{
			if (abs(ep[i].x - ps[j].x)<thresh & abs(ep[i].y - ps[j].y)<thresh)
				flag = false;
		}
		if (flag)
			nps.push_back(ep[i]);
	}
	return nps;
}
vector<Point> selectp(vector<Point> ep, int thresh)
{
	if (ep.size() < 2) return ep;
	vector<Point> nps;
	for (int i = 0; i < ep.size() - 1; i++)
	{
		bool flag = true;
		for (int j = i + 1; j < ep.size(); j++)
		{
			if (abs(ep[i].x - ep[j].x)<thresh & abs(ep[i].y - ep[j].y)<thresh)
				flag = false;
		}
		if (flag)
			nps.push_back(ep[i]);
	}
	nps.push_back(ep[ep.size() - 1]);
	return nps;
}
vector<Vec4i> fill(vector<Vec4i> lines, Vec4i lv, Size size, Mat &bg)
{
	double angle = 0;
	for (int i = 0; i < lines.size(); i++)
		angle += getAngle(lines[i]);
	angle = angle / lines.size();
	if (lines.size() < 2) return lines;
	int thresh = 10;
	double a = angle + 90;
	Point p = { (lines[0][0] + lines[0][2]) / 2, (lines[0][1] + lines[0][3]) / 2 };
	vector<Point> ps;
	vector<vector<int>> v;
	if (angle > 45 & angle < 135)
	{
		for (int i = 0; i < lines.size(); i++)
		{
			Point pp = intersection(lv, lines[i]);
			ps.push_back(pp);
			vector<int> temp = { pp.x,i };
			v.push_back(temp);
		}
	}
	else
	{
		for (int i = 0; i < lines.size(); i++)
		{
			Point pp = intersection(lv, lines[i]);
			ps.push_back(pp);
			vector<int> temp = { pp.y,i };
			v.push_back(temp);
		}
	}
	for (int i = 0; i < v.size() - 1; i++)
	{
		for (int j = i + 1; j < v.size(); j++)
		{
			if (v[i][0] > v[j][0])
			{
				vector<int> temp = v[i];
				v[i] = v[j];
				v[j] = temp;
			}
		}
	}
	//x或y从小到大排好序的point表
	vector<double> dstns;
	for (int i = 0; i < v.size() - 1; i++)
	{
		Point p1 = ps[v[i][1]], p2 = ps[v[i + 1][1]];
		double d = sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)* (p1.y - p2.y));
		dstns.push_back(d);
	}
	vector<int> dvd;
	double min = 50;
	for (int i = 0; i < dstns.size(); i++)
		if (min - dstns[i] > thresh | dstns[i] - min > thresh)
			dvd.push_back(i);
	cout << min << endl;
	//最小间距
	vector<Point> nps; //新加的点
	for (int i = 0; i < dvd.size(); i++)
	{
		int times = dstns[dvd[i]] / min + 0.5;
		if (times == 0) continue;
		Point p1 = ps[v[dvd[i]][1]], p2 = ps[v[dvd[i] + 1][1]];
		int x = (p1.x - p2.x) / times, y = (p1.y - p2.y) / times;
		for (int j = 1; j < times; j++)
			nps.push_back(Point(p1.x - x*j, p1.y - y*j));
	}
	vector<Vec4i> result;
	for (int i = 0; i < nps.size(); i++)
	{
		result.push_back(pointAngle(nps[i], (double)angle, size));
	}
	lines.insert(lines.end(), result.begin(), result.end());
	result = sortLines(lines, lv, angle);
	return result;
}
vector<Vec4i> combine(vector<Vec4i> lines, int thresh)
{
	if (lines.size() == 0) return lines;
	for (int i = 0; i < lines.size() - 1; i++)
	{
		vector<int> vj;
		for (int j = i + 1; j < lines.size(); j++)
		{
			Vec4i l1 = lines[i], l2 = lines[j];
			vector<Point> points;
			points.push_back(Point(l1[0], l1[1]));
			points.push_back(Point(l1[2], l1[3]));
			points.push_back(Point(l2[0], l2[1]));
			points.push_back(Point(l2[2], l2[3]));
			RotatedRect rect = minAreaRect(points);
			if (rect.size.width < thresh)
			{
				l1[0] = rect.center.x + (rect.size.height*cos((90 - rect.angle)*3.14159 / 180));
				l1[1] = rect.center.x + (rect.size.height*sin((90 - rect.angle)*3.14159 / 180));
				l1[2] = rect.center.x - (rect.size.height*cos((90 - rect.angle)*3.14159 / 180));
				l1[3] = rect.center.x - (rect.size.height*sin((90 - rect.angle)*3.14159 / 180));
				vj.push_back(j);
			}
			else if (rect.size.height < thresh)
			{
				l1[0] = rect.center.x + (rect.size.height*cos((-rect.angle)*3.14159 / 180));
				l1[1] = rect.center.x + (rect.size.height*sin((-rect.angle)*3.14159 / 180));
				l1[2] = rect.center.x - (rect.size.height*cos((-rect.angle)*3.14159 / 180));
				l1[3] = rect.center.x - (rect.size.height*sin((-rect.angle)*3.14159 / 180));
				vj.push_back(j);
			}
		}
		for (int k = vj.size() - 1; k >= 0; k--)
		{
			lines.erase(lines.begin() + vj[k]);
		}
	}
	return lines;
}
vector<Vec4i> select(vector<Vec4i> lines, double angle)
{
	if (lines.size() < 7) return lines;
	vector<vector<int>> vl;
	vector<int> l1 = { getAngle(lines[0]),0 };
	vl.push_back(l1);
	for (int i = 1; i < lines.size(); i++)
	{
		int a = getAngle(lines[i]);
		bool flag = true;
		for (int j = 0; j < vl.size(); j++)
		{
			if ((a - vl[j][0] < 10 & vl[j][0] - a < 10) | (a < 10 & vl[j][0] > 170 & a + 170 - vl[j][0] < 0)
				| (vl[j][0] < 10 & a > 170 & vl[j][0] + 170 - a < 0))
			{
				vl[j].push_back(i);
				vl[j][0] = (vl[j][0] * (vl.size() - 1) + a) / vl.size();
				flag = false;
			}
		}
		if (flag)
		{
			vector<int> l = { getAngle(lines[i]),i };
			vl.push_back(l);
		}
	}
	vector<int> sz;
	for (int i = 0; i < vl.size(); i++)
		sz.push_back(vl[i].size());
	int n1 = 0, n2 = 0;
	for (int i = 0; i < sz.size(); i++)
		if (sz[i] >= sz[n1]) n1 = i;
	int i = 0;
	if (n1 == 0)
	{
		i = 1;
		n2 = 1;
	}
	for (; i < sz.size(); i++)
		if (sz[i] >= sz[n2] & i != n1) n2 = i;
	vector<Vec4i> selected;

	if (abs(vl[n1][0] - angle) > 20)
		for (int i = 1; i < vl[n1].size(); i++)
			selected.push_back(lines[vl[n1][i]]);
	else
		for (int i = 1; i < vl[n2].size(); i++)
			selected.push_back(lines[vl[n2][i]]);
	return selected;
}
vector<Vec4i> hLines(Mat &img,Size size)
{
	Mat imgcp = img.clone();
	blur(img, img, Size(3, 3));
	minBlur(img, 4);
	imwrite("C:\\Users\\97306\\Desktop\\b2.jpg", img);
	Mat imgBlur;
	blur(img, imgBlur, Size(5, 5));
	img = img - imgBlur;
	threshold(img, img, 17, 255, THRESH_BINARY);
	Mat element = getStructuringElement(MORPH_RECT, Size(2, 2));
	dilate(img, img, element);
	skel(img);
	vector<Vec4i> lines;
	HoughLinesP(img, lines, 1, CV_PI / 180, 7, 10, 20);
	Mat m = Mat::zeros(size, CV_8UC1);
	return lines;
}
vector<Vec4i> getLlines(Vec4i l1, Vec4i l2, Size size)
{
	l1 = extension(l1, size);
	l2 = extension(l2, size);
	double a = (getAngle(l1) + getAngle(l2)) / 2;
	if ((a > 0 & a < 45) | (a>135 & a < 180))
	{
		if (l1[0] > l1[2])
		{
			int temp = l1[0];
			l1[0] = l1[2];
			l1[2] = temp;
			temp = l1[1];
			l1[1] = l1[3];
			l1[3] = temp;
		}
		if (l2[0] > l2[2])
		{
			int temp = l2[0];
			l2[0] = l2[2];
			l2[2] = temp;
			temp = l2[1];
			l2[1] = l2[3];
			l2[3] = temp;
		}
	}
	else if (a > 45 & a < 135)
	{
		if (l1[1] > l1[3])
		{
			int temp = l1[0];
			l1[0] = l1[2];
			l1[2] = temp;
			temp = l1[1];
			l1[1] = l1[3];
			l1[3] = temp;
		}
		if (l2[1] > l2[3])
		{
			int temp = l2[0];
			l2[0] = l2[2];
			l2[2] = temp;
			temp = l2[1];
			l2[1] = l2[3];
			l2[3] = temp;
		}
	}
	vector<Vec4i> lLines;
	lLines.push_back(l1);
	Vec4i l = { l1[0] * 3 / 4 + l2[0] / 4, l1[1] * 3 / 4 + l2[1] / 4, l1[2] * 3 / 4 + l2[2] / 4, l1[3] * 3 / 4 + l2[3] / 4 };
	lLines.push_back(l);
	l = { l1[0] / 2 + l2[0] / 2, l1[1] / 2 + l2[1] / 2, l1[2] / 2 + l2[2] / 2, l1[3] / 2 + l2[3] / 2 };
	lLines.push_back(l);
	l = { l1[0] / 4 + l2[0] * 3 / 4, l1[1] / 4 + l2[1] * 3 / 4, l1[2] / 4 + l2[2] * 3 / 4, l1[3] / 4 + l2[3] * 3 / 4 };
	lLines.push_back(l);
	lLines.push_back(l2);
	lLines = extension(lLines, size);
	return lLines;
}
vector<Vec4i> sortLines(vector<Vec4i> lines, Vec4i lv, double angle)
{
	vector<Point> ps;
	vector<vector<int>> v;
	if (angle > 45 & angle < 135)
	{
		for (int i = 0; i < lines.size(); i++)
		{
			Point pp = intersection(lv, lines[i]);
			ps.push_back(pp);
			vector<int> temp = { pp.x,i };
			v.push_back(temp);
		}
	}
	else
	{
		for (int i = 0; i < lines.size(); i++)
		{
			Point pp = intersection(lv, lines[i]);
			ps.push_back(pp);
			vector<int> temp = { pp.y,i };
			v.push_back(temp);
		}
	}
	for (int i = 0; i < v.size() - 1; i++)
	{
		for (int j = i + 1; j < v.size(); j++)
		{
			if (v[i][0] > v[j][0])
			{
				vector<int> temp = v[i];
				v[i] = v[j];
				v[j] = temp;
			}
		}
	}
	vector<Vec4i> nlines(lines.size());
	for (int i = 0; i < lines.size(); i++)
		nlines[i] = lines[v[i][1]];
	return nlines;
}
vector<Vec4i> getPos(vector<Vec4i> lLines, vector<Vec4i> sLines)
{
	vector<Vec4i> pos;
	for (int i = 0; i < lLines.size(); i++)
	{
		for (int j = 0; j < sLines.size(); j++)
		{
			Vec4i l1 = lLines[i], l2 = sLines[j];
			Point p = intersection(l1, l2);
			int x11 = min(l1[0], l1[2]), x12 = max(l1[0], l1[2]), x21 = min(l2[0], l2[2]), x22 = max(l2[0], l2[2]);
			int y11 = min(l1[1], l1[3]), y12 = max(l1[1], l1[3]), y21 = min(l2[1], l2[3]), y22 = max(l2[1], l2[3]);
			if (p.x > x11 & p.x<x12 & p.x>x21 & p.x<x22 & p.y>y11 & p.y<y12 & p.y>y21 & p.y < y22)
			{
				Vec4i temp = { p.x,p.y,i,j };
				pos.push_back(temp);
			}
		}
	}
	return pos;
}
vector<Vec4i> trans(vector<Vec4i> pos, Size size)
{
	Vec2i ab = { size.width / 16,size.height / 4 };
	vector<Vec4i> map = creatMap(size, ab);
	Mat data = dataMat(pos, ab);
	Mat xy = xyMat(pos);
	Mat t = data.t();
	Mat h = (t*data).inv()*t*xy;
	Mat transMat(3, 3, CV_32F);
	transMat.at<float>(0, 0) = h.at<float>(0);
	transMat.at<float>(1, 0) = h.at<float>(1);
	transMat.at<float>(2, 0) = h.at<float>(2);
	transMat.at<float>(0, 1) = h.at<float>(3);
	transMat.at<float>(1, 1) = h.at<float>(4);
	transMat.at<float>(2, 1) = h.at<float>(5);
	transMat.at<float>(0, 2) = h.at<float>(6);
	transMat.at<float>(1, 2) = h.at<float>(7);
	transMat.at<float>(2, 2) = 1;

	//cout << transMat << endl;
	vector<Vec4i> result;
	result = transLines(map, transMat);
	//warpPerspective(map, result, transMat, size, 1);
	return result;
}
vector<Vec4i> creatMap(Size size, Vec2i ab)
{
	vector<Vec4i> map;
	for (int i = 0; i <= 4; i++)
		map.push_back(Vec4i{ 0, ab[1] * i, size.width - 1, ab[1] * i });
	for (int i = 0; i <= 16; i++)
		map.push_back(Vec4i{ ab[0] * i, 0, ab[0] * i, size.height - 1 });
	return map;
}
Mat dataMat(vector<Vec4i> pos, Vec2i ab)
{
	Mat data(pos.size()*2, 8, CV_32F);
	for (int i = 0; i < pos.size(); i++)
	{
		int x0 = (pos[i][3] + 2)* ab[0], y0 = pos[i][2] * ab[1], x = pos[i][0], y = pos[i][1];
		data.at<float>(i * 2, 0) = x0;
		data.at<float>(i * 2, 1) = y0;
		data.at<float>(i * 2, 2) = 1;
		data.at<float>(i * 2, 3) = 0;
		data.at<float>(i * 2, 4) = 0;
		data.at<float>(i * 2, 5) = 0;
		data.at<float>(i * 2, 6) = -x0*x;
		data.at<float>(i * 2, 7) = -y0*x;
		data.at<float>(i * 2 + 1, 0) = 0;
		data.at<float>(i * 2 + 1, 1) = 0;
		data.at<float>(i * 2 + 1, 2) = 0;
		data.at<float>(i * 2 + 1, 3) = x0;
		data.at<float>(i * 2 + 1, 4) = y0;
		data.at<float>(i * 2 + 1, 5) = 1;
		data.at<float>(i * 2 + 1, 6) = -x0*y;
		data.at<float>(i * 2 + 1, 7) = -y0*y;
	}
	return data;
}
Mat xyMat(vector<Vec4i> pos)
{
	Mat data(pos.size()*2, 1, CV_32F);
	for (int i = 0; i < pos.size(); i++)
	{
		data.at<float>(i * 2, 0) = pos[i][0];
		data.at<float>(i * 2 + 1, 0) = pos[i][1];
	}
	return data;
}
vector<Vec4i> transLines(vector<Vec4i> map, Mat transMat)
{
	vector<Vec4i> result;
	for (int i = 0; i < map.size(); i++)
	{
		Mat lines(1, 3, CV_32F);
		lines.at<float>(0, 0) = map[i][0];
		lines.at<float>(0, 1) = map[i][1];
		lines.at<float>(0, 2) = 1;
		Mat r = lines*transMat;
		Vec4i rl = { (int)(r.at<float>(0,0) / r.at<float>(0,2)),(int)(r.at<float>(0,1) / r.at<float>(0,2)) ,0,0 };
		lines.at<float>(0, 0) = map[i][2];
		lines.at<float>(0, 1) = map[i][3];
		lines.at<float>(0, 2) = 1;
		r = lines*transMat;
		rl[2] = (int)(r.at<float>(0, 0) / r.at<float>(0, 2));
		rl[3] = (int)(r.at<float>(0, 1) / r.at<float>(0, 2));
		result.push_back(rl);
	}
	return result;
}
bool ifErr(vector<Vec4i> lines)
{
	double al1 = getAngle(lines[0]), al2 = getAngle(lines[4]);
	double as1 = getAngle(lines[5]), as2 = getAngle(lines[10]);
	if (al1 < 30 & al2 > 150) al1 += 180;
	if (al1 > 150 & al2 < 30) al2 += 180;
	if (as1 < 30 & as2 > 150) as1 += 180;
	if (as1 > 150 & as2 < 30) as2 += 180;
	double al = (al1 + al2) / 2, as = (as1 + as2) / 2;
	if (abs(al1 - al2) > 25) return true;
	else if (abs(as1 - as2) > 30) return true;
	else if (abs(al - as) < 60) return true;
	else if (dstns(lines[5], lines[6], lines[2]) < 30) return true;
	else if (dstns(lines[5], lines[6], lines[2]) > 80) return true;
	else return false;
}

double dstns(Vec4i l1, Vec4i l2, Vec4i lv)
{
	Point p1 = intersection(l1, lv);
	Point p2 = intersection(l2, lv);
	return sqrt((p1.x - p2.x)*(p1.x - p2.x) + (p1.y - p2.y)*(p1.y - p2.y));
}







/****************/
vector<vector<Vec4i>> fitLines(vector<Vec4i> lines, vector<Vec4i> hlines, int thresh)
{
	vector<vector<Vec4i>> temp(lines.size());
	for (int i = 0; i < hlines.size(); i++)
	{
		double a1 = getAngle(hlines[i]);
		Vec4i l1 = hlines[i];
		for (int j = 0; j < lines.size(); j++)
		{
			double a2 = getAngle(lines[j]);
			Vec4i l2 = lines[j];
			vector<Point> points;
			points.push_back(Point(l1[0], l1[1]));
			points.push_back(Point(l1[2], l1[3]));
			points.push_back(Point(l2[0], l2[1]));
			points.push_back(Point(l2[2], l2[3]));
			RotatedRect rect = minAreaRect(points);
			if ((rect.size.width < thresh | rect.size.height < thresh) & abs(a1 - a2) < 10)
			{
				temp[j].push_back(l1);
				break;
			}
		}
	}
	return temp;
}
vector<Vec4i> getPos(vector<vector<Vec4i>> lLines, vector<vector<Vec4i>> sLines)
{
	//x,y,行,列
	vector<Vec4i> pos;
	for (int i = 0; i < lLines.size(); i++)
	{
		for (int j = 0; j < lLines[i].size(); j++)
		{
			for (int k = 0; k < sLines.size(); k++)
			{
				for (int h = 0; h < sLines[k].size(); h++)
				{
					Vec4i l1 = lLines[i][j], l2 = sLines[k][h];
					Point p = intersection(l1, l2);
					int x11 = min(l1[0], l1[2]), x12 = max(l1[0], l1[2]), x21 = min(l2[0], l2[2]), x22 = max(l2[0], l2[2]);
					int y11 = min(l1[1], l1[3]), y12 = max(l1[1], l1[3]), y21 = min(l2[1], l2[3]), y22 = max(l2[1], l2[3]);
					if (p.x>x11 & p.x<x12 & p.x>x21 & p.x<x22 & p.y>y11 & p.y<y12 & p.y>y21 & p.y<y22)
					{
						Vec4i temp = { p.x,p.y,i,k };
						pos.push_back(temp);
						//break;
					}
				}
			}
		}
	}
	return pos;
}
void postest(vector<Vec4i> pos, Mat bg)
{
	//检测pos中的坐标是否准确
	for (int i = 0; i < pos.size(); i++)
	{
		cout << pos[i][2];
		cout << pos[i][3] << endl;
		waitKey(0);
	}
}
Mat tMat(vector<Vec4i> pos, Vec2i ab)
{
	//用OpenCV自带仿射变换方法进行准确性对比
	vector<Point2f> po, pa;
	for (int i = 0; i < 3; i++)
	{
		po.push_back(Point2f((pos[i][3] + 2)* ab[0], pos[i][2] * ab[1]));
	}
	for (int i = 0; i < 3; i++)
	{
		pa.push_back(Point2f(pos[i][0], pos[i][1]));
	}
	Mat data = getAffineTransform(po, pa);
	cout << data << endl;
	return data;
}
Mat draw(Mat map, vector<Vec4i> pos, Vec2i ab)
{
	for (int i = 0; i < pos.size(); i++)
	{
		circle(map, Point((pos[i][3] + 2)* ab[0], pos[i][2] * ab[1]), 4, Scalar(255), 4);
	}
	return map;
}
Mat draw(Mat map, vector<Vec4i> pos)
{
	for (int i = 0; i < pos.size(); i++)
	{
		circle(map, Point(pos[i][0], pos[i][1] ), 2, Scalar(150,255,255), 2);
	}
	return map;
}
/***************/