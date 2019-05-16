

// include std
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <queue>
#include <thread>
#include <memory>

// opencv
#include <opencv2/opencv.hpp>

#include "SKEncoder.h"
#include "version.h"

#define MAJOR_VER 1
#define MINOR_VER 0

std::vector<std::string> CollectFiles(std::string dir, std::vector<std::string> allowedExtensions)
{
	std::vector<std::string> ret;
	//std::vector<std::string> allowedExtensions = { ".jpg", ".png" ,".jpeg" };
	for (int i = 0; i < allowedExtensions.size(); i++) {
		std::vector<cv::String> imageNamesCurrentExtension;
		cv::glob(
			dir + "/*" + allowedExtensions[i],
			imageNamesCurrentExtension,
			true
		);
		ret.insert(
			ret.end(),
			imageNamesCurrentExtension.begin(),
			imageNamesCurrentExtension.end()
		);
	}
	return ret;
}


int main(int argc, char* argv[]) 
{
	int stat_p = 10;
	SKCommon::infoOutput("Version = %d.%d.%s", MAJOR_VER, MINOR_VER, __GIT_VERSION__);
	if (argc < 2)
	{
		SKCommon::infoOutput("Usage : SKEncoder [Folder/VidFile] ([DstName])");
		return -1;
	}
	std::string input = argv[1];
	std::string output = "output.h265";
	if (argc > 2)
	{
		output = argv[2];
	}
	std::vector<cv::Mat> data;
	if (SKCommon::getFileExtention(input) == "") //it's a folder
	{
		output = input + ".h265";
		auto files = CollectFiles(input, std::vector<std::string>{".jpg", ".png", ".tiff"});
		if (files.size() == 0)
		{
			SKCommon::errorOutput(SKCOMMON_DEBUG_STRING + "No file found in folder %s", input.c_str());
			return -1;
		}
		data.resize(files.size());
		for (int i = 0; i < data.size(); i++)
		{
			data[i] = cv::imread(files[i]);
			if (i % (data.size() / stat_p) == 0)
			{
				SKCommon::infoOutput("Read image %d%% percent", i / (data.size() / stat_p) * stat_p);
			}
		}
	}
	else //it's a video file
	{
		output = input.substr(0, input.find_last_of('.')) + ".h265";
		cv::VideoCapture vc;
		vc.open(input);
		if (vc.isOpened() == false)
		{
			SKCommon::errorOutput(SKCOMMON_DEBUG_STRING + "Video file %s not supported or not found", input.c_str());
			return -1;
		}
		data.resize(vc.get(cv::CAP_PROP_FRAME_COUNT));
		for (int i = 0; i < data.size(); i++)
		{
			vc >> data[i];
			if (i % (data.size() / stat_p) == 0)
			{
				SKCommon::infoOutput("Read video %d%% percent", i / (data.size() / stat_p) * stat_p);
			}
		}
		vc.release();
	}
	SKEncoder encoder;
	encoder.init(data.size(), data[0].size(), output, SKEncoder::FrameType::ARGB);
	for (int i = 0; i < data.size(); i++)
	{
		cv::Mat t0;
		cv::cvtColor(data[i], t0, cv::COLOR_BGR2BGRA);
		cv::cuda::GpuMat tg(t0);
		encoder.encode(tg.data, tg.step);
		//if (i % (data.size() / stat_p) == 0)
		//{
		//	SKCommon::infoOutput("Encode video %d%% percent", i / (data.size() / stat_p) * stat_p);
		//}
	}
	encoder.endEncode();
}


/*
#define FRAME_NUM 100
//#define TEST_YUV
#define TEST_ABGR
//#define TEST_ARGB
const int w = 4112, h = 3008;
#ifdef TEST_YUV
encoder.init(FRAME_NUM, cv::Size(w, h));
cv::cuda::GpuMat gm[FRAME_NUM][3];
for (int i = 0; i < FRAME_NUM; i++)
{
cv::Mat im = cv::imread(SysUtil::format("data/CUCAU1731016_00_%05d.jpg", i));
cv::Mat YUV;
cv::cvtColor(im, YUV, cv::COLOR_BGR2YUV_I420);
//cv::cvtColor(im, YUV, cv::COLOR_BGR2YUV);
cv::Mat Y = YUV(cv::Rect(0, 0, im.cols, im.rows));
cv::Mat U = YUV(cv::Rect(0, im.rows, im.cols, im.rows / 4));
U = U.reshape(1, im.rows / 2);
cv::Mat V = YUV(cv::Rect(0, im.rows + im.rows / 4, im.cols, im.rows / 4));
V = V.reshape(1, im.rows / 2);
gm[i][0].upload(Y);
gm[i][1].upload(U);
gm[i][2].upload(V);
if(i % 20 == 0)
SysUtil::infoOutput(SysUtil::format("Buffering Data Frame = %d, All = %d", i, FRAME_NUM));
}
SysUtil::infoOutput("Data All Buffered, Start Encoding.");
for (int i = 0; i < FRAME_NUM; i++)
{
std::vector<void*> YUV_pointer(3);
std::vector<uint32_t> steps(3);
for (int j = 0; j < 3; j++)
{
YUV_pointer[j] = gm[i][j].data;
steps[j] = gm[i][j].step;
}
encoder.encode(YUV_pointer, steps);
}
#elif defined(TEST_ABGR)
encoder.init(FRAME_NUM, cv::Size(w, h),"outputABGR.h265", SKEncoder::FrameType::ABGR);
cv::cuda::GpuMat gm[FRAME_NUM];
for (int i = 0; i < FRAME_NUM; i++)
{
cv::Mat im = cv::imread(SKCommon::format("data/CUCAU1731016_00_%05d.jpg", i));
cv::Mat ABGR;
cv::cvtColor(im, ABGR, cv::COLOR_BGR2RGBA);
gm[i].upload(ABGR);
if (i % 20 == 0)
SKCommon::infoOutput("Buffering Data Frame = %d, All = %d", i, FRAME_NUM);
}
SKCommon::infoOutput("Data All Buffered, Start Encoding.");
for (int i = 0; i < FRAME_NUM; i++)
{
encoder.encode(gm[i].data, gm[i].step);
}
#endif
encoder.endEncode();
getchar();
return 0;
*/