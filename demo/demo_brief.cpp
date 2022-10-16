/**
 * File: demo_brief.cpp
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DLoopDetector
 * License: see the LICENSE.txt file
 */

#include <iostream>
#include <vector>
#include <string>

// DLoopDetector and DBoW2
#include <DBoW2/DBoW2.h> // defines BriefVocabulary
#include "DLoopDetector.h" // defines BriefLoopDetector
#include <DVision/DVision.h> // Brief

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

#include "demoDetector.h"

using namespace DLoopDetector;
using namespace DBoW2;
using namespace DVision;
using namespace std;

// ----------------------------------------------------------------------------

static const char *VOC_FILE = "./resources/brief_k10L6.voc.gz";//词典
static const char *IMAGE_DIR = "./resources/images";
static const char *POSE_FILE = "./resources/pose.txt"; //轨迹的坐标信息
static const int IMAGE_W = 640; // image size
static const int IMAGE_H = 480;
static const char *BRIEF_PATTERN_FILE = "./resources/brief_pattern.yml";

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
//声明一个Brief特征提取器类
/// This functor extracts BRIEF descriptors in the required format
class BriefExtractor: public FeatureExtractor<FBrief::TDescriptor>
{
public:
  /** 
   * Extracts features from an image
   * @param im image
   * @param keys keypoints extracted
   * @param descriptors descriptors extracted
   */
  virtual void operator()(const cv::Mat &im, 
    vector<cv::KeyPoint> &keys, vector<BRIEF256::bitset> &descriptors) const override;

  /**
   * Creates the brief extractor with the given pattern file 使用给定的模式文件创建brief提取器
   * @param pattern_file
   */
  BriefExtractor(const std::string &pattern_file);

private:

  /// BRIEF descriptor extractor
  BRIEF256 m_brief;
};

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

int main()
{
  // prepares the demo  创建一个范例检测器demo。　BriefVocabulary：词典类。　BriefLoopDetector：回环检测器类。 FBrief::TDescriptor：Brief描述子
  demoDetector<BriefVocabulary, BriefLoopDetector, FBrief::TDescriptor> 
    demo(VOC_FILE, IMAGE_DIR, POSE_FILE, IMAGE_W, IMAGE_H);
  
  try 
  {
    // run the demo with the given functor to extract features
    BriefExtractor extractor(BRIEF_PATTERN_FILE); //创建一个Brief特征提取器对象，使用brief_pattern.yml文件，此文件是计算brief描述子时所用的２５６对点对坐标
    demo.run("BRIEF", extractor);  //开始检测回环
  }
  catch(const std::string &ex)  //捕获运行时的异常并打印
  {
    cout << "Error: " << ex << endl;
  }

  return 0;
}

// ----------------------------------------------------------------------------
//载入brief_pattern.yml文件，此文件是计算brief描述子时所用的２５６对点对坐标
BriefExtractor::BriefExtractor(const std::string &pattern_file)
{
  // The DVision::BRIEF extractor computes a random pattern by default when
  // the object is created.
  // We load the pattern that we used to build the vocabulary, to make
  // the descriptors compatible with the predefined vocabulary 我们加载用于构建词典的模式，以使描述子与预定义词典兼容
  
  // loads the pattern
  cv::FileStorage fs(pattern_file.c_str(), cv::FileStorage::READ);
  if(!fs.isOpened()) throw string("Could not open file ") + pattern_file;
  
  vector<int> x1, y1, x2, y2;
  fs["x1"] >> x1;
  fs["x2"] >> x2;
  fs["y1"] >> y1;
  fs["y2"] >> y2;
  
  m_brief.importPairs(x1, y1, x2, y2);//设置提取特征的模式，即使用此２５６对点对坐标。
}

// ----------------------------------------------------------------------------
//运算符重载，计算当前图片的关键点、描述子
void BriefExtractor::operator() (const cv::Mat &im, 
  vector<cv::KeyPoint> &keys, vector<FBrief::TDescriptor> &descriptors) const
{
  // extract FAST keypoints with opencv 用opencv检测FAST关键点
  const int fast_th = 20; // corner detector response threshold 角点探测器响应阈值
  cv::FAST(im, keys, fast_th, true);  //true:对拐点施加非极大值抑制
  
  // compute their BRIEF descriptor
  m_brief.compute(im, keys, descriptors); //计算当前图片的关键点、描述子    (vector<BRIEF::bitset>&)强制类型转换
}

// ----------------------------------------------------------------------------

