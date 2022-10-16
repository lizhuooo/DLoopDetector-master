/**
 * File: demoDetector.h
 * Date: November 2011
 * Author: Dorian Galvez-Lopez
 * Description: demo application of DLoopDetector
 * License: see the LICENSE.txt file
 */

#ifndef __DEMO_DETECTOR__
#define __DEMO_DETECTOR__

#include <iostream>
#include <vector>
#include <string>

// OpenCV
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>

// DLoopDetector and DBoW2
#include <DBoW2/DBoW2.h>
#include "DLoopDetector.h"
#include <DUtils/DUtils.h>
#include <DUtilsCV/DUtilsCV.h>
#include <DVision/DVision.h>

using namespace DLoopDetector;
using namespace DBoW2;
using namespace std;

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

/// Generic class to create functors to extract features
template<class TDescriptor>
class FeatureExtractor
{
public:
  /**
   * Extracts features
   * @param im image
   * @param keys keypoints extracted
   * @param descriptors descriptors extracted
   */
  virtual void operator()(const cv::Mat &im, 
    vector<cv::KeyPoint> &keys, vector<TDescriptor> &descriptors) const = 0;
};

// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 

/// @param TVocabulary vocabulary class (e.g: BriefVocabulary)
/// @param TDetector detector class (e.g: BriefLoopDetector)
/// @param TDescriptor descriptor class (e.g: bitset for Brief)
template<class TVocabulary, class TDetector, class TDescriptor>
/// Class to run the demo 
class demoDetector
{
public:

  /**
   * @param vocfile vocabulary file to load
   * @param imagedir directory to read images from
   * @param posefile pose file
   * @param width image width
   * @param height image height
   */
  demoDetector(const std::string &vocfile, const std::string &imagedir,
    const std::string &posefile, int width, int height);
    
  ~demoDetector(){}

  /**
   * Runs the demo
   * @param name demo name
   * @param extractor functor to extract features
   */
  void run(const std::string &name, 
    const FeatureExtractor<TDescriptor> &extractor);

protected:

  /**
   * Reads the robot poses from a file
   * @param filename file
   * @param xs
   * @param ys
   */
  void readPoseFile(const char *filename, std::vector<double> &xs, 
    std::vector<double> &ys) const;

protected:

  std::string m_vocfile;
  std::string m_imagedir;
  std::string m_posefile;
  int m_width;
  int m_height;
};

// ---------------------------------------------------------------------------
//创建一个范例检测器，vocfile:词典文件，imagedir：图片库，posefile：用于特征提取的文件，width：，height：
template<class TVocabulary, class TDetector, class TDescriptor>
demoDetector<TVocabulary, TDetector, TDescriptor>::demoDetector
  (const std::string &vocfile, const std::string &imagedir,
  const std::string &posefile, int width, int height)
  : m_vocfile(vocfile), m_imagedir(imagedir), m_posefile(posefile),
    m_width(width), m_height(height)
{
}

// ---------------------------------------------------------------------------
//开始对图片库的每张图片进行回环检测
template<class TVocabulary, class TDetector, class TDescriptor>
void demoDetector<TVocabulary, TDetector, TDescriptor>::run
  (const std::string &name, const FeatureExtractor<TDescriptor> &extractor)
{
  cout << "DLoopDetector Demo" << endl 
    << "Dorian Galvez-Lopez" << endl
    << "http://doriangalvez.com" << endl << endl;
  
  // Set loop detector parameters
  typename TDetector::Parameters params(m_height, m_width);  //params:参数变量
  
  // Parameters given by default are:
  // use nss = true
  // alpha = 0.3
  // k = 3
  // geom checking = GEOM_DI
  // di levels = 0
  
  // We are going to change these values individually:
  params.use_nss = true; // 使用标准化相似性评分代替原始评分use normalized similarity score instead of raw score
  params.alpha = 0.3; // 标准化相似性评分的阈值nss threshold
  params.k = 1; // 回环必须与之前的1个匹配项一致,即时间一致性检验k＝１， a loop must be consistent with 1 previous matches
  params.geom_check = GEOM_DI; // 使用直接索引进行几何一致性检验use direct index for geometrical checking
  params.di_levels = 2; // 直接索引使用层数为：２，use two direct index levels
  
  // To verify loops you can select one of the next geometrical checkings:
  // GEOM_EXHAUSTIVE: correspondence points are computed by comparing all
  //    the features between the two images.
  // GEOM_FLANN: as above, but the comparisons are done with a Flann structure,
  //    which makes them faster. However, creating the flann structure may
  //    be slow.
  // GEOM_DI: the direct index is used to select correspondence points between
  //    those features whose vocabulary node at a certain level is the same.
  //    The level at which the comparison is done is set by the parameter
  //    di_levels:
  //      di_levels = 0 -> features must belong to the same leaf (word).
  //         This is the fastest configuration and the most restrictive one.
  //      di_levels = l (l < L) -> node at level l starting from the leaves.
  //         The higher l, the slower the geometrical checking, but higher
  //         recall as well.
  //         Here, L stands for the depth levels of the vocabulary tree.
  //      di_levels = L -> the same as the exhaustive technique.
  // GEOM_NONE: no geometrical checking is done.
  //
  // In general, with a 10^6 vocabulary, GEOM_DI with 2 <= di_levels <= 4 
  // yields the best results in recall/time.
  // Check the T-RO paper for more information.
  //
  
  // Load the vocabulary to use
  cout << "Loading " << name << " vocabulary..." << endl;
  TVocabulary voc(m_vocfile);  //载入已经创建好的一个brief词典  m_vocfile:词典地址
  
  // Initiate loop detector with the vocabulary 
  cout << "Processing sequence..." << endl;
  TDetector detector(voc, params);  //创建具有给定参数和给定词典的回环检测器detector(在TemplatedLoopDetector.h中)
  
  // Process images
  vector<cv::KeyPoint> keys;
  vector<TDescriptor> descriptors;

  // load image filenames  
  vector<string> filenames = 
    DUtils::FileFunctions::Dir(m_imagedir.c_str(), ".png", true); //载入图片集信息
  
  // load robot poses
  vector<double> xs, ys;
  readPoseFile(m_posefile.c_str(), xs, ys); //读取pose.txt文件,将数据压入xs、ys中
  
  // we can allocate memory for the expected number of images  
  detector.allocate(filenames.size());  //为预期数量的图像分配内存
  
  // prepare visualization windows 准备可视化窗口
  DUtilsCV::GUI::tWinHandler win = "Current image";
  DUtilsCV::GUI::tWinHandler winplot = "Trajectory";
  
  DUtilsCV::Drawing::Plot::Style normal_style(2); // thickness 画图形式：正常风格（颜色为黑色）:轨迹粗细为２
  DUtilsCV::Drawing::Plot::Style loop_style('r', 2); // color, thickness　画图形式：回环的风格:颜色为红色，轨迹粗细为２
  
  DUtilsCV::Drawing::Plot implot(240, 320,
    - *std::max_element(xs.begin(), xs.end()),
    - *std::min_element(xs.begin(), xs.end()),
    *std::min_element(ys.begin(), ys.end()),
    *std::max_element(ys.begin(), ys.end()), 20);  //设置绘图参数：图片尺寸，x、y最大、小值。 20:靠近图像边框的空白区域的边距宽度
  
  // prepare profiler to measure times 
  DUtils::Profiler profiler;  //准备探查器以测量时间
  
  int count = 0;  //count:检测到回环的次数
  
  // go
  for(unsigned int i = 0; i < filenames.size(); ++i)  //遍历图片集，对每张图片进行回环检测
  {
    cout << "Adding image " << i << ": " << filenames[i] << "... " << endl;
    
    // get image
    cv::Mat im = cv::imread(filenames[i].c_str(), 0); // grey scale 读取图片集第i张图片
    
    // show image
    DUtilsCV::GUI::showImage(im, true, &win, 10); //显示当前查询的图片
    
    // get features
    profiler.profile("features");  //计算特征检测的时间。计时开始     map<features, 时间vector<double> >，features对应的时间容器内将储存每张图片所用的时间
    extractor(im, keys, descriptors);  //计算当前图片的关键点、描述子
    profiler.stop();  //计时结束
        
    // add image to the collection and check if there is some loop
    DetectionResult result;  //result:回环检测的结果
    
    profiler.profile("detection"); //计算当前图片回环检测的时间。计时开始  map<detection, 时间vector<double> >，detection对应的时间容器内将储存每张图片所用的时间
    detector.detectLoop(keys, descriptors, result); //检测当前图片是否发生回环,输入当前查询图片的关键点、描述子,返回回环检测结果
    profiler.stop(); //计时结束
    
    if(result.detection()) //若detection＝ture 则检测到回环
    {
      cout << "- Loop found with image " << result.match << "!"
        << endl;
      ++count;  //检测到回环的次数
    }
    else  //若detection＝false 则未检测到回环
    {
      cout << "- No loop: ";
      switch(result.status)  //显示未检测到回环的各种原因
      {
        case CLOSE_MATCHES_ONLY:
          cout << "All the images in the database are very recent" << endl;
          break;
          
        case NO_DB_RESULTS:
          cout << "There are no matches against the database (few features in"
            " the image?)" << endl;
          break;
          
        case LOW_NSS_FACTOR:
          cout << "Little overlap between this image and the previous one"
            << endl;
          break;
            
        case LOW_SCORES:
          cout << "No match reaches the score threshold (alpha: " <<
            params.alpha << ")" << endl;
          break;
          
        case NO_GROUPS:
          cout << "Not enough close matches to create groups. "
            << "Best candidate: " << result.match << endl;
          break;
          
        case NO_TEMPORAL_CONSISTENCY:
          cout << "No temporal consistency (k: " << params.k << "). "
            << "Best candidate: " << result.match << endl;
          break;
          
        case NO_GEOMETRICAL_CONSISTENCY:
          cout << "No geometrical consistency. Best candidate: " 
            << result.match << endl;
          break;
          
        default:
          break;
      }
    }
    
    cout << endl;
    
    // show trajectory
    if(i > 0) //画轨迹图
    {
      if(result.detection()) //检测到回环的轨迹用loop_style标注
        implot.line(-xs[i-1], ys[i-1], -xs[i], ys[i], loop_style);
      else   //未检测到回环的轨迹用normal_style标注
        implot.line(-xs[i-1], ys[i-1], -xs[i], ys[i], normal_style);
      
      DUtilsCV::GUI::showImage(implot.getImage(), true, &winplot, 10); //函数的功能是显示绘制的轨迹图，其中参数10单位是毫秒，表示图像刷新频率
    }
  }
  
  if(count == 0) //检测到的回环次数为０，则显示此图像序列中没有发现回环
  {
    cout << "No loops found in this image sequence" << endl;
  }
  else
  {
    cout << count << " loops found in this image sequence!" << endl;
  } 

  cout << endl << "Execution time:" << endl
    << " - Feature computation: " << profiler.getMeanTime("features") * 1e3  //计算每幅图片平均计算特征的时间
    << " ms/image" << endl
    << " - Loop detection: " << profiler.getMeanTime("detection") * 1e3    //计算每幅图片平均进行回环检测的时间
    << " ms/image" << endl;

  cout << endl << "Press a key to finish..." << endl;
  DUtilsCV::GUI::showImage(implot.getImage(), true, &winplot, 0);
}

// ---------------------------------------------------------------------------
//读取pose.txt中的文件
template<class TVocabulary, class TDetector, class TDescriptor>
void demoDetector<TVocabulary, TDetector, TDescriptor>::readPoseFile
  (const char *filename, std::vector<double> &xs, std::vector<double> &ys)
  const
{
  xs.clear();
  ys.clear();
  
  fstream f(filename, ios::in); //文件以输入方式打开（文件数据输入到内存）
  
  string s;
  double ts, x, y, t;
  while(!f.eof()) //f.eof刚开始返回的值是0；直到文件尾才返回真
  {
    getline(f, s);  //getline(cin,str); cin:输入流 str:string对象  完成读入一行数据
    if(!f.eof() && !s.empty())
    {
      sscanf(s.c_str(), "%lf, %lf, %lf, %lf", &ts, &x, &y, &t);
      xs.push_back(x);  //将文件中第二列字符压给xs
      ys.push_back(y);  //将文件中第三列字符压给xs
    }
  }
  
  f.close();
}

// ---------------------------------------------------------------------------

#endif

