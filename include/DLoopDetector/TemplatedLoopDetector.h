/**
 * File: TemplatedLoopDetector
 * Date: March 2011
 * Author: Dorian Galvez-Lopez
 * Description: templated loop detector
 * License: see the LICENSE.txt file
 *
 */

#ifndef __D_T_TEMPLATED_LOOP_DETECTOR__
#define __D_T_TEMPLATED_LOOP_DETECTOR__

#include <vector>
#include <numeric>
#include <fstream>
#include <string>

#include <opencv2/core.hpp>
#include <opencv2/features2d.hpp>

#include "TemplatedVocabulary.h"
#include "TemplatedDatabase.h"
#include "QueryResults.h"
#include "BowVector.h"

#include "DUtils.h"
#include "DUtilsCV.h"
#include "DVision.h"
using namespace std;
using namespace DUtils;
using namespace DBoW2;
#include <time.h>
namespace DLoopDetector {


/// Geometrical checking methods
enum GeometricalCheck
{
  /// Exhaustive search
  GEOM_EXHAUSTIVE,
  /// Use direct index
  GEOM_DI,
  /// Use a Flann structure
  GEOM_FLANN,
  /// Do not perform geometrical checking
  GEOM_NONE
};

/// Reasons for dismissing loops
enum DetectionStatus
{
  /// Loop correctly detected
  LOOP_DETECTED,
  /// All the matches are very recent
  CLOSE_MATCHES_ONLY,
  /// No matches against the database
  NO_DB_RESULTS,
  /// Score of current image against previous one too low
  LOW_NSS_FACTOR,
  /// Scores (or NS Scores) were below the alpha threshold
  LOW_SCORES,
  /// Not enough matches to create groups  
  NO_GROUPS,
  /// Not enough temporary consistent matches (k)
  NO_TEMPORAL_CONSISTENCY,
  /// The geometrical consistency failed
  NO_GEOMETRICAL_CONSISTENCY
};

/// Result of a detection
struct DetectionResult
{
  /// Detection status. LOOP_DETECTED iff loop detected
  DetectionStatus status;
  /// Query id
  EntryId query;
  /// Matched id if loop detected, otherwise, best candidate 
  EntryId match;
  
  /**
   * Checks if the loop was detected
   * @return true iff a loop was detected
   */
  inline bool detection() const
  {
    return status == LOOP_DETECTED;
  }
};

/// TDescriptor: class of descriptor
/// F: class of descriptor functions
template<class TDescriptor, class F>
/// Generic Loop detector
class TemplatedLoopDetector
{
public:
   
  /// Parameters to create a loop detector
  struct Parameters
  {
    /// Height of expected images
    int image_rows;
    /// Width of expected images
    int image_cols;
    
    // Main loop detector parameters
    
    /// Use normalized similarity score?
    bool use_nss;
    /// Alpha threshold
    float alpha;
    /// Min consistent matches to pass the temporal check
    int k;
    /// Geometrical check
    GeometricalCheck geom_check;
    /// If using direct index for geometrical checking, direct index levels
    int di_levels;
    
    // These are less deciding parameters of the system
    
    /// Distance between entries to be consider a match 查询图片与考虑匹配图片之间的距离
    int dislocal;
    /// Max number of results from db queries to consider
    int max_db_results;
    /// Min raw score between current entry and previous one to consider a match 
    float min_nss_factor;
    /// Min number of close matches to consider some of them
    int min_matches_per_group; 
    /// Max separation between matches to consider them of the same group
    int max_intragroup_gap; 
    /// Max separation between groups of matches to consider them consistent
    int max_distance_between_groups;
    /// Max separation between two queries to consider them consistent
    int max_distance_between_queries; 
  
    // These are for the RANSAC to compute the F
    
    /// Min number of inliers when computing a fundamental matrix
    int min_Fpoints;
    /// Max number of iterations of RANSAC
    int max_ransac_iterations;
    /// Success probability of RANSAC
    double ransac_probability;
    /// Max reprojection error of fundamental matrices
    double max_reprojection_error;
    
    // This is to compute correspondences
    
    /// Max value of the neighbour-ratio of accepted correspondences
    double max_neighbor_ratio;
  
    /**
     * Creates parameters by default
     */ 
    Parameters();
    
    /**
     * Creates parameters by default
     * @param height image height
     * @param width image width
     * @param frequency set the value of some parameters according to the 
     *   expected working frequency (Hz) 
     * @param nss use normalized similarity score
     * @param _alpha alpha parameter
     * @param _k k parameter (number of temporary consistent matches)
     * @param geom type of geometrical check
     * @param dilevels direct index levels when geom == GEOM_DI
     */ 
    Parameters(int height, int width, float frequency = 1, bool nss = true,
      float _alpha = 0.3, int _k = 3, 
      GeometricalCheck geom = GEOM_DI, int dilevels = 0);
      
  private:
    /**
     * Sets some parameters according to the frequency
     * @param frequency
     */
    void set(float frequency);
  };
  
public:

  /**
   * Empty constructor
   */
  TemplatedLoopDetector(const Parameters &params = Parameters());

  /**
   * Creates a loop detector with the given parameters and with a BoW2 database 
   * with the given vocabulary
   * @param voc vocabulary
   * @param params loop detector parameters
   */
  TemplatedLoopDetector(const TemplatedVocabulary<TDescriptor, F> &voc,
    const Parameters &params = Parameters());
  
  /**
   * Creates a loop detector with a copy of the given database, but clearing
   * its contents
   * @param db database to copy
   * @param params loop detector parameters
   */
  TemplatedLoopDetector(const TemplatedDatabase<TDescriptor, F> &db,
    const Parameters &params = Parameters());

  /**
   * Creates a loop detector with a copy of the given database, but clearing
   * its contents
   * @param T class derived from TemplatedDatabase
   * @param db database to copy
   * @param params loop detector parameters
   */
  template<class T>
  TemplatedLoopDetector(const T &db, const Parameters &params = Parameters());

  /**
   * Destructor
   */
  virtual ~TemplatedLoopDetector(void);
  
  /**
   * Retrieves a reference to the database used by the loop detector
   * @return const reference to database
   */
  inline const TemplatedDatabase<TDescriptor, F>& getDatabase() const;
  
  /**
   * Retrieves a reference to the vocabulary used by the loop detector
   * @return const reference to vocabulary
   */
  inline const TemplatedVocabulary<TDescriptor, F>& getVocabulary() const;
  
  /**
   * Sets the database to use. The contents of the database and the detector
   * entries are cleared
   * @param T class derived from TemplatedDatabase
   * @param db database to copy
   */
  template<class T>
  void setDatabase(const T &db);
  
  /**
   * Sets a new DBoW2 database created from the given vocabulary
   * @param voc vocabulary to copy
   */
  void setVocabulary(const TemplatedVocabulary<TDescriptor, F>& voc);
  
  /**
   * Allocates some memory for the first entries
   * @param nentries number of expected entries
   * @param nkeys number of keypoints per image expected
   */
  void allocate(int nentries, int nkeys = 0);

  /**
   * Adds the given tuple <keys, descriptors, current_t> to the database
   * and returns the match if any
   * @param keys keypoints of the image
   * @param descriptors descriptors associated to the given keypoints
   * @param match (out) match or failing information
   * @return true iff there was match
   */
  bool detectLoop(const std::vector<cv::KeyPoint> &keys, 
    const std::vector<TDescriptor> &descriptors,
    DetectionResult &match);

  /**
   * Resets the detector and clears the database, such that the next entry
   * will be 0 again
   */
  inline void clear();

protected:
  
  /// Matching island
  struct tIsland
  {
    /// Island starting entry
    EntryId first;
    /// Island ending entry
    EntryId last;
    /// Island score
    double score; // score of island
    
    /// Entry of the island with the highest score
    EntryId best_entry; // id and score of the entry with the highest score
    /// Highest single score in the island
    double best_score;  // in the island
    
    /**
     * Creates an empty island
     */
    tIsland(){}
    
    /**
     * Creates an island
     * @param f first entry
     * @param l last entry
     */
    tIsland(EntryId f, EntryId l): first(f), last(l){}
    
    /**
     * Creates an island
     * @param f first entry
     * @param l last entry
     * @param s island score
     */
    tIsland(EntryId f, EntryId l, double s): first(f), last(l), score(s){}
    
    /**
     * Says whether this score is less than the score of another island
     * @param b
     * @return true iff this score < b.score
     */
    inline bool operator < (const tIsland &b) const
    {
      return this->score < b.score;
    }
    
    /**
     * Says whether this score is greater than the score of another island
     * @param b
     * @return true iff this score > b.score
     */
    inline bool operator > (const tIsland &b) const
    {
      return this->score > b.score;
    }
    
    /** 
     * Returns true iff a > b
     * This function is used to sort in descending order
     * @param a
     * @param b
     * @return a > b
     */
    static inline bool gt(const tIsland &a, const tIsland &b)
    {
      return a.score > b.score;
    }
        
    /**
     * Returns true iff entry ids of a are less then those of b.
     * Assumes there is no overlap between the islands
     * @param a
     * @param b
     * @return a.first < b.first
     */
    static inline bool ltId(const tIsland &a, const tIsland &b)
    {
      return a.first < b.first;
    }
    
    /**
     * Returns the length of the island
     * @return length of island
     */
    inline int length() const { return last - first + 1; }
    
    /**
     * Returns a printable version of the island
     * @return printable island
     */
    std::string toString() const
    {
      stringstream ss;
      ss << "[" << first << "-" << last << ": " << score << " | best: <"
        << best_entry << ": " << best_score << "> ] ";
      return ss.str();
    }
  };
  
  /// Temporal consistency window
  struct tTemporalWindow
  {
    /// Island matched in the last query
    tIsland last_matched_island;
    /// Last query id
    EntryId last_query_id;
    /// Number of consistent entries in the window
    int nentries;
    
    /**
     * Creates an empty temporal window
     */
    tTemporalWindow(): nentries(0) {}
  };
  
  
protected:
  
  /**
   * Removes from q those results whose score is lower than threshold
   * (that should be alpha * ns_factor)
   * @param q results from query
   * @param threshold min value of the resting results
   */
  void removeLowScores(QueryResults &q, double threshold) const;
  
  /**
   * Returns the islands of the given matches in ascending order of entry ids
   * @param q 
   * @param islands (out) computed islands
   */
  void computeIslands(QueryResults &q, vector<tIsland> &islands) const;
  
  /**
   * Returns the score of the island composed of the entries of q whose indices
   * are in [i_first, i_last] (both included)
   * @param q
   * @param i_first first index of q of the island
   * @param i_last last index of q of the island
   * @return island score
   */
  double calculateIslandScore(const QueryResults &q, unsigned int i_first, 
    unsigned int i_last) const;

  /**
   * Updates the temporal window by adding the given match <island, id>, such 
   * that the window will contain only those islands which are consistent
   * @param matched_island
   * @param entry_id
   */
  void updateTemporalWindow(const tIsland &matched_island, EntryId entry_id);
  
  /**
   * Returns the number of consistent islands in the temporal window
   * @return number of temporal consistent islands
   */
  inline int getConsistentEntries() const
  {
    return m_window.nentries;
  }
  
  /**
   * Check if an old entry is geometrically consistent (by calculating a 
   * fundamental matrix) with the given set of keys and descriptors
   * @param old_entry entry id of the stored image to check
   * @param keys current keypoints
   * @param descriptors current descriptors associated to the given keypoints
   * @param curvec feature vector of the current entry 
   */
  bool isGeometricallyConsistent_DI(EntryId old_entry, 
    const std::vector<cv::KeyPoint> &keys, 
    const std::vector<TDescriptor> &descriptors, 
    const FeatureVector &curvec) const;
  
  /**
   * Checks if an old entry is geometrically consistent (by using FLANN and 
   * computing an essential matrix by using the neighbour ratio 0.6) 
   * with the given set of keys and descriptors
   * @param old_entry entry id of the stored image to check
   * @param keys current keypoints
   * @param descriptors current descriptors
   * @param flann_structure flann structure with the descriptors of the current entry
   */
  bool isGeometricallyConsistent_Flann(EntryId old_entry,
    const std::vector<cv::KeyPoint> &keys, 
    const std::vector<TDescriptor> &descriptors,
    cv::FlannBasedMatcher &flann_structure) const;

  /**
   * Creates a flann structure from a set of descriptors to perform queries
   * @param descriptors
   * @param flann_structure (out) flann matcher
   */
  void getFlannStructure(const std::vector<TDescriptor> &descriptors, 
    cv::FlannBasedMatcher &flann_structure) const;

  /**
   * Check if an old entry is geometrically consistent (by calculating a
   * fundamental matrix from left-right correspondences) with the given set 
   * of keys and descriptors,
   * without using the direct index
   * @param old_keys keys of old entry
   * @param old_descriptors descriptors of old keys
   * @param cur_keys keys of current entry
   * @param cur_descriptors descriptors of cur keys
   */
  bool isGeometricallyConsistent_Exhaustive(
    const std::vector<cv::KeyPoint> &old_keys,
    const std::vector<TDescriptor> &old_descriptors,
    const std::vector<cv::KeyPoint> &cur_keys,
    const std::vector<TDescriptor> &cur_descriptors) const; 

  /**
   * Calculate the matches between the descriptors A[i_A] and the descriptors
   * B[i_B]. Applies a left-right matching without neighbour ratio 
   * @param A set A of descriptors
   * @param i_A only descriptors A[i_A] will be checked
   * @param B set B of descriptors
   * @param i_B only descriptors B[i_B] will be checked
   * @param i_match_A (out) indices of descriptors matched (s.t. A[i_match_A])
   * @param i_match_B (out) indices of descriptors matched (s.t. B[i_match_B])
   */
  void getMatches_neighratio(const std::vector<TDescriptor> &A, 
    const vector<unsigned int> &i_A, const vector<TDescriptor> &B,
    const vector<unsigned int> &i_B,
    vector<unsigned int> &i_match_A, vector<unsigned int> &i_match_B) const;

protected:

  /// Database
  // The loop detector stores its own copy of the database
  TemplatedDatabase<TDescriptor,F> *m_database;
  
  /// KeyPoints of images
  vector<vector<cv::KeyPoint> > m_image_keys;
  
  /// Descriptors of images
  vector<vector<TDescriptor> > m_image_descriptors;
  
  /// Last bow vector added to database
  BowVector m_last_bowvec;
  
  /// Temporal consistency window
  tTemporalWindow m_window;
  
  /// Parameters of loop detector
  Parameters m_params;
  
  /// To compute the fundamental matrix
  DVision::FSolver m_fsolver;
  
};

// --------------------------------------------------------------------------

template <class TDescriptor, class F> 
TemplatedLoopDetector<TDescriptor,F>::Parameters::Parameters():
  use_nss(true), alpha(0.3), k(4), geom_check(GEOM_DI), di_levels(0)
{
  set(1);
}

// --------------------------------------------------------------------------

template <class TDescriptor, class F> 
TemplatedLoopDetector<TDescriptor,F>::Parameters::Parameters
  (int height, int width, float frequency, bool nss, float _alpha, 
  int _k, GeometricalCheck geom, int dilevels)
  : image_rows(height), image_cols(width), use_nss(nss), alpha(_alpha), k(_k),
    geom_check(geom), di_levels(dilevels)
{
  set(frequency);
}

// --------------------------------------------------------------------------
//设置系统参数
template <class TDescriptor, class F> 
void TemplatedLoopDetector<TDescriptor,F>::Parameters::set(float f)
{
  dislocal = 20 * f;  //设定的查询图片与匹配图片id之差
  max_db_results = 50 * f;  //匹配结果中图片的数量
  min_nss_factor = 0.005;   
  min_matches_per_group = f; //岛内最小图片数
  max_intragroup_gap = 3 * f;  //岛内图片id最大间隔值
  max_distance_between_groups = 3 * f;
  max_distance_between_queries = 2 * f; 

  min_Fpoints = 12;   //适用于模型的最少数据个数
  max_ransac_iterations = 500; //最大迭代次数
  ransac_probability = 0.99;  //迭代过程中从数据集内随机选取的点均为内点的概率
  max_reprojection_error = 2.0;  //容错范围,即重投影误差允许范围
  
  max_neighbor_ratio = 0.6;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
TemplatedLoopDetector<TDescriptor,F>::TemplatedLoopDetector
  (const Parameters &params)
  : m_database(NULL), m_params(params)
{
}

// --------------------------------------------------------------------------
 //创建具有给定参数和给定词典的回环检测器
template<class TDescriptor, class F>
TemplatedLoopDetector<TDescriptor,F>::TemplatedLoopDetector
  (const TemplatedVocabulary<TDescriptor, F> &voc, const Parameters &params)
  : m_params(params) 
{
  m_database = new TemplatedDatabase<TDescriptor, F>(voc, 
    params.geom_check == GEOM_DI, params.di_levels);
  
  m_fsolver.setImageSize(params.image_cols, params.image_rows);
}

// --------------------------------------------------------------------------
//利用给定词典创建新DBoW2数据库
template<class TDescriptor, class F>
void TemplatedLoopDetector<TDescriptor,F>::setVocabulary
  (const TemplatedVocabulary<TDescriptor, F>& voc)
{
  delete m_database;
  m_database = new TemplatedDatabase<TDescriptor, F>(voc, 
    m_params.geom_check == GEOM_DI, m_params.di_levels);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
TemplatedLoopDetector<TDescriptor, F>::TemplatedLoopDetector
  (const TemplatedDatabase<TDescriptor, F> &db, const Parameters &params)
  : m_params(params)
{
  m_database = new TemplatedDatabase<TDescriptor, F>(db.getVocabulary(),
    params.geom_check == GEOM_DI, params.di_levels);
  
  m_fsolver.setImageSize(params.image_cols, params.image_rows);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
template<class T>
TemplatedLoopDetector<TDescriptor, F>::TemplatedLoopDetector
  (const T &db, const Parameters &params)
  : m_params(params)
{
  m_database = new T(db);
  m_database->clear();
  
  m_fsolver.setImageSize(params.image_cols, params.image_rows);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
template<class T>
void TemplatedLoopDetector<TDescriptor, F>::setDatabase(const T &db)
{
  delete m_database;
  m_database = new T(db);
  clear();
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
TemplatedLoopDetector<TDescriptor, F>::~TemplatedLoopDetector(void)
{
  delete m_database;
  m_database = NULL;
}

// --------------------------------------------------------------------------
//为预期数量的图像分配内存
template<class TDescriptor, class F>
void TemplatedLoopDetector<TDescriptor,F>::allocate
  (int nentries, int nkeys)  //nentries:图片集中图片数量,   nkeys:每张图片的描述子个数  
{
  const int sz = (const int)m_image_keys.size(); //sz:数据集中图片的数量
  
  if(sz < nentries)
  {
    m_image_keys.resize(nentries);  //改变数据集中关键点容器的元素数量
    m_image_descriptors.resize(nentries); //改变数据集中描述子容器的元素数量
  }
  
  if(nkeys > 0)
  {
    for(int i = sz; i < nentries; ++i)
    {
      m_image_keys[i].reserve(nkeys);
      m_image_descriptors[i].reserve(nkeys);
    }
  }
  
  m_database->allocate(nentries, nkeys);
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
inline const TemplatedDatabase<TDescriptor, F>& 
TemplatedLoopDetector<TDescriptor, F>::getDatabase() const
{
  return *m_database;
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
inline const TemplatedVocabulary<TDescriptor, F>& 
TemplatedLoopDetector<TDescriptor, F>::getVocabulary() const
{
  return m_database->getVocabulary();
}

// --------------------------------------------------------------------------
//检测当前查询图片是否发生回环,输入keys:当前查询图片所有的关键点、descriptors:当前查询图片所有的描述子，输出match:回环检测结果
template<class TDescriptor, class F>
bool TemplatedLoopDetector<TDescriptor, F>::detectLoop(
  const std::vector<cv::KeyPoint> &keys, 
  const std::vector<TDescriptor> &descriptors,
  DetectionResult &match)
{
  EntryId entry_id = m_database->size(); //entry_id：当前查询图片id    m_database:数据集模板类的指针
  match.query = entry_id; //match:回环检测结果。　match.query：当前查询图片id
  
  BowVector bowvec;  //声明一个词袋向量变量
  FeatureVector featvec;  //声明一个直接索引变量
  
  if(m_params.geom_check == GEOM_DI) //m_params：回环检测参数。m_params.geom_check == GEOM_DI表示进行几何一致性检验
    m_database->getVocabulary()->transform(descriptors, bowvec, featvec,
      m_params.di_levels);  //计算出查询图片的词袋向量、直接索引
  else
    m_database->getVocabulary()->transform(descriptors, bowvec); //不进行几何一致性检验，则仅计算出图片的词袋向量

  if((int)entry_id <= m_params.dislocal)  //当前查询图片id小于等于dislocal：设定的查询图片与匹配图片id之差
  {
    // only add the entry to the database and finish
    m_database->add(bowvec, featvec);  //更新数据库(倒排索引与直接索引)
    match.status = CLOSE_MATCHES_ONLY; //status：回环检测结果的状态。　此处显示查询图片与被考虑匹配图片太过接近
  }
  else
  {
    int max_id = (int)entry_id - m_params.dislocal;  //max_id：匹配结果中图片id的最大值。　
    
    QueryResults qret;  //qret:匹配结果:相似性评分从高到低的一系列图片
    m_database->query(bowvec, qret, m_params.max_db_results, max_id);  //查询当前图片的匹配结果，　m_params.max_db_results:匹配结果中图片的数量

    // update database 更新数据库
    m_database->add(bowvec, featvec); // returns entry_id
    
    if(!qret.empty())
    {
      // factor to compute normalized similarity score, if necessary
      double ns_factor = 1.0;
      
      if(m_params.use_nss) //m_params.use_nss=ture　使用标准化相似性评分,标准化相似性评分=相似性评分除以“最佳值”
      {                                           //计算查询图片与它上一帧的相似性评分，设此值名为“最佳值”
        ns_factor = m_database->getVocabulary()->score(bowvec, m_last_bowvec); 
      }
      
      if(!m_params.use_nss || ns_factor >= m_params.min_nss_factor) //不使用标准化相似性评分　或　最佳值大于等于0.005
      {
        // scores in qret must be divided by ns_factor to obtain the
        // normalized similarity score, but we can
        // speed this up by moving ns_factor to alpha's
        
        // remove those scores whose nss is lower than alpha
        // (ret is sorted in descending score order now)       qret现在按得分降序排序
        removeLowScores(qret, m_params.alpha * ns_factor); //匹配结果中剔除相似性评分低于“alpha*最佳值”的图片 即：匹配结果中剔除那些标准化相似性评分低于alpha的图片
        
        if(!qret.empty())
        {
          // the best candidate is the one with highest score by now  最好的匹配图片是目前得分最高的图片,即qret中第一个结果
          match.match = qret[0].Id;
          
          // compute islands
          vector<tIsland> islands;  //islands:储存岛的容器
          computeIslands(qret, islands); //将匹配结果分组成岛
          // this modifies qret and changes the score order 
          
          // get best island
          if(!islands.empty())
          {
            const tIsland& island = 
              *std::max_element(islands.begin(), islands.end()); //选出得分最高的岛
            
            // check temporal consistency of this island 
            updateTemporalWindow(island, entry_id); //更新时间一致性检验窗口
            
            // get the best candidate (maybe match)
            match.match = island.best_entry; //将岛中得分最高的图片id赋给match.match
            
            if(getConsistentEntries() > m_params.k) //若时间一致性检验窗口中图片数量大于 k(设定值，此处为1),即通过了时间一致性检验
            {
              // candidate loop detected
              // check geometry
              bool detection;

              if(m_params.geom_check == GEOM_DI) //使用直接索引进行几何一致性检验，geom_check：几何检验方式。GEOM_DI：使用直接索引进行几何检验
              { 
 

                // all the DI stuff is implicit in the database
                detection = isGeometricallyConsistent_DI(island.best_entry, 
                  keys, descriptors, featvec); //通过几何一致性检验,则detection = true
               

              }
              else if(m_params.geom_check == GEOM_FLANN) //GEOM_FLANN：使用快速最近邻搜索进行几何一致性检验
              {
                cv::FlannBasedMatcher flann_structure;     
                getFlannStructure(descriptors, flann_structure);
                            
                detection = isGeometricallyConsistent_Flann(island.best_entry, 
                  keys, descriptors, flann_structure); 
              }
              else if(m_params.geom_check == GEOM_EXHAUSTIVE)  //GEOM_EXHAUSTIVE：使用穷尽搜索进行几何一致性检验
              { 
                detection = isGeometricallyConsistent_Exhaustive(
                  m_image_keys[island.best_entry], 
                  m_image_descriptors[island.best_entry],
                  keys, descriptors);            
              }
              else // GEOM_NONE, accept the match 不进行几何一致性检验，直接接受此匹配
              {
                detection = true;
              }
              
              if(detection) //detection 为 true　几何一致性检验通过，回环检测的状态为检测到了回环
              {
                match.status = LOOP_DETECTED; 
              }
              else   //detection 为 false　回环检测结果的状态为几何一致性检验不通过
              {
                match.status = NO_GEOMETRICAL_CONSISTENCY;  
              }
              
            } // if enough temporal matches
            else  //如果没有足够的时间匹配,则回环检测结果的状态为NO_TEMPORAL_CONSISTENCY
            {
              match.status = NO_TEMPORAL_CONSISTENCY;
            }
            
          } // if there is some island
          else //若岛的容器为空,则回环检测结果的状态为NO_GROUPS
          {
            match.status = NO_GROUPS;
          }
        } // if !qret empty after removing low scores
        else  //若剔除低评分后候选匹配容器为空,则回环检测结果的状态为LOW_SCORES
        {
          match.status = LOW_SCORES;
        }
      } // if (ns_factor > min normal score)
      else //若最佳值小于阈值,则回环检测结果的状态为LOW_NSS_FACTOR
      {
        match.status = LOW_NSS_FACTOR;
      }
    } // if(!qret.empty())
    else //若查询图片的匹配结果为空，则回环检测结果的状态为NO_DB_RESULTS
    {
      match.status = NO_DB_RESULTS;
    }
  }

  // update record  更新数据库
  // m_image_keys and m_image_descriptors have the same length
  if(m_image_keys.size() == entry_id) //若数据库中图片的数量与当前查询图片id相等
  {
    m_image_keys.push_back(keys);  //将当前查询图片关键点压入数据库的关键点容器中
    m_image_descriptors.push_back(descriptors);  //将当前查询图片的描述子压入数据库的描述子容器中
  }
  else //若数据库中图片的数量与当前查询图片id不等
  {
    m_image_keys[entry_id] = keys; //将当前查询图片的关键点放入数据库关键点容器的对应位置
    m_image_descriptors[entry_id] = descriptors;  //将当前查询图片的描述子放入数据库描述子容器的对应位置
  }
  
  // store this bowvec if we are going to use it in next iteratons
  if(m_params.use_nss && (int)entry_id + 1 > m_params.dislocal) //若使用标准化相似性评分　且　当前查询图片id加１大于　设定的id差,则保存查询图片的词袋向量为m_last_bowvec
  {
    m_last_bowvec = bowvec;
  }

  return match.detection(); //返回检测结果　ture（检测到回环） 或　false
}

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
inline void TemplatedLoopDetector<TDescriptor, F>::clear()
{
  m_database->clear();
  m_window.nentries = 0;
}

// --------------------------------------------------------------------------
//将匹配结果分组成岛
template<class TDescriptor, class F>
void TemplatedLoopDetector<TDescriptor, F>::computeIslands
  (QueryResults &q, vector<tIsland> &islands) const     //q:匹配结果　　islands：储存岛的容器
{
  islands.clear();
  
  if(q.size() == 1) //若匹配结果中只有一张图片
  {
    islands.push_back(tIsland(q[0].Id, q[0].Id, calculateIslandScore(q, 0, 0))); //将一个岛的信息(岛内第一张图片id，岛内最后一张图片id,岛内所有图片的评分之和)压入岛的容器中
    islands.back().best_entry = q[0].Id;   //best_entry:岛内得分最高的图片id
    islands.back().best_score = q[0].Score;  //best_score:岛内单张图片的最高得分
  }
  else if(!q.empty())
  {
    // sort query results in ascending order of ids 
    std::sort(q.begin(), q.end(), Result::ltId); //按id升序排序匹配结果
    
    // create long enough islands
    QueryResults::const_iterator dit = q.begin();
    int first_island_entry = dit->Id;  //first_island_entry:此时为匹配结果中第一张图片的id，在下面循环内意义为岛内第一张图片的id
    int last_island_entry = dit->Id;  //last_island_entry:此时为匹配结果中第一张图片的id,在下面循环内意义为岛内最后一张图片的id
    
    // these are indices of q  用来标记岛内首尾图片在匹配结果中的序号
    unsigned int i_first = 0;  
    unsigned int i_last = 0;
    
    double best_score = dit->Score;
    EntryId best_entry = dit->Id;

    ++dit;
    for(unsigned int idx = 1; dit != q.end(); ++dit, ++idx) //遍历匹配结果中的图片
    {
      if((int)dit->Id - last_island_entry < m_params.max_intragroup_gap) //若当前图片id减去上一张图片id　小于　岛内图片id最大间隔(设定值)，则可放入此岛内
      {
        // go on until find the end of the island
        last_island_entry = dit->Id;
        i_last = idx;
        if(dit->Score > best_score) //找出岛内最高得分与其图片id
        {
          best_score = dit->Score;
          best_entry = dit->Id;
        }
      }
      else
      {
        // end of island reached 到达岛尾
        int length = last_island_entry - first_island_entry + 1;  //length:当前岛内图片数量
        if(length >= m_params.min_matches_per_group) //若当前岛内图片数量大于等于min_matches_per_group:岛内最小图片数(设定值),则创建此岛
        {
          islands.push_back( tIsland(first_island_entry, last_island_entry,
            calculateIslandScore(q, i_first, i_last)) ); //将此岛信息压入岛的容器，calculateIslandScore()：计算匹配结果q中第i_first张到第i_last张图片相似性评分的和
          
          islands.back().best_score = best_score; 
          islands.back().best_entry = best_entry;
        }
        
        // prepare next island 准备建下一个岛(给下一个岛的信息赋初值)
        first_island_entry = last_island_entry = dit->Id;
        i_first = i_last = idx;
        best_score = dit->Score;
        best_entry = dit->Id;
      }
    }
    // add last island 添加最后一个岛
    if(last_island_entry - first_island_entry + 1 >= 
      m_params.min_matches_per_group)   
    {
      islands.push_back( tIsland(first_island_entry, last_island_entry,
        calculateIslandScore(q, i_first, i_last)) );
        
      islands.back().best_score = best_score;
      islands.back().best_entry = best_entry;
    }
  }
  
}

// --------------------------------------------------------------------------
//计算匹配结果q中第i_first到第i_last张图片的相似性评分的和
template<class TDescriptor, class F>
double TemplatedLoopDetector<TDescriptor, F>::calculateIslandScore(
  const QueryResults &q, unsigned int i_first, unsigned int i_last) const
{
  // get the sum of the scores
  double sum = 0;
  while(i_first <= i_last) sum += q[i_first++].Score;
  return sum;
}

// --------------------------------------------------------------------------
//更新时间一致性检验窗口
template<class TDescriptor, class F>
void TemplatedLoopDetector<TDescriptor, F>::updateTemporalWindow
  (const tIsland &matched_island, EntryId entry_id)
{
  // if m_window.nentries > 0, island > m_window.last_matched_island and
  // entry_id > m_window.last_query_id hold
  
  if(m_window.nentries == 0 || int(entry_id - m_window.last_query_id)
    > m_params.max_distance_between_queries) //若时间一致性窗口中图片数为０ 或 当前查询图片id减上次一致性检验图片id大于　两次查询图片id之差的最大值
  {
    m_window.nentries = 1;
  }
  else
  {
    EntryId a1 = m_window.last_matched_island.first; //a1:上一次查询的匹配岛的第一张图片id
    EntryId a2 = m_window.last_matched_island.last;  //a2:上一次查询的匹配岛的最后一张图片id
    EntryId b1 = matched_island.first;  //b1:当前查询的匹配岛的第一张图片id
    EntryId b2 = matched_island.last;   //b2:当前查询的匹配岛的最后一张图片id
    
    bool fit = (b1 <= a1 && a1 <= b2) || (a1 <= b1 && b1 <= a2); //若前后两次查询的匹配岛的图片范围有重叠，则fit=ture
    /* ——>id增大                          
                b1             b2     或　   a1              a2
                       a1                           b1
    */                                    
    if(!fit) //若fit=false 即两次查询的匹配岛的图片范围没有重叠
    {
      int d1 = (int)a1 - (int)b2;
      int d2 = (int)b1 - (int)a2;
      int gap = (d1 > d2 ? d1 : d2); //gap:两次查询的匹配岛的间隔数
      
      fit = (gap <= m_params.max_distance_between_groups); //若gap小于等于　岛间最大间隔id值(设定值),则fit=ture
    }
    
    if(fit) m_window.nentries++; //若fit=ture,则时间一致性窗口中图片数量加1
    else m_window.nentries = 1;
  }
  
  m_window.last_matched_island = matched_island; //将当前查询的岛　赋予last_matched_island
  m_window.last_query_id = entry_id;  //将当前查询图片id赋予last_query_id
}

// --------------------------------------------------------------------------
//利用直接索引对查询图片与匹配图片进行几何一致性检验，输入最佳匹配图片id：old_entry、查询图片关键点：keys、查询图片描述子：descriptors、查询图片直接索引：bowvec,若通过检验则输出ture 若不通过则输出false
template<class TDescriptor, class F>
bool TemplatedLoopDetector<TDescriptor, F>::isGeometricallyConsistent_DI(
  EntryId old_entry, const std::vector<cv::KeyPoint> &keys, 
  const std::vector<TDescriptor> &descriptors, 
  const FeatureVector &bowvec) const
{
  const FeatureVector &oldvec = m_database->retrieveFeatures(old_entry); //得到匹配图片的直接索引:oldvec
  
  // for each word in common, get the closest descriptors 对将图片进行特征点匹配
  
  vector<unsigned int> i_old, i_cur; //分别储存两图片匹配特征点的序号
  
  FeatureVector::const_iterator old_it, cur_it; 
  const FeatureVector::const_iterator old_end = oldvec.end();
  const FeatureVector::const_iterator cur_end = bowvec.end();
  
  old_it = oldvec.begin();
  cur_it = bowvec.begin();
  
  while(old_it != old_end && cur_it != cur_end) //遍历查询图片与匹配图片的直接索引
  {
    if(old_it->first == cur_it->first) //若两图片直接索引的第一个节点id相同
    {
      // compute matches between 
      // features old_it->second of m_image_keys[old_entry] and
      // features cur_it->second of keys
      vector<unsigned int> i_old_now, i_cur_now; //分别储存当前节点下两张图片完成特征点匹配后的特征点序号
      
      getMatches_neighratio(
        m_image_descriptors[old_entry], old_it->second, 
        descriptors, cur_it->second,  
        i_old_now, i_cur_now);    //匹配两图片相同节点id下的特征点
      
      i_old.insert(i_old.end(), i_old_now.begin(), i_old_now.end());//将匹配图片当前节点下匹配好的特征点序号插入容器中
      i_cur.insert(i_cur.end(), i_cur_now.begin(), i_cur_now.end());//将查询图片当前节点下匹配好的特征点序号插入容器中
      
      // move old_it and cur_it forward
      ++old_it;
      ++cur_it;
    }
    else if(old_it->first < cur_it->first) 
    {
      // move old_it forward
      old_it = oldvec.lower_bound(cur_it->first);
      // old_it = (first element >= cur_it.id)
    }
    else
    {
      // move cur_it forward
      cur_it = bowvec.lower_bound(old_it->first);
      // cur_it = (first element >= old_it.id)
    }
  }
  
  // calculate now the fundamental matrix 计算基础矩阵
  if((int)i_old.size() >= m_params.min_Fpoints) //若匹配的点对数量大于等于　设定值（12）
  {
    vector<cv::Point2f> old_points, cur_points;
    
    // add matches to the vectors to calculate the fundamental matrix
    vector<unsigned int>::const_iterator oit, cit;
    oit = i_old.begin();
    cit = i_cur.begin();
    
    for(; oit != i_old.end(); ++oit, ++cit) //将查询图片与匹配图片匹配后的关键点坐标(与特征点序号对应)分别压入old_points、cur_points中
    {
      const cv::KeyPoint &old_k = m_image_keys[old_entry][*oit];
      const cv::KeyPoint &cur_k = keys[*cit];
      
      old_points.push_back(old_k.pt); //pt为关键点坐标
      cur_points.push_back(cur_k.pt);
    }
  
    cv::Mat oldMat(old_points.size(), 2, CV_32F, &old_points[0]); //将匹配图片完成特征点匹配后的关键点坐标写成矩阵形式
    cv::Mat curMat(cur_points.size(), 2, CV_32F, &cur_points[0]); //将查询图片完成特征点匹配后的关键点坐标写成矩阵形式
    
    clock_t start,finish; 
     double totaltime;
     start = clock();
   
    bool temp =  m_fsolver.checkFundamentalMat(oldMat, curMat, 
      m_params.max_reprojection_error, m_params.min_Fpoints,
      m_params.ransac_probability, m_params.max_ransac_iterations);  //利用匹配的特征点来计算基础矩阵，输入oldMat、curMat两个矩阵及一些参数，输出ture(即计算出了符合要求的基础矩阵) 或 false
  
   finish = clock();
     totaltime = (double)(finish - start) / CLOCKS_PER_SEC;
     cout << "\n几何一致性程序的运行时间为" << totaltime << "秒!" << endl;

    return temp;
   
  }
   
  return false;
}

// --------------------------------------------------------------------------
//利用穷尽搜索对查询图片与匹配图片进行几何一致性检验,输入最佳匹配图片关键点：old_keys、最佳匹配图片描述子：old_descriptors、查询图片关键点：cur_keys、查询图片描述子：cur_keys,若通过则输出ture 若不通过则输出false
template<class TDescriptor, class F>
bool TemplatedLoopDetector<TDescriptor, F>::
isGeometricallyConsistent_Exhaustive(
  const std::vector<cv::KeyPoint> &old_keys,
  const std::vector<TDescriptor> &old_descriptors,
  const std::vector<cv::KeyPoint> &cur_keys,
  const std::vector<TDescriptor> &cur_descriptors) const
{
  vector<unsigned int> i_old, i_cur;
  vector<unsigned int> i_all_old, i_all_cur;
  
  i_all_old.reserve(old_keys.size());
  i_all_cur.reserve(cur_keys.size());
  
  for(unsigned int i = 0; i < old_keys.size(); ++i)
  {
    i_all_old.push_back(i);
  }
  
  for(unsigned int i = 0; i < cur_keys.size(); ++i)
  {
    i_all_cur.push_back(i);
  }
  
  getMatches_neighratio(old_descriptors, i_all_old, 
    cur_descriptors, i_all_cur,  i_old, i_cur);
  
  if((int)i_old.size() >= m_params.min_Fpoints)
  {
    // add matches to the vectors to calculate the fundamental matrix
    vector<unsigned int>::const_iterator oit, cit;
    oit = i_old.begin();
    cit = i_cur.begin();
    
    vector<cv::Point2f> old_points, cur_points;
    old_points.reserve(i_old.size());
    cur_points.reserve(i_cur.size());
    
    for(; oit != i_old.end(); ++oit, ++cit)
    {
      const cv::KeyPoint &old_k = old_keys[*oit];
      const cv::KeyPoint &cur_k = cur_keys[*cit];
      
      old_points.push_back(old_k.pt);
      cur_points.push_back(cur_k.pt);
    }
  
    cv::Mat oldMat(old_points.size(), 2, CV_32F, &old_points[0]);
    cv::Mat curMat(cur_points.size(), 2, CV_32F, &cur_points[0]);
    
    return m_fsolver.checkFundamentalMat(oldMat, curMat, 
      m_params.max_reprojection_error, m_params.min_Fpoints,
      m_params.ransac_probability, m_params.max_ransac_iterations);
  }
  
  return false;
} 

// --------------------------------------------------------------------------

template<class TDescriptor, class F>
void TemplatedLoopDetector<TDescriptor, F>::getFlannStructure(
  const std::vector<TDescriptor> &descriptors, 
  cv::FlannBasedMatcher &flann_structure) const
{
  vector<cv::Mat> features(1);
  F::toMat32F(descriptors, features[0]); //将当前查询图片的所有描述子放入一个矩阵中
  
  flann_structure.clear();
  flann_structure.add(features);
  flann_structure.train();
}

// --------------------------------------------------------------------------
//利用快速最近邻搜索对查询图片与匹配图片进行几何一致性检验,
template<class TDescriptor, class F>
bool TemplatedLoopDetector<TDescriptor, F>::isGeometricallyConsistent_Flann
  (EntryId old_entry,
  const std::vector<cv::KeyPoint> &keys, 
  const std::vector<TDescriptor> &descriptors,
  cv::FlannBasedMatcher &flann_structure) const
{
  vector<unsigned int> i_old, i_cur; // indices of correspondences
  
  const vector<cv::KeyPoint>& old_keys = m_image_keys[old_entry];
  const vector<TDescriptor>& old_descs = m_image_descriptors[old_entry];
  const vector<cv::KeyPoint>& cur_keys = keys;
  
  vector<cv::Mat> queryDescs_v(1);
  F::toMat32F(old_descs, queryDescs_v[0]);
  
  vector<vector<cv::DMatch> > matches;
  
  flann_structure.knnMatch(queryDescs_v[0], matches, 2);
  
  for(int old_idx = 0; old_idx < (int)matches.size(); ++old_idx)
  {
    if(!matches[old_idx].empty())
    {
      int cur_idx = matches[old_idx][0].trainIdx;
      float dist = matches[old_idx][0].distance;
      
      bool ok = true;
      if(matches[old_idx].size() >= 2)
      {
        float dist_ratio = dist / matches[old_idx][1].distance;
        ok = dist_ratio <= m_params.max_neighbor_ratio;
      }
      
      if(ok)
      {
        vector<unsigned int>::iterator cit =
          std::find(i_cur.begin(), i_cur.end(), cur_idx);
        
        if(cit == i_cur.end())
        {
          i_old.push_back(old_idx);
          i_cur.push_back(cur_idx);
        }
        else
        {
          int idx = i_old[ cit - i_cur.begin() ];
          if(dist < matches[idx][0].distance)
          {
            i_old[ cit - i_cur.begin() ] = old_idx;
          }
        }
      }
    }
  }
  
  if((int)i_old.size() >= m_params.min_Fpoints)
  {
    // add matches to the vectors for calculating the fundamental matrix
    vector<unsigned int>::const_iterator oit, cit;
    oit = i_old.begin();
    cit = i_cur.begin();
    
    vector<cv::Point2f> old_points, cur_points;
    old_points.reserve(i_old.size());
    cur_points.reserve(i_cur.size());
    
    for(; oit != i_old.end(); ++oit, ++cit)
    {
      const cv::KeyPoint &old_k = old_keys[*oit];
      const cv::KeyPoint &cur_k = cur_keys[*cit];
      
      old_points.push_back(old_k.pt);
      cur_points.push_back(cur_k.pt);
    }
    
    cv::Mat oldMat(old_points.size(), 2, CV_32F, &old_points[0]);
    cv::Mat curMat(cur_points.size(), 2, CV_32F, &cur_points[0]);
    
    return m_fsolver.checkFundamentalMat(oldMat, curMat, 
      m_params.max_reprojection_error, m_params.min_Fpoints,
      m_params.ransac_probability, m_params.max_ransac_iterations);
  }
  
  return false;
}

// --------------------------------------------------------------------------
//匹配两图片相同节点id下的特征点，A、B为两图片的描述子容器，i_A、i_B为两图片当前节点下储存的特征点序号，i_match_A、i_match_B储存匹配完成后两图片在当前节点下的特征点序号
template<class TDescriptor, class F>
void TemplatedLoopDetector<TDescriptor, F>::getMatches_neighratio(
  const vector<TDescriptor> &A, const vector<unsigned int> &i_A,
  const vector<TDescriptor> &B, const vector<unsigned int> &i_B,
  vector<unsigned int> &i_match_A, vector<unsigned int> &i_match_B) const 
{
  i_match_A.resize(0);
  i_match_B.resize(0);
  i_match_A.reserve( min(i_A.size(), i_B.size()) );
  i_match_B.reserve( min(i_A.size(), i_B.size()) );
  
  vector<unsigned int>::const_iterator ait, bit;
  unsigned int i, j;
  i = 0;
  for(ait = i_A.begin(); ait != i_A.end(); ++ait, ++i) //遍历查询图片属于此节点的描述子
  {
    int best_j_now = -1;  //best_j_now:匹配图片属于此节点的描述子中与当前查询图片描述子的最佳匹配项在i_B中的序号
    double best_dist_1 = 1e9;
    double best_dist_2 = 1e9;
    
    j = 0;
    for(bit = i_B.begin(); bit != i_B.end(); ++bit, ++j) //遍历匹配图片属于此节点的描述子
    {
      double d = F::distance(A[*ait], B[*bit]); //计算两描述子距离
            
      // in i        选出查询图片当前描述子在匹配图片属于此节点的描述子中的最佳匹配项
      if(d < best_dist_1)
      {
        best_j_now = j;  
        best_dist_2 = best_dist_1;
        best_dist_1 = d;
      }
      else if(d < best_dist_2)
      {
        best_dist_2 = d;
      }
    }
    //此时best_dist_1：查询图片当前描述子与匹配图片属于此节点的多个描述子的最小距离，best_dist_2：倒数第二小距离
    if(best_dist_1 / best_dist_2 <= m_params.max_neighbor_ratio) //若两值之比小于等于 设定的邻域比的最大值
    {
      unsigned int idx_B = i_B[best_j_now]; //idx_B:匹配图片属于此节点的描述子中与查询图片当前描述子距离最小的描述子的序号
      bit = find(i_match_B.begin(), i_match_B.end(), idx_B); //在i_match_B中找到idx_B
      
      if(bit == i_match_B.end()) //若上一步未找到,则将idx_B压入i_match_B,将查询图片当前描述子序号压入i_match_A
      {
        i_match_B.push_back(idx_B);
        i_match_A.push_back(*ait);
      }
      else //若在i_match_B中找到idx_B,
      {
        unsigned int idx_A = i_match_A[ bit - i_match_B.begin() ]; //计算出与idx_B这个描述子匹配的查询图片的描述子的序号
        double d = F::distance(A[idx_A], B[idx_B]); //计算匹配的两描述子的距离
        if(best_dist_1 < d) //比较idx_B这个描述子两次匹配的距离，最终选择距离小的描述子作为匹配
        {
          i_match_A[ bit - i_match_B.begin() ] = *ait;
        }
      }
        
    }
  }
}

// --------------------------------------------------------------------------
//匹配结果中剔除那些标准化相似性评分低于alpha的图片
template<class TDescriptor, class F>
void TemplatedLoopDetector<TDescriptor, F>::removeLowScores(QueryResults &q,
  double threshold) const
{
  // remember scores in q are in descending order now
  //QueryResults::iterator qit = 
  //  lower_bound(q.begin(), q.end(), threshold, Result::geqv);
  
  Result aux(0, threshold);
  QueryResults::iterator qit = 
    lower_bound(q.begin(), q.end(), aux, Result::geq); //返回第一个不小于阈值的元素的地址
  
  // qit = first element < m_alpha_minus || end
  
  if(qit != q.end()) //若qit没有指向q.end　则仅保留结果中大于等于阈值的元素
  {
    int valid_entries = qit - q.begin();
    q.resize(valid_entries);
  }
}

// --------------------------------------------------------------------------

} // namespace DLoopDetector

#endif
