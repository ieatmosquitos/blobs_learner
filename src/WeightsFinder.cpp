#include <cv.h>
#include <highgui.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include "FileReader.h"
#include "tools.cpp"
#include "mutex.cpp"
// #include <vector>


Mutex * _image_mutex;
Mutex * _learner_mutex;


// returns the "absolute" value of the argument
template <class T> T abs(T arg){
  if (arg < 0) return -arg;
  return arg;
}


// returns the lesser number amongst the two given
template <class T> T min(T arg1, T arg2){
  if(arg1 > arg2) return arg2;
  return arg1;
}


class BlobsLearner;


struct callBackParameters{
  cv::Mat * hsv_image;
  BlobsLearner * blobs_learner;
  Mutex * image_mutex;
  Mutex * learner_mutex;
};


class BlobsLearner{
  
  unsigned int _filter_size;
  double * _filter;
  double _weights[3];	// similarity weights for H, S, and V
  std::vector<cv::Vec3b> _samples;
  double computeProbability(cv::Vec3b hsv);
  void setParameters(unsigned int filter_size, double w1, double w2, double w3);

public:
  BlobsLearner();
  BlobsLearner(unsigned int filter_size, double w1, double w2, double w3);
  ~BlobsLearner(){delete[] _filter;};
  cv::Mat* selectPixels(cv::Mat *image);
  std::vector<FloatCouple> extractCentroids(cv::Mat * hsv_image);
  void addObservation(cv::Vec3b);
  void setSamples(std::vector<cv::Vec3b> samples){this->_samples = samples;};
  void setWeights(double w1, double w2, double w3){this->_weights[0] = w1; this->_weights[1] = w2; this->_weights[2] = w3;};
  void setFilterSize(unsigned int);
  double * getWeights(){return this->_weights;};
  unsigned int getFilterSize(){return _filter_size;};
  void mutate(double prob_mut_filter, double prob_mut_w1, double prob_mut_w2, double prob_mut_w3);
};

// <<<<<<< genetics stuff >>>>>>>
void BlobsLearner::mutate(double prob_mut_filter, double prob_mut_w1, double prob_mut_w2, double prob_mut_w3){
  if((double)rand()/RAND_MAX < prob_mut_filter){
    unsigned int new_size = this->getFilterSize() + (((double)rand()/RAND_MAX) * 40) - 20;
    if(new_size < 0) new_size = 0;
    this->setFilterSize(new_size);
  }
  
  if((double)rand()/RAND_MAX < prob_mut_w1){
    unsigned int new_w1 = this->_weights[0] + (((double)rand()/RAND_MAX) * 5) - 2.5;
    if(new_w1 < 0) new_w1 = 0;
    if(new_w1 > 5) new_w1 = 5;
    this->_weights[0] = new_w1;
  }
  
  if((double)rand()/RAND_MAX < prob_mut_w2){
    unsigned int new_w2 = this->_weights[1] + (((double)rand()/RAND_MAX) * 5) - 2.5;
    if(new_w2 < 0) new_w2 = 0;
    if(new_w2 > 5) new_w2 = 5;
    this->_weights[1] = new_w2;
  }
  
  if((double)rand()/RAND_MAX < prob_mut_w3){
    unsigned int new_w3 = this->_weights[2] + (((double)rand()/RAND_MAX) * 5) - 2.5;
    if(new_w3 < 0) new_w3 = 0;
    if(new_w3 > 5) new_w3 = 5;
    this->_weights[0] = new_w3;
  }
}

BlobsLearner * mating(BlobsLearner * bl1, BlobsLearner * bl2){
  // 50% probability of inheriting features from each of the parents
  double weights[3];
  for(unsigned int i=0; i<3; i++){
    if( rand() > RAND_MAX/2){
      weights[i] = bl1->getWeights()[i];
    }
    else{
      weights[i] = bl2->getWeights()[i];
    }
  }
  double filter_size;
  if( rand() > RAND_MAX/2){
    filter_size = bl1->getFilterSize();
  }
  else{
    filter_size = bl2->getFilterSize();
  }
  
  BlobsLearner * ret = new BlobsLearner(filter_size, weights[0], weights[1], weights[2]);
  return ret;
}


// <<<<<<<<<<<<<<<>>>>>>>>>>>>>>>



double BlobsLearner::computeProbability(cv::Vec3b hsv){
  // look for the most similar sample
  double min_dist = _filter_size;
  for(unsigned int i = 0; i<_samples.size(); i++){
    // get the distance
    double distance = _weights[0] * (double)abs(hsv[0] - _samples[i][0]) + _weights[1] * (double)abs(hsv[1] - _samples[i][1]) + _weights[2] * (double)abs(hsv[2] - _samples[i][2]);
    min_dist = min(min_dist, distance);
  }
  
  // if the minimum distance is outside the range return 0
  if(min_dist >= _filter_size) return 0;
  
  // else return the value of the probability corresponding to the found distance
  return _filter[(unsigned int)min_dist];
}


void BlobsLearner::setFilterSize(unsigned int size){
  delete[] _filter;
  _filter_size = size;
  
  _filter = new double[_filter_size];
  for(unsigned int i = 0; i<_filter_size; i++){
    _filter[i] = 1 - i/_filter_size;
  }
}


void BlobsLearner::setParameters(unsigned int filter_size, double w1, double w2, double w3){
  // set weights
  _weights[0] = w1;
  _weights[1] = w2;
  _weights[2] = w3;
  
  // create the filter
  _filter_size = filter_size;
  _filter = new double[_filter_size];
  for(unsigned int i = 0; i<_filter_size; i++){
    _filter[i] = 1 - i/_filter_size;
  }
}


BlobsLearner::BlobsLearner(){
  setParameters(50, 1, 1, 1);
}


BlobsLearner::BlobsLearner(unsigned int filter_size, double w1, double w2, double w3){
  setParameters(filter_size, w1, w2, w3);
}


void BlobsLearner::addObservation(cv::Vec3b hsv){
  this->_samples.push_back(hsv);
}


void mouseCallBack(int event_type, int x, int y, int flags, void * param){
  callBackParameters * parameters = (callBackParameters *) param;
  cv::Mat * hsv_image = parameters->hsv_image;
  BlobsLearner * b_learner = parameters->blobs_learner;
  Mutex * image_mutex = parameters->image_mutex;
  Mutex * learner_mutex = parameters->learner_mutex;
  
  switch( event_type ){
  	case CV_EVENT_LBUTTONDOWN:
	  {
	    // get H,S,V values
	    image_mutex->lock();
	    cv::Vec3b hsv = hsv_image->at<cv::Vec3b>(y,x);
	    image_mutex->unlock();
	    std::cout << "H:"<<(int)hsv[0] << " S:" << (int)hsv[1] << " V:" << (int)hsv[2] << "\n";
	    
	    // update probabilities
	    b_learner->addObservation(hsv);
	    
	    break;
	  }
  	case CV_EVENT_LBUTTONUP:
	  break;
  	case CV_EVENT_MOUSEMOVE:
	  break;
  	default:
	  break;
  }
}


cv::Mat* BlobsLearner::selectPixels(cv::Mat *image){
  // create the output image, totally black
  int rows = image->rows;
  int cols = image->cols;
  cv::Mat * selected;
  selected = new cv::Mat(rows,cols,CV_8UC1, cv::Scalar(0));
  
  int h, s, v;
  
  // for each pixel in the original image
  for(int y = 0; y < rows; y++){
    for(int x = 0; x < cols; x++){
      // get the HSV values
      cv::Vec3b pixel = image->at<cv::Vec3b>(y,x);
      
      // compute the probability that this is an interesting pixel
      double p = computeProbability(pixel);
      if(p > (double)rand()/RAND_MAX){
	selected->at<uchar>(y,x) = 255;
      }
    }
  }
  
  return selected;
}


std::vector<FloatCouple> BlobsLearner::extractCentroids(cv::Mat * hsv_image){
  cv::Mat * selected = this->selectPixels(hsv_image);
  std::vector<Blob*> blobs;
  getBlobs(selected, &blobs);
  std::vector<Blob*> big_blobs = blobs;
  purgeBlobs(&big_blobs, 10);
  
  std::vector<FloatCouple> centroids;
  getCentroids(&big_blobs, &centroids);

  
  delete selected;
  for(unsigned int i=0; i<blobs.size(); i++){
    delete blobs[i];
  }
  
  return centroids;
}


void configFromFile(std::string filename, double * weights, unsigned int * filter_size){
  FileReader fr(filename);
  if(!fr.is_open()){
    std::cout << "cannot open configuration file, using default configurations" << std::endl;
    return;
  }
  
  std::vector<std::string> textline;
  fr.readLine(&textline);
  while(fr.good()){
    if(textline.size() > 1){
      if((textline[0].compare(std::string("filter_size"))) == 0){	// compare returns 0 if the two strings are equal
	*filter_size = atof(textline[1].c_str());
      }
      if((textline[0].compare(std::string("wH"))) == 0){
	weights[0] = atof(textline[1].c_str());
      }
      if((textline[0].compare(std::string("wS"))) == 0){
	weights[1] = atof(textline[1].c_str());
      }
      if((textline[0].compare(std::string("wV"))) == 0){
	weights[2] = atof(textline[1].c_str());
      }
    }
    textline.clear();
    fr.readLine(&textline);
  }
}


void test(){
  std::vector<FloatCouple> v1;
  std::vector<FloatCouple> v2;
  
  v1.push_back(FloatCouple(2,2));
  v1.push_back(FloatCouple(0.9,1));
  v2.push_back(FloatCouple(1,1));
  v2.push_back(FloatCouple(7,7));
  
  std::cout << "distance between the test maps: " << computeMapsDistance(v1, v2, 10, 3, 2) << std::endl;
}


void init(){
  _image_mutex = new Mutex();
  _learner_mutex = new Mutex();
}


int main(int argc, char**argv){
  init();
  test();
  
  // create the Blobs Learner and set its parameters
  double weights[3];
  weights[0] = 1;
  weights[1] = 1;
  weights[2] = 1;
  unsigned int filter_size = 50;
  if(argc > 1){
    configFromFile(std::string(argv[1]), weights, &filter_size);
  }
  std::cout << "configuration:" << std::endl;
  std::cout << "weights = <" << weights[0] << ", " << weights[1] << ", " << weights[2] << ">" << std::endl;
  std::cout << "filter size: " << filter_size << std:: endl;
  BlobsLearner b_learner(filter_size, weights[0], weights[1], weights[2]);
  
  
  int capture_dev;
  if(argc < 2) capture_dev = CV_CAP_ANY;
  else capture_dev = atoi(argv[1]);
  
  CvCapture* capture = cvCaptureFromCAM(capture_dev);
  if(!capture){  
    fprintf(stderr, "ERROR: capture is NULL \n");  
    getchar();  
    return -1;  
  }
  
  int frame_number = 0;
  IplImage * frame;
  
  const char * main_image_window_name = "camera";
  const char * centroids_window_name = "centroids";
  
  cvNamedWindow(main_image_window_name, CV_WINDOW_NORMAL);
  cvNamedWindow(centroids_window_name, CV_WINDOW_NORMAL);
  cv::moveWindow(main_image_window_name,100,100);
  cv::moveWindow(centroids_window_name, 500,100);
  cvStartWindowThread;
  
  cv::Mat hsv_image;
    
  // prepare the structure for passing arguments to the mouse callback
  callBackParameters to_callback;
  to_callback.hsv_image = &hsv_image;
  to_callback.blobs_learner = &b_learner;
  to_callback.image_mutex = _image_mutex;
  to_callback.learner_mutex = _learner_mutex;
  
  cvSetMouseCallback(main_image_window_name, mouseCallBack, &to_callback);
  
  
  // start the streaming
  char key = -1;
  while(key!=27 && key!='q'){
    
    frame_number = cvGrabFrame(capture);
    
    frame = cvRetrieveFrame(capture,frame_number);
    if(!frame){
      fprintf(stderr,"ERROR: frame is null.. \n");
      getchar();
      continue;
    }
    
    cv::Mat original_image(frame, true);
    
    cvShowImage(main_image_window_name, frame);
    
    cv::cvtColor(original_image,hsv_image, CV_BGR2HSV);
    cv::Mat * selected = b_learner.selectPixels(&hsv_image);
    
    // get the blobs and produce the blobs image
    std::vector<Blob*> blobs;
    getBlobs(selected, &blobs);
    std::vector<Blob*> big_blobs = blobs;
    purgeBlobs(&big_blobs, 10);
    
    cv::Mat blobs_image(original_image.rows, original_image.cols, CV_8UC3, cv::Scalar(0,0,0));
    cv::Mat big_blobs_image(original_image.rows, original_image.cols, CV_8UC3, cv::Scalar(0,0,0));
    
    // get centroids
    std::vector<FloatCouple> centroids;
    getCentroids(&big_blobs, &centroids);
    
    // paint centroids image
    cv::Mat centroids_image(original_image.rows, original_image.cols, CV_8UC3, cv::Scalar(0,0,0));
    centroidsPainter(&centroids_image, &centroids);
    
    cv::imshow(centroids_window_name, centroids_image);
    
    
    // delete stuff
    delete selected;
    for(unsigned int i=0; i<blobs.size(); i++){
      delete blobs[i];
    }
    
    key = (char) cv::waitKey(20); // non blocking, returns -1 if nothing was pressed
  }
    
  cvReleaseCapture(&capture);
  cvDestroyWindow(main_image_window_name);
  
  return 0;
}