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

const char * main_image_window_name = "Original image";
const char * hsv_h_window_name = "HUE";
const char * hsv_s_window_name = "SATURATION";
const char * hsv_v_window_name = "VALUE";
const char * selected_pixels_window_name = "Selected pixels";
const char * blobs_window_name = "Blobs";
const char * big_blobs_window_name = "Big Blobs";
const char * centroids_window_name = "Centroids";
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
  cv::Mat* selectPixels(cv::Mat *image);
  void addObservation(cv::Vec3b);
};


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


void init(){
  // instantiate mutexes
  _image_mutex = new Mutex();
  _learner_mutex = new Mutex();
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


int main(int argc, char**argv){
  init();
  
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
  
  cvNamedWindow(main_image_window_name, CV_WINDOW_NORMAL);
  cvNamedWindow(selected_pixels_window_name, CV_WINDOW_NORMAL);
  cvNamedWindow(hsv_h_window_name, CV_WINDOW_NORMAL);
  cvNamedWindow(hsv_s_window_name, CV_WINDOW_NORMAL);
  cvNamedWindow(hsv_v_window_name, CV_WINDOW_NORMAL);
  cvNamedWindow(blobs_window_name, CV_WINDOW_NORMAL);
  cvNamedWindow(big_blobs_window_name, CV_WINDOW_NORMAL);
  cvNamedWindow(centroids_window_name, CV_WINDOW_NORMAL);
  cv::moveWindow(main_image_window_name,100,100);
  cv::moveWindow(selected_pixels_window_name,450,100);
  cv::moveWindow(hsv_h_window_name, 100,400);
  cv::moveWindow(hsv_s_window_name, 150,450);
  cv::moveWindow(hsv_v_window_name, 200,500);
  cv::moveWindow(blobs_window_name, 600,400);
  cv::moveWindow(big_blobs_window_name, 650,450);
  cv::moveWindow(centroids_window_name, 800, 100);
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
    
    _image_mutex->lock();
    cv::cvtColor(original_image,hsv_image, CV_BGR2HSV);
    _image_mutex->unlock();
    _learner_mutex->lock();
    cv::Mat * selected = b_learner.selectPixels(&hsv_image);
    _learner_mutex->unlock();
    
    // prepare probabilities image
    cv::Mat hsv_h(hsv_image.rows,hsv_image.cols,CV_8UC1, cv::Scalar(0));
    cv::Mat hsv_s(hsv_image.rows,hsv_image.cols,CV_8UC1, cv::Scalar(0));
    cv::Mat hsv_v(hsv_image.rows,hsv_image.cols,CV_8UC1, cv::Scalar(0));
    for(unsigned int r=0; r<hsv_image.rows; r++){
        for(unsigned int c=0; c<hsv_image.cols; c++){
            cv::Vec3b pix = hsv_image.at<cv::Vec3b>(r,c);
            hsv_h.at<uchar>(r,c) = pix[0];
            hsv_s.at<uchar>(r,c) = pix[1];
            hsv_v.at<uchar>(r,c) = pix[2];
        }
    }
    
    cv::imshow(selected_pixels_window_name, *selected);
    
    cv::imshow(hsv_h_window_name, hsv_h);
    cv::imshow(hsv_s_window_name, hsv_s);
    cv::imshow(hsv_v_window_name, hsv_v);
    
    
    // get the blobs and produce the blobs image
    std::vector<Blob*> blobs;
    getBlobs(selected, &blobs);
    std::vector<Blob*> big_blobs = blobs;
    purgeBlobs(&big_blobs, 10);
    
    cv::Mat blobs_image(original_image.rows, original_image.cols, CV_8UC3, cv::Scalar(0,0,0));
    blobsPainter(&blobs_image, &blobs);
    cv::Mat big_blobs_image(original_image.rows, original_image.cols, CV_8UC3, cv::Scalar(0,0,0));
    blobsPainter(&big_blobs_image, &big_blobs);
    
    cv::imshow(blobs_window_name, blobs_image);
    cv::imshow(big_blobs_window_name, big_blobs_image);
    
    
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
