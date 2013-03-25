#include <cv.h>
#include <highgui.h>
#include <cstdlib>
#include <iostream>
#include "FileReader.h"
#include "mutex.cpp"
// #include <vector>

const char * main_image_window_name = "Original image";
const char * hsv_h_window_name = "HUE";
const char * hsv_s_window_name = "SATURATION";
const char * hsv_v_window_name = "VALUE";
const char * selected_pixels_window_name = "Selected pixels";
const char * blobs_window_name = "Blobs";
const char * big_blobs_window_name = "Big Blobs";
Mutex * _image_mutex;
Mutex * _learner_mutex;


class BlobsLearner;


// Structure created to identify an <x,y> couple, used for storing pixels positions
struct Coordinate{
  int x;
  int y;
  
  Coordinate(int x, int y){
    this->x = x;
    this->y = y;
  }
};


// Structure used to identify a connected area of pixels
struct Blob{
    std::vector<Coordinate> points;
    
    void add(int y, int x){
      this->points.push_back(Coordinate(x,y));
    }
};


struct callBackParameters{
  cv::Mat * hsv_image;
  BlobsLearner * blobs_learner;
  Mutex * image_mutex;
  Mutex * learner_mutex;
};


template <class T> T max(T arg1, T arg2){
  if(arg1>arg2) return arg1;
  return arg2;
}


class BlobsLearner{
  double Hp[256];	// probability distribution for the Hue Channel
  double Sp[256];	// probability distribution for the Saturation Channel
  double Vp[256];	// probability distribution for the Value Channel
  double * probabilities[3];
  
  double _decay[3];	// decay rate for H, S and V
  unsigned int _filter_size;	// the range of influence of the addObservation function
  double * _filter;
  
  void setParameters(double * probs, double * decay_rate, unsigned int filter_size);
  void computeFilter();
  
public:
  BlobsLearner();
  BlobsLearner(double * hue_p, double * decay_rate, unsigned int filter_size);
  cv::Mat* selectPixels(cv::Mat *image);
  void setDecay(double d){this->_decay[0] = d;this->_decay[1] = d; this->_decay[2] = d;};
  void setFilterSize(unsigned int size);
  void addObservation(uchar hue, uchar sat, uchar val);
  double getProb(unsigned int channel, unsigned int at){return this->probabilities[channel][at];};
  double ** getProbs(){return probabilities;};
};


void BlobsLearner::computeFilter(){
  _filter = new double[_filter_size];
  for(unsigned int i = 0; i<_filter_size; i++){
    // quadratic
    _filter[i] = (1 - pow(i/this->_filter_size, 2));
    
    // linear
    // _filter[i] = (1 - i/this->_filter_size);
  }
}


void BlobsLearner::setFilterSize(unsigned int size){
  delete this->_filter;
  this->_filter_size = size;
  this->computeFilter();
}


void BlobsLearner::setParameters(double * probs, double * decay_rate, unsigned int filter_size){
  for(unsigned int i = 0; i<255; i++){
    Hp[i] = probs[0];
    Sp[i] = probs[1];
    Vp[i] = probs[2];
  }
  this->probabilities[0] = Hp;
  this->probabilities[1] = Sp;
  this->probabilities[2] = Vp;
  
  _decay[0] = decay_rate[0];
  _decay[1] = decay_rate[1];
  _decay[2] = decay_rate[2];
  
  this->_filter_size = filter_size;
  this->computeFilter();
}


BlobsLearner::BlobsLearner(){
  double probs[3];
  probs[0] = 0.1;
  probs[1] = 0.1;
  probs[2] = 0.1;
  double decay[3];
  decay[0] = 0.02;
  decay[1] = 0.02;
  decay[2] = 0.02;
  unsigned int filter_size = 50;
  this->setParameters(probs, decay, filter_size);
}


BlobsLearner::BlobsLearner(double * probs, double * decay_rate, unsigned int filter_size){
  this->setParameters(probs, decay_rate, filter_size);
}


void BlobsLearner::addObservation(uchar hue, uchar sat, uchar val){
  for(unsigned int i=0; i<255; i++){
    if(this->Hp[i] < this->_decay[0] ) this->Hp[i] = 0;
    else this->Hp[i] = this->Hp[i] - this->_decay[0];
    
    if(this->Sp[i] < this->_decay[1] ) this->Sp[i] = 0;
    else this->Sp[i] = this->Sp[i] - this->_decay[1];
    
    if(this->Vp[i] < this->_decay[2] ) this->Vp[i] = 0;
    else this->Vp[i] = this->Vp[i] - this->_decay[2];
  }
  
  double increase_value = 1;
  
  this->Hp[hue] += increase_value;
  if(Hp[hue] > 1) Hp[hue] = 1;
  this->Sp[sat] += increase_value;
  if(Hp[sat] > 1) Hp[sat] = 1;
  this->Vp[val] += increase_value;
  if(Hp[val] > 1) Hp[val] = 1;
  for (unsigned int i = 1; i < this->_filter_size; i++){
    uchar index;
    
    index = hue - i;
    if(index > 0 && index < 255) this->Hp[index] += increase_value*_filter[i];
    if(this->Hp[index] > 1) this->Hp[index] = 1;
    
    index = hue + i;
    if(index > 0 && index < 255) this->Hp[index] += increase_value*_filter[i];
    if(this->Hp[index] > 1) this->Hp[index] = 1;
    
    index = sat - i;
    if(index > 0 && index < 255) this->Sp[index] += increase_value*_filter[i];
    if(this->Sp[index] > 1) this->Sp[index] = 1;
    
    index = sat + i;
    if(index > 0 && index < 255) this->Sp[index] += increase_value*_filter[i];
    if(this->Sp[index] > 1) this->Sp[index] = 1;
    
    index = val - i;
    if(index > 0 && index < 255) this->Vp[index] += increase_value*_filter[i];
    if(this->Vp[index] > 1) this->Vp[index] = 1;
    
    index = val + i;
    if(index > 0 && index < 255) this->Vp[index] += increase_value*_filter[i];
    if(this->Vp[index] > 1) this->Vp[index] = 1;
  }

}



void getBlobsStepReiterative(cv::Mat *image, int y, int x, Blob *blob, int rows, int cols, bool * visited){
    
  std::vector<Coordinate> check_these;
    
  Coordinate pixel(x,y);
  check_these.push_back(pixel);
    
  for(unsigned int i=0; i<check_these.size(); i++){
    
    pixel = check_these[i];
      
    // add this point to the blob
    blob->add(pixel.y, pixel.x);
      
    // put neighbors in the to-check list
    for (int r=pixel.y-1; r<=pixel.y+1; r++){   // r will scan the rows
      //            std::cout<<"row:" << r << '\n';
      if((r<0)||(r>=rows)){
	//                std::cout<<"OUT OF BOUNDS -- continue\n";
	continue;
      }
      for (int c=pixel.x-1; c<=pixel.x+1; c++){       // c will scan the columns
	//                std::cout<<"col:" << c << '\n';
	if ((c<0)||(c>=cols)){  // don't consider out of bounds pixels
	  //                    std::cout<<"OUT OF BOUNDS -- continue\n";
	  continue;
	}
	if(!visited[r*cols+c]){ // if that neighbor has NOT been visited
	  if(image->at<uchar>(r,c) > 0){      // if it is colored
	    // set point as visited
	    visited[(r*cols)+c] = true;
	    check_these.push_back(Coordinate(c,r));
	  }
	  else{       // if it is black
	    visited[r*cols+c] = true;       // just set it as visited
	  }
	}
	  
      }
    }
  }
}
    
/*!
 * getBlobs extracts the blobs (I.E. connected areas) from the image.
 * A pixel is considered "interesting" if it is not black.
 * \param image pointer to the image
 * \param blobs pointer to the vector where blobs must be stored
 */
void getBlobs(cv::Mat *image, std::vector<Blob*> *blobs){
  // get informations about the image size
  int rows = image->rows;
  int cols = image->cols;
  int totalcells = rows*cols;
        
  // prepare the visited list
  bool visited[totalcells];
  for(int i=0; i<totalcells; i++){
    visited[i] = false;
  }
        
  // search for the Blobs
  for (int y = 0; y<rows; y++){   // for each row
    for (int x = 0; x<cols; x++){       // for each column
      //                std::cout<<"\npixel <" << y <<','<<x<<">:\n";
      if(!visited[(y*cols)+x]){       // if this pixel has not been visited
	//                    std::cout<<"pixel not yet visited\n";
	if(image->at<uchar>(y,x)){  // and if this is not black
	  //                        std::cout<<"good point, creating a new blob\n";
	  Blob *blob = new Blob;  // create a new Blob
	  //                        std::cout<<"new blob created\n";
	  // getBlobsLoop(image,y,x,blob,rows,cols,visited);      // expand to fill the blob
	  getBlobsStepReiterative(image,y,x,blob,rows,cols,visited);
	  //                        std::cout<<"blob expanded\n";
	  blobs->push_back(blob);
	  //                        std::cout<<"blob added to the vector\n";
	}
	else{
	  visited[y*cols+x] = true;
	}
      }
      //                else{
      //                    std::cout<<"pixel already visited\n";
      //                }
    }
  }
}
    

// this function purges the blobs vector.
// at the moment, this only removes tiny blobs
// possible developments:
//          • shape filter
//          • density filter
void purgeBlobs(std::vector<Blob*> *blobs, int size_threshold){
  //        std::cout<<"\tPURGING BLOBS:\n";
  //        std::cout<<"\tthreshold: "<< size_threshold <<'\n';
  int blobs_sizes[blobs->size()];
  int max_size = 0;
  // first scan to find the maximum size
  for(unsigned int i = 0; i<blobs->size(); i++){
    blobs_sizes[i] = (*blobs)[i]->points.size();
    if(blobs_sizes[i] > max_size){
      max_size = blobs_sizes[i];
    }
  }
  //        std::cout<<"\tthe bigger blob is " << max_size << " pixels\n";
        
  // second scan to select big blobs
  Blob* blobs_temp[blobs->size()];
  int copied_blobs=0;
  for(unsigned int i = 0; i<blobs->size(); i++){
    if(max_size/blobs_sizes[i] < size_threshold){
      //                std::cout<<"\tBIG blob: " << blobs_sizes[i] << " pixels, ACCEPTED\n";
      blobs_temp[copied_blobs++] = (*blobs)[i];
    }
    else{
      //                std::cout<<"\tSMALL blob: " << blobs_sizes[i] << " pixels, REFUSED\n";
    }
  }
        
  // clear the blobs vector and reinsert only the big blobs
  blobs->clear();
  for(int i=0; i<copied_blobs; i++){
    blobs->push_back(blobs_temp[i]);
  }
}
    
/*!
 * blobsPainter draws on the given image the blobs in the given vector, each blob has a different color.
 * \param image pointer to the image
 * \param blobs pointer to the blobs vector
 */
void blobsPainter(cv::Mat *image, std::vector<Blob*> *blobs){
  int needed_colors = blobs->size();
  int layers_per_channel = 1;;
  if (needed_colors>3){
    layers_per_channel = needed_colors/3;
  }
        
  for(unsigned int i=0; i<blobs->size(); i++){
    // select color
    int channel = i%3;
    int layer = (i/3);
    uchar color = (uchar)(255 - (200/layers_per_channel)*layer);
            
    // fill with that color all the pixels of the blob
    for (unsigned int k=0; k<(*blobs)[i]->points.size(); k++){
      Coordinate *c = &((*blobs)[i]->points.at(k));
      cv::Vec3b *p = &(image->at<cv::Vec3b>(c->y,c->x));
      (*p)[channel] = color;
    }
  }
        
}

void mouseCallBack(int event_type, int x, int y, int flags, void * param){
  callBackParameters * parameters = (callBackParameters *) param;
  cv::Mat * hsv_image = parameters->hsv_image;
  BlobsLearner * b_learner = parameters->blobs_learner;
  Mutex * image_mutex = parameters->image_mutex;
  Mutex * learner_mutex = parameters->learner_mutex;
  
  // std::cout << "learner H probability: " << std::endl;
  // for(unsigned int i=0; i<255; i++){
  //   std::cout << " " << b_learner->getProb(0,i);
  // }
  // std::cout << std::endl;
  
  switch( event_type ){
  	case CV_EVENT_LBUTTONDOWN:
	  {
	    // get H,S,V values
	    image_mutex->lock();
	    cv::Vec3b hsv = hsv_image->at<cv::Vec3b>(y,x);
	    image_mutex->unlock();
	    uchar hue = hsv[0];
	    uchar sat = hsv[1];
	    uchar val = hsv[2];
	    std::cout << "H:"<<(int)hue << " S:" << (int)sat << " V:" << (int)val << "\n";
	    
	    // update probabilities
	    b_learner->addObservation(hue,sat,val);
	    
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
      h = pixel[0];
      s = pixel[1];
      v = pixel[2];
      // compute the aggregate probability
      double p = (this->Hp)[h] * (this->Sp)[s] * (this->Vp)[v];
      if(p > (double)rand()/RAND_MAX){
	selected->at<uchar>(y,x) = 255;
      }
    }
  }
  
  return selected;
}


void paintProbabilities(cv::Mat * in_image, cv::Mat * out_image, double * probs){
  for(unsigned int r=0; r<in_image->rows; r++){
    for(unsigned int c=0; c<in_image->cols; c++){
      double prob = probs[in_image->at<uchar>(r,c)];
      out_image->at<uchar>(r,c) = (uchar)(prob*255);
    }
  }
}


void init(){
  _image_mutex = new Mutex();
  _learner_mutex = new Mutex();
}


void configFromFile(std::string filename, double *probs, unsigned int * filter_size, double *decay){
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
      if((textline[0].compare(std::string("Hp"))) == 0){
	probs[0] = atof(textline[1].c_str());
      }
      if((textline[0].compare(std::string("Sp"))) == 0){
	probs[1] = atof(textline[1].c_str());
      }
      if((textline[0].compare(std::string("Vp"))) == 0){
	probs[2] = atof(textline[1].c_str());
      }
      if((textline[0].compare(std::string("Hd"))) == 0){
	decay[0] = atof(textline[1].c_str());
      }
      if((textline[0].compare(std::string("Sd"))) == 0){
	decay[1] = atof(textline[1].c_str());
      }
      if((textline[0].compare(std::string("Vd"))) == 0){
	decay[2] = atof(textline[1].c_str());
      }
      
    }
    textline.clear();
    fr.readLine(&textline);
  }
}


int main(int argc, char**argv){
  init();
  
  // create the Blobs Learner and set its parameters
  double probs[3];
  probs[0] = 0.1;
  probs[1] = 0.1;
  probs[2] = 0.1;
  double decay[3];
  decay[0] = 0.02;
  decay[1] = 0.02;
  decay[2] = 0.02;
  unsigned int filter_size = 10;
  if(argc > 1){
    configFromFile(std::string(argv[1]), probs, &filter_size, decay);
  }
  std::cout << "configurations:" << std::endl;
  std::cout << "probs HSV: <" << probs[0] << ", " << probs[1] << ", " << probs[2] << ">" << std::endl;
  std::cout << "decay HSV: <" << decay[0] << ", " << decay[1] << ", " << decay[2] << ">" << std::endl;
  BlobsLearner b_learner(probs, decay, filter_size);
  
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
  cv::moveWindow(main_image_window_name,100,100);
  cv::moveWindow(selected_pixels_window_name,500,100);
  cv::moveWindow(hsv_h_window_name, 100,400);
  cv::moveWindow(hsv_s_window_name, 150,450);
  cv::moveWindow(hsv_v_window_name, 200,500);
  cv::moveWindow(blobs_window_name, 600,400);
  cv::moveWindow(big_blobs_window_name, 650,450);
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
    cv::Mat hsv_Hp(hsv_image.rows,hsv_image.cols,CV_8UC1, cv::Scalar(0));
    paintProbabilities(&hsv_h, &hsv_Hp, b_learner.getProbs()[0]);
    cv::Mat hsv_Sp(hsv_image.rows,hsv_image.cols,CV_8UC1, cv::Scalar(0));
    paintProbabilities(&hsv_s, &hsv_Sp, b_learner.getProbs()[1]);
    cv::Mat hsv_Vp(hsv_image.rows,hsv_image.cols,CV_8UC1, cv::Scalar(0));
    paintProbabilities(&hsv_v, &hsv_Vp, b_learner.getProbs()[2]);

    
    cv::imshow(selected_pixels_window_name, *selected);
    
    cv::imshow(hsv_h_window_name, hsv_Hp);
    cv::imshow(hsv_s_window_name, hsv_Sp);
    cv::imshow(hsv_v_window_name, hsv_Vp);
    
    
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
