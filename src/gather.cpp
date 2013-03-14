#include <iostream>
#include <fstream>
#include <vector>
#include <cv.h>
#include <highgui.h>
#include "mutex.cpp"
#include <vector>

struct hsv_data{
  int x;
  int y;
  uchar h;
  uchar s;
  uchar v;
};

Mutex * _vector_mutex;
std::vector<hsv_data> _data;

void mouseCallBack_RGB(int event_type, int x, int y, int flags, void* param){
  cv::Mat* image = (cv::Mat*) param;
  switch( event_type ){
  case CV_EVENT_LBUTTONDOWN:
    {
      std::cout << "x:" << x << "\ty:" << y;
      cv::Vec3b bgr = image->at<cv::Vec3b>(y,x);
      uchar blue = bgr[0];
      uchar green = bgr[1];
      uchar red = bgr[2];
      std::cout << "\tR:"<<(int)red << "\tG:" << (int)green << "\tB:" << (int)blue << std::endl;
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


void mouseCallBack_HSV(int event_type, int x, int y, int flags, void* param){
  cv::Mat * image = (cv::Mat *) param;
  switch( event_type ){
  	case CV_EVENT_LBUTTONDOWN:
	  {
	    cv::Vec3b hsv;
	    std::cout << "x:" << x << "\ty:" << y;
	    hsv = image->at<cv::Vec3b>(y,x);
	    std::cout << "\tH:" << (int)(hsv[0]) << "\tS:" << (int)(hsv[1]) << "\tV:" << (int)(hsv[2]) << std::endl;
	    hsv_data newdata;
	    newdata.x = x;
	    newdata.y = y;
	    newdata.h = hsv[0];
	    newdata.s = hsv[1];
	    newdata.v = hsv[2];
	    _vector_mutex->lock();
	    _data.push_back(newdata);
	    _vector_mutex->unlock();
	    break;
	  }
  	default:
	  break;
  }
}


void processArguments(int argc, char ** argv){
  if(argc < 3){
    std::cout << "Usage: gathersample <image file name> <output file name>" << std::endl;
    exit(0);
  }
}


void writeOut(const char* image_file_name, const char* output_file_name){
  std::ofstream fout(output_file_name, std::ios_base::app);
  
  if(!fout.is_open()){
    std::cout << "WOOPS! Something went wrong opening file " << output_file_name << " for writing... nothing was written" << std::endl;
    return;
  }
  
  
  fout << "images/" << image_file_name << std::endl;
  for(unsigned int i = 0; i < _data.size(); i++){
    fout << _data[i].x << " " << _data[i].y << " " << (int)(_data[i].h) << " " << (int)(_data[i].s) << " " << (int)(_data[i].v) << std::endl;
  }
  fout << "--next--" << std::endl;
  std::cout << "written everything on " << output_file_name << std::endl;
}

int main(int argc, char** argv){
  processArguments(argc, argv);
  _vector_mutex = new Mutex();
  
  cv::Mat image = cv::imread(argv[1],1);
  
  if(!image.data){
    std::cout << "unable to read image " << argv[1] << std::endl << "Exiting..." << std::endl;
    exit(1);
  }
  
  cv::Mat hsv_image;
  cv::cvtColor(image,hsv_image, CV_BGR2HSV);
  
  const char* window_name = argv[1];
  
  cv::namedWindow(window_name, CV_WINDOW_AUTOSIZE);
  cv::imshow(window_name, image);
  
  cvSetMouseCallback(window_name, mouseCallBack_HSV, (void*) &hsv_image);
  
  std::cout << "click on the areas to be recognized, then press ESC or Q to quit" << std::endl;
  
  int button = cv::waitKey(0);
  while(button!=27 && (char)button!='q'){
    button = cv::waitKey(0);
  }
  
  writeOut(argv[1], argv[2]);
  
  
  return 0;
}
