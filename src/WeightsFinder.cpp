#include <cv.h>
#include <highgui.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include <string>
#include "FileReader.h"
#include "tools.cpp"
#include <pthread.h>
// #include <vector>

#define LIST_SIZE 40
#define MAX_GENERATIONS 40
#define PROB_MUT_FILTER 0.5
#define PROB_MUT_W1 0.5
#define PROB_MUT_W2 0.5
#define PROB_MUT_W3 0.5
#define PROB_MUT_THRESHOLD 0.5
#define NumThreads 4

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

class BlobsLearner{
  
  unsigned int _filter_size;
  double * _filter;
  double _weights[3];	// similarity weights for H, S, and V
  double _blobs_size_threshold;
  std::vector<cv::Vec3b> _samples;
  double computeProbability(cv::Vec3b hsv);
  void setParameters(unsigned int filter_size, double w1, double w2, double w3, double blobs_size_threshold);

public:
  BlobsLearner();
  BlobsLearner(unsigned int filter_size, double w1, double w2, double w3, double blobs_size_threshold);
  ~BlobsLearner(){delete[] _filter;};
  cv::Mat* selectPixels(cv::Mat *image);
  std::vector<FloatCouple> extractCentroids(cv::Mat * hsv_image);
  void addObservation(cv::Vec3b);
  void setSamples(std::vector<cv::Vec3b> samples){this->_samples = samples;};
  void setWeights(double w1, double w2, double w3){this->_weights[0] = w1; this->_weights[1] = w2; this->_weights[2] = w3;};
  void setFilterSize(unsigned int);
  double * getWeights(){return this->_weights;};
  unsigned int getFilterSize(){return _filter_size;};
  void mutate(double prob_mut_filter, double prob_mut_w1, double prob_mut_w2, double prob_mut_w3, double prob_mut_threshold);
  double getBlobsThreshold(){return this->_blobs_size_threshold;};
};


// <<<<<<< Multithread stuff >>>>>>>
struct thread_struct{
  BlobsLearner * learner;
  cv::Mat * image;
  std::vector<FloatCouple> * test_coordinates;
  double max_dist;
  double extra;
  double lack;
  double * put_result_here;
};


void * threadFunc(void * t_struct){
  thread_struct * arg = (thread_struct *) t_struct;
  *(arg->put_result_here) = computeMapsDistance(arg->learner->extractCentroids(arg->image), *(arg->test_coordinates), arg->max_dist, arg->extra, arg->lack);
}


// <<<<<<<<<<<<<<<>>>>>>>>>>>>>>>

// <<<<<<< genetics stuff >>>>>>>
void BlobsLearner::mutate(double prob_mut_filter, double prob_mut_w1, double prob_mut_w2, double prob_mut_w3, double prob_mut_threshold){
  if((double)rand()/RAND_MAX < prob_mut_filter){
    int new_size = (int)(this->getFilterSize() + (((double)rand()/RAND_MAX) * 40) - 20);
    if(new_size < 1) new_size = 1;
    this->setFilterSize((unsigned int)new_size);
  }
  
  if((double)rand()/RAND_MAX < prob_mut_w1){
    double new_w1 = this->_weights[0] + (((double)rand()/RAND_MAX) * 2) - 1;
    if(new_w1 < 0 ) new_w1 = 0;
    else if(new_w1 > 10) new_w1 = 10;
    this->_weights[0] = new_w1;
  }
  
  if((double)rand()/RAND_MAX < prob_mut_w2){
    double new_w2 = this->_weights[1] + (((double)rand()/RAND_MAX) * 2) - 1;
    if(new_w2 < 0) new_w2 = 0;
    else if(new_w2 > 10) new_w2 = 10;
    this->_weights[1] = new_w2;
  }
  
  if((double)rand()/RAND_MAX < prob_mut_w3){
    double new_w3 = this->_weights[2] + (((double)rand()/RAND_MAX) * 2) - 1;
    if(new_w3 < 0) new_w3 = 0;
    else if(new_w3 > 10) new_w3 = 10;
    this->_weights[2] = new_w3;
  }
  
  if((double)rand()/RAND_MAX < prob_mut_threshold){
    double new_threshold = this->_blobs_size_threshold * (0.5 + ((double)rand()/RAND_MAX));	// [min 0.5, max 1.5] times the current threshold
    this->_blobs_size_threshold = new_threshold;
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
  
  double blobs_theshold;
  if( rand() > RAND_MAX/2){
    blobs_theshold = bl1->getBlobsThreshold();
  }
  else{
    blobs_theshold = bl2->getBlobsThreshold();
  }
  
  BlobsLearner * ret = new BlobsLearner(filter_size, weights[0], weights[1], weights[2], blobs_theshold);
  return ret;
}

// <<<<<<<<<<<<<<<>>>>>>>>>>>>>>>


unsigned int chooseFromList(double * probs, unsigned int how_many){
  double draw = (double)rand()/RAND_MAX;
  bool chosen = false;
  unsigned int i = 0;
  while(!chosen){
    if(i==how_many){
      i = 0;
    }
    if(draw < probs[i]){
      chosen = true;
    }
    else{
      draw = draw - probs[i];
      i++;
    }
  }
  
  return i;
}


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
  if(_filter_size==0) _filter_size = 1;
  
  _filter = new double[_filter_size];
  for(unsigned int i = 0; i<_filter_size; i++){
    _filter[i] = 1 - i/_filter_size;
  }
}


void BlobsLearner::setParameters(unsigned int filter_size, double w1, double w2, double w3, double blobs_size_threshold){
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
  
  // set the blobs size threshold
  this->_blobs_size_threshold = blobs_size_threshold;
}


BlobsLearner::BlobsLearner(){
  setParameters(50, 1, 1, 1, 10);
}


BlobsLearner::BlobsLearner(unsigned int filter_size, double w1, double w2, double w3, double blobs_size_threshold){
  setParameters(filter_size, w1, w2, w3, blobs_size_threshold);
}


void BlobsLearner::addObservation(cv::Vec3b hsv){
  this->_samples.push_back(hsv);
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


// BUBBLESORT
// rearranges the two lists according to the fitness value
// fitness is supposed to be "better" for smaller values (0 is the best)
void bubbleSort(void * * elements, double * fitness, unsigned int how_many){
  for (unsigned int i=0; i<how_many; i++){
    for (int j=i-1; j>=0; j--){
      unsigned int index = (unsigned int) j;
      if(fitness[j+1] <= fitness[j]){	// less OR EQUAL in order to prefer new solutions (when the fitness is identical)
	// switch the fitness value
	double tmp_fit = fitness[index+1];
	fitness[index+1] = fitness[j];
	fitness[index] = tmp_fit;
	
	
	void * tmp_el = elements[index+1];
	elements[index+1] = elements[index];
	elements[index] = tmp_el;
      }
    }
  }
}


// COMPUTECHOOSINGLIST
// prepares the list according to which the blobs will be chosen with a probability that depends on their fitness
// Note: this assumes that the fitness list is in ascending order
void computeChoosingList(double * fitness, double * probs, unsigned int how_many){
  double bigger_fitness = fitness[how_many-1];
  
  // check special check: all the probabilities are the same
  if(fitness[0] == bigger_fitness){
    double fixed_prob = (double)1/how_many;
    for(unsigned int i=0; i<how_many; i++){
      probs[i] = fixed_prob;
    }
    return;
  }
  
  double total_probs = 0;
  for(unsigned int i=0; i<how_many; i++){
    probs[i] = (bigger_fitness - fitness[i]);
    total_probs += probs[i];
  }
  for(unsigned int i=0; i<how_many; i++){
    probs[i] = probs[i]/total_probs;
  }
}


BlobsLearner * generateRandomBlobsLearner(){
  unsigned int filter_size = ((double)rand()/RAND_MAX) * 100;
  double w1 = ((double)rand()/RAND_MAX) * 10;
  double w2 = ((double)rand()/RAND_MAX) * 10;
  double w3 = ((double)rand()/RAND_MAX) * 10;
  double bst = ((double)rand()/RAND_MAX) *100;
  
  return new BlobsLearner(filter_size, w1, w2, w3, bst);
}


std::vector<FloatCouple> BlobsLearner::extractCentroids(cv::Mat * hsv_image){
  cv::Mat * selected = this->selectPixels(hsv_image);
  std::vector<Blob*> blobs;
  getBlobs(selected, &blobs);
  std::vector<Blob*> big_blobs = blobs;
  purgeBlobs(&big_blobs, this->_blobs_size_threshold);
  
  std::vector<FloatCouple> centroids;
  getCentroids(&big_blobs, &centroids);

  
  delete selected;
  for(unsigned int i=0; i<blobs.size(); i++){
    delete blobs[i];
  }
  
  return centroids;
}


void configFromFile(std::string filename, double * weights, unsigned int * filter_size, double * blobs_size_threshold){
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
      if((textline[0].compare(std::string("blobs_size_threshold"))) == 0){
	*blobs_size_threshold = atof(textline[1].c_str());
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


void prepareTestSet(std::string filename, std::vector<std::string> * images, std::vector< std::vector<FloatCouple> > * centroids){
  FileReader fr(filename);
  if(!fr.is_open()){
    std::cout << "ERROR! Cannot open test set file " << filename << std::endl;
    return;
  }
  
  std::vector<std::string> textline;
  fr.readLine(&textline);
  if(!fr.good()){
    std::cout << "ERROR! test set file " << filename << " is empty" << std::endl;
    return;
  }
  
  unsigned int image_number = 0;
  
  bool another_image = true;
  while(another_image){
    
    std::string img_name = textline[0];
    for(unsigned int i=1; i<textline.size(); i++){
      img_name = img_name.append(textline[i]);
    }
    images->push_back(img_name);
    
    textline.clear();
    fr.readLine(&textline);
    
    std::vector<FloatCouple> float_couple_vector;
    
    while(textline[0].compare(std::string("--next--"))){
      float_couple_vector.push_back(FloatCouple(atof(textline[0].c_str()), atof(textline[1].c_str())));
      textline.clear();
      fr.readLine(&textline);
    }
    
    centroids->push_back(float_couple_vector);
    image_number++;
    
    textline.clear();
    fr.readLine(&textline);
    if(!fr.good()){	// end of file
      another_image = false;
    }
  }
}


std::vector<cv::Vec3b> prepareSamples(std::string filename){
  
  std::vector<cv::Vec3b> ret;
  
  FileReader fr(filename);
  if(!fr.is_open()){
    std::cout << "ERROR! Cannot open samples file " << filename << std::endl;
  }
  
  std::vector<std::string> textline;
  fr.readLine(&textline);
  textline.clear();	// the first line is an image file name
  fr.readLine(&textline);
  while(fr.good()){
    if(textline[0].compare(std::string("--next--"))){	// if the string is not "--next--"
      uint h = atoi(textline[2].c_str());
      uint s = atoi(textline[3].c_str());
      uint v = atoi(textline[4].c_str());
      ret.push_back(cv::Vec3b(h,s,v));
    }
    else{	// if the string is "--next--" just jump the next line
      textline.clear();
      fr.readLine(&textline);
    }
    
    textline.clear();
    fr.readLine(&textline);
  }

  return ret;
}


int main(int argc, char**argv){
  // check arguments
  if(argc < 3){
    std::cout << "Usage: WeightsFinder <samples_filename> <test_filename>" << std::endl << std::endl;
    return 0;
  }
  
  std::vector<cv::Vec3b> samples = prepareSamples(std::string(argv[1]));
  std::vector<std::string> test_names;
  std::vector<std::vector<FloatCouple> > test_coordinates;
  prepareTestSet(std::string(argv[2]), &test_names, &test_coordinates);
  for(unsigned int i=0; i<test_names.size(); i++){
    std::cout << test_names[i] << std::endl;
    for(unsigned int j=0; j<test_coordinates[i].size(); j++){
      std::cout << test_coordinates[i][j].x << " " << test_coordinates[i][j].y << std::endl;
    }
    std::cout << std::endl;
  }
  
  BlobsLearner * list[LIST_SIZE];
  double fitness[LIST_SIZE];
  double probs[LIST_SIZE];
  
  // prepare the first learners generation
  std::cout << "generating the first learners generation...";
  for(unsigned int i=0; i<LIST_SIZE; i++){
    list[i] = generateRandomBlobsLearner();
    list[i]->setSamples(samples);
  }
  std::cout << "DONE" << std::endl;
  
  for(unsigned int generation=0; generation<MAX_GENERATIONS; generation++){
    // put all the fitness values to 0
    for(unsigned int i=0; i<LIST_SIZE; i++){
      fitness[i] = 0;
    }
    
    // compute the new fitness value of every learner
    for(unsigned int img=0; img<test_names.size(); img++){
      std::cout << "opening " << test_names[img] << std::endl;
      std::cout.flush();
      cv::Mat image = cv::imread(test_names[img].c_str(), 1);
      cv::Mat hsv_image;
      cv::cvtColor(image,hsv_image, CV_BGR2HSV);
      
      double max_dist = sqrt(pow(image.cols,2)+pow(image.rows,2));
      
      std::cout << "this image has " << test_coordinates[img].size() << " POIs" << std::endl;
      
      thread_struct t_structs[LIST_SIZE];
      pthread_t threads[LIST_SIZE];
      double fitness_increment[LIST_SIZE];
      
      // prepare thread structs
      std::cout << "preparing thread structures" << std::endl;
      for(unsigned int i=0; i<LIST_SIZE; i++){
	t_structs[i].learner = list[i];
	t_structs[i].image = &hsv_image;
	t_structs[i].test_coordinates = &(test_coordinates[img]);
	t_structs[i].max_dist = max_dist;
	t_structs[i].extra = 1;
	t_structs[i].lack = 1;
	t_structs[i].put_result_here = fitness_increment+i;
	
	pthread_create(threads+i, NULL, threadFunc, t_structs+i);
      }
      
      // launch threads
      std::cout << "computing fitnesses (" << NumThreads << " threads)" << std::endl;
      for(unsigned int i=0; i<LIST_SIZE; i += NumThreads){
	unsigned int launched = 0;
	for(unsigned int t=0; t<NumThreads; t++){
	  if(i+t < LIST_SIZE){
	    launched++;
	    pthread_create(threads+(i+t), NULL, threadFunc, t_structs+(i+t));
	  }
	}
	for(unsigned int l=0; l<launched; l++){
	  pthread_join(threads[i+l], NULL);
	  fitness[i+l] += fitness_increment[i+l];
	  std::cout << "fitness[" << i+l << "] = " << fitness[i+l] << std::endl;
	}
      }
    }
    
    // now order the learners in fitness order
    std::cout << "bubble sort" << std::endl;
    bubbleSort((void**)list, fitness, LIST_SIZE);
    
    if(generation < MAX_GENERATIONS-1){
    
      // prepare the choosing list
      std::cout << "prepare choosing list" << std::endl;
      computeChoosingList(fitness, probs, LIST_SIZE);
    
      // copy the learners pointers into a side list
      std::cout << "prepare side list" << std::endl;
      BlobsLearner * side_list[LIST_SIZE];
      for(unsigned int i=0; i<LIST_SIZE; i++){
	side_list[i] = list[i];
      }
    
      // elitism: copy the best child into the next generation
      // two copies are done, because one will be kept as it is, the other will mutate
      std::cout << "elitism" << std::endl;
      list[0] = new BlobsLearner(side_list[0]->getFilterSize(), side_list[0]->getWeights()[0], side_list[0]->getWeights()[1], side_list[0]->getWeights()[2], side_list[0]->getBlobsThreshold());
      list[1] = new BlobsLearner(side_list[0]->getFilterSize(), side_list[0]->getWeights()[0], side_list[0]->getWeights()[1], side_list[0]->getWeights()[2], side_list[0]->getBlobsThreshold());
    
      // generate other children
      std::cout << "mating" << std::endl;
      for(unsigned int i=2; i<LIST_SIZE-2; i++){
	unsigned int draw1 = chooseFromList(probs, LIST_SIZE);
	unsigned int draw2 = chooseFromList(probs, LIST_SIZE);
	list[i] = mating(side_list[draw1], side_list[draw2]);
      }
    
      // generate two new random children
      std::cout << "random children" << std::endl;
      list[LIST_SIZE-2] = generateRandomBlobsLearner();
      list[LIST_SIZE-1] = generateRandomBlobsLearner();
    
      // set the samples for all the learners
      std::cout << "set samples" << std::endl;
      for(unsigned int i=0; i<LIST_SIZE; i++){
	list[i]->setSamples(samples);
      }
    
      // mutations (except for the copied one)
      std::cout << "mutating" << std::endl;
      for(unsigned int i=1; i<LIST_SIZE; i++){
	list[i]->mutate(PROB_MUT_FILTER, PROB_MUT_W1, PROB_MUT_W2, PROB_MUT_W3, PROB_MUT_THRESHOLD);
      }
    
      // delete the old generation
      std::cout << "deleting old generation" << std::endl;
      for(unsigned int i=0; i<LIST_SIZE; i++){
	delete side_list[i];
      }
    }
    std::cout << "at the end of generation " << generation << " the best fitness value is " << fitness[0] << std::endl;
    std::cout << "parameters of the best one: FSIZE = " << list[0]->getFilterSize() << "\tweights = <" << list[0]->getWeights()[0] << ", " << list[0]->getWeights()[1] << ", " << list[0]->getWeights()[2] << ">\tBTHRESH = " << list[0]->getBlobsThreshold() << std::endl;
  }

  // report the winner configs
  std::cout << "the configurations of the winner are:" << std::endl;
  std::cout << "filter size = " << list[0]->getFilterSize() << std::endl;
  std::cout << "weights = <" << list[0]->getWeights()[0] << ", " << list[0]->getWeights()[1] << ", " << list[0]->getWeights()[2] << ">" << std::endl;
  std::cout << "blobs size threshold = " << list[0]->getBlobsThreshold() << std::endl;

  return 0;
}
