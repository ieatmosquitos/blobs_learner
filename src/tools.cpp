#include <vector>

#ifndef BLUE_CHANNEL
#define BLUE_CHANNEL 0
#endif

#ifndef GREEN_CHANNEL
#define GREEN_CHANNEL 1
#endif

#ifndef RED_CHANNEL
#define RED_CHANNEL 2
#endif

#define CROSS_CHANNEL GREEN_CHANNEL // change this to change the color of the crosses drawn over the centroids

// Structure created to identify an <x,y> couple, used for storing pixels positions
struct Coordinate{
  int x;
  int y;
  
  Coordinate(int x, int y){
    this->x = x;
    this->y = y;
  }
};


// Structure similar to Coordinate, but uses floats (used for the centroids)
struct FloatCouple{
  float x;
  float y;
  
  FloatCouple(){
    x = 0;
    y = 0;
  }
  
  FloatCouple(float x, float y){
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


/*!
 * getCentroid computes the centroid of the given Coordinate vector
 * \param points the vector listing all the points
 */
FloatCouple getCentroid(std::vector<Coordinate> * points){
  FloatCouple * c = new FloatCouple(0,0);
  int points_number = points->size();
  for(int i=0; i<points_number; i++){
    c->x += (*points)[i].x;
    c->y += (*points)[i].y;
  }
  c->x = c->x / points_number;
  c->y = c->y / points_number;
        
  return *c;
}
    
/*!
 * getCentroids computes the centroids for each of the blob in the given array
 * \param blobs pointer to the vector listing the blobs
 * \param centroids the vector where to insert the centroids
 */
void getCentroids(std::vector<Blob *> * blobs, std::vector<FloatCouple> * centroids){
  for(unsigned int i=0; i<blobs->size(); i++){     // for each blob
    FloatCouple cent = getCentroid(&((*blobs)[i]->points));
    centroids->push_back(cent);
  }
}


/*!
 * drawCross draws a cross of the given size (size = half width) on the image at the given position
 * \param image pointer to the image
 * \param fc pointer to the couple with the position
 * \param crossSize size of the cross
 */
void drawCross(cv::Mat * image, FloatCouple * fc, int crossSize){
  Coordinate c((int)fc->x, (int)fc->y);
    
  cv::Vec3b black_pix(0,0,0);
  // image->at<cv::Vec3b>(c.y, c.x)[RED_CHANNEL]=(uchar)0;
  // image->at<cv::Vec3b>(c.y, c.x)[BLUE_CHANNEL]=(uchar)0;
  // image->at<cv::Vec3b>(c.y, c.x)[GREEN_CHANNEL]=(uchar)0;
  image->at<cv::Vec3b>(c.y,c.x)=black_pix;
  image->at<cv::Vec3b>(c.y, c.x)[CROSS_CHANNEL]=(uchar)255;
  for(int offset = 1; offset<crossSize; offset++){
    Coordinate point(c.x-offset, c.y-offset);
    if((point.x>=0) && (point.x<image->cols) && (point.y>=0) && (point.y<image->rows)){
      image->at<cv::Vec3b>(point.y, point.x)=black_pix;
      image->at<cv::Vec3b>(point.y, point.x)[CROSS_CHANNEL]=(uchar)255;
    }

    point.x = c.x+offset;
    point.y = c.y-offset;
    if((point.x>=0) && (point.x<image->cols) && (point.y>=0) && (point.y<image->rows)){
      image->at<cv::Vec3b>(point.y, point.x)=black_pix;
      image->at<cv::Vec3b>(point.y, point.x)[CROSS_CHANNEL]=(uchar)255;
    }

    point.x = c.x-offset;
    point.y = c.y+offset;
    if((point.x>=0) && (point.x<image->cols) && (point.y>=0) && (point.y<image->rows)){
      image->at<cv::Vec3b>(point.y, point.x)=black_pix;
      image->at<cv::Vec3b>(point.y, point.x)[CROSS_CHANNEL]=(uchar)255;
    }

    point.x = c.x+offset;
    point.y = c.y+offset;
    if((point.x>=0) && (point.x<image->cols) && (point.y>=0) && (point.y<image->rows)){
      image->at<cv::Vec3b>(point.y, point.x)=black_pix;
      image->at<cv::Vec3b>(point.y, point.x)[CROSS_CHANNEL]=(uchar)255;
    }
  }
}


/*!
 * centroidsPainter draws crosses in correspondance of the centroids in the given image
 * \param image pointer to the image where to draw the crosses
 * \param centroids vector storing the centroids coordinates
 */
void centroidsPainter(cv::Mat * image, std::vector<FloatCouple> * centroids){
  for(unsigned int i=0; i<centroids->size(); i++){ // for each centroid
    FloatCouple * c = &((*centroids)[i]);
    drawCross(image, c, image->rows/20);
  }
}


// // returns true if the given vector contains the given element
// template <class T> bool contains(std::vector<T> vec, T element){
//   for(unsigned int i=0; i<vec.size(); i++){
//     if(vec[i] == element) return true;
//   }
  
//   return false;
// }



// returns the index of the smaller element in an array that is not in the blacklist.
// If all the elements are in the blacklist, returns array_size, that is the first out of bound index.
// blacklist is supposed to be the same size of the array, and a true in blacklist means that that element must not be considered.
unsigned int nextMin(double * the_array, bool * blacklist, unsigned int array_size){
  
  unsigned int min_index = array_size;
  
  // first initialization, just look for the first non-blacklisted element
  for(unsigned int i = 0; i<array_size; i++){
    if(!blacklist[i]){
      min_index = i;
      break;
    }
  }
  
  // check remaining elements
  for(unsigned int i=min_index; i<array_size; i++){
    if((!blacklist[i]) && (the_array[i] < the_array[min_index])){
      min_index = i;
    }
  }
  
  return min_index;
}


// COMPUTEMAPSDISTANCE
// computes the "distance" between two maps. A map is a set of float couples, which indicates the POI coordinates.
// The distance may be not simmetric, due to the weights given to the missing or extra landmarks.
// The maps are supposed to be extracted from the same image, so the user is asked to give the MAXIMUM DISTANCE, which is the scale factor used to shrink all the found distances (the maximum distance should be the diagonal of the image used to detect the POIs).
// inputs:
//	• map1 - first map
//	• map2 - second map
//	• max_dist - the maximum distance (two landmarks at this distance will be consodered at distance 1)
double computeMapsDistance(std::vector<FloatCouple> map1, std::vector<FloatCouple> map2, double max_dist, double extra_POI_multiplier, double lacking_POI_multiplier){
  
  double distances[map1.size()][map2.size()];	// distances[i][j] will store the distance between the i-th point in the first map and the j-th point in the second map
  for(unsigned int m1=0; m1<map1.size(); m1++){
    for(unsigned int m2=0; m2<map2.size(); m2++){
      distances[m1][m2] = sqrt(pow((double)(map1[m1].x - map2[m2].x),2) + pow((double)(map1[m1].y - map2[m2].y),2)) / max_dist;
    }
  }
  
  bool to_be_checked[map1.size()];	// will tell which points of the first map are to be further checked
  for(unsigned int i=0; i<map1.size(); i++){
    to_be_checked[i] = true;
  }
  
  unsigned int associated_to[map2.size()];	// will tell which point in map1 has been associated to the corresponding map2 point
  for(unsigned int i=0; i<map2.size(); i++){
    associated_to[i] = map1.size();	// this index is out of bounds, so it can be used as an invalid value
  }
  
  bool checked_everything = false;
  unsigned int check_this_one = map1.size();
  while(!checked_everything){
    check_this_one = map1.size();
    for(unsigned int i=0; i<map1.size(); i++){
      if (to_be_checked[i]){
	check_this_one = i;
      }
    }
    
    if(check_this_one == map1.size()){
      // everything has been checked
      checked_everything = true;
    }
    
    if(!checked_everything){
      to_be_checked[check_this_one] = false;
      
      // look for te closer map2 point that is not associated to a closer point in map1
      bool blacklist[map2.size()];
      for(unsigned int i=0; i<map2.size(); i++){
	blacklist[i] = false;
      }
      bool found_an_association = false;
      unsigned int p2_index = map2.size();
      while(!found_an_association){
	p2_index = nextMin(distances[check_this_one], blacklist, map2.size());
	if(p2_index < map2.size()){
	  if(associated_to[p2_index] == map1.size() || distances[check_this_one][p2_index] < distances[associated_to[p2_index]][p2_index]){
	    if(associated_to[p2_index] == map1.size()) to_be_checked[associated_to[p2_index]] = true;
	    associated_to[p2_index] = check_this_one;
	    found_an_association = true;
	  }
	  else{
	    blacklist[p2_index] = true;
	  }
	}
	else{	// all the map2 points have been associated to some other map1 point. This one is an extra point.
	  break;	// this quits the while cicle
	}
      }
      
    }
    else{	// checked_everything is true
      
    }
  }

  // compute the distance
  double dist = 0;
  if(map2.size() > map1.size()){
    dist += (map2.size()-map1.size()) * lacking_POI_multiplier;
  }
  
  if(map1.size() > map2.size()){
    dist += (map1.size()-map2.size()) * extra_POI_multiplier;
  }
  
  for(unsigned int i=0; i<map2.size(); i++){
    if(associated_to[i] != map1.size()){
      dist += distances[associated_to[i]][i];
    }
  }
  
  return dist;
}
