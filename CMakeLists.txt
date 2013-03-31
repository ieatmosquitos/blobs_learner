project( BlobsLearner )

cmake_minimum_required(VERSION 2.6)

include_directories( ${BlobsLearner_SOURCE_DIR}/include )
find_package( OpenCV REQUIRED )
find_package (Threads)
add_executable( BlobsLearner src/BlobsLearner.cpp src/FileReader.cpp)
add_executable( BlobsLearner2 src/BlobsLearner2.cpp src/FileReader.cpp)
add_executable( WeightsFinder src/WeightsFinder.cpp src/FileReader.cpp)
add_executable( SamplesGathering src/gather.cpp)
target_link_libraries( BlobsLearner ${OpenCV_LIBS})
target_link_libraries( BlobsLearner2 ${OpenCV_LIBS})
target_link_libraries( WeightsFinder ${OpenCV_LIBS} ${CMAKE_THREAD_LIBS_INIT} )
target_link_libraries( SamplesGathering ${OpenCV_LIBS})

execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/images
    ${CMAKE_BINARY_DIR}/images
)

execute_process(COMMAND ${CMAKE_COMMAND} -E copy_directory
    ${CMAKE_SOURCE_DIR}/misc
    ${CMAKE_BINARY_DIR}
)