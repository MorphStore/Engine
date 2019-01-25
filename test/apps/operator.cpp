/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

#include <iostream>
#include "core/storage/storage_container_uncompressed.h"

int main( void ) {

    std::cout << "Test filter started" << std::endl;
    
    int count_results=0;
    storage_container_uncompressed<int> * myStore=new storastorage_container_uncompressed<int>();
    for (int i=0;i<myStore->count_values();i++){
        if (myStore->data[i]<100) count_results++;
    }
    
    std::cout << "Found " << count_results << " values" << std::endl;
    std::cout << "Test filter finished" << std::endl;
   return 0;
}