/*
 * File:   tally.h
 * Author: Lennart
 *
 * Created on 6. January 2020, 18:00
 */

/*
 * Creates a Tally of all functions calls of the 7 different primitives.
 * This infromation can later be used to create better approximations.
 *
 * Future Work?
 * 	differentiate between different simd types
 * 	work in Granularity? so 256<64>, 256<32> etc.
 *
*/

/*
#if tally
io_simd = io_simd + 1;
#endif
*/
#ifndef TALLY_H
#define TALLY_H

#include <iostream>

uint64_t calc_simd = 0;
uint64_t calc_scalar = 0;

uint64_t compare_simd = 0;
uint64_t compare_scalar = 0;

uint64_t create_simd = 0;
uint64_t create_scalar = 0;

uint64_t extract_simd = 0;
uint64_t extract_scalar = 0;

uint64_t io_simd = 0;
uint64_t io_scalar = 0;

uint64_t logic_simd = 0;
uint64_t logic_scalar = 0;

uint64_t manipulate_simd = 0;
uint64_t manipulate_scalar = 0;

void output_tally(){
std::cout << "type\tcalc\tcompare\tcreate\textract\tio\tlogic\tmanipulate\n";
std::cout << "simd\t" << calc_simd << "\t" << compare_simd
			<< "\t" << create_simd << "\t" << extract_simd << "\t" << io_simd
			<< "\t" << logic_simd << "\t" << manipulate_simd << std::endl;
std::cout << "scalar\t" << calc_scalar << "\t" << compare_scalar
			<< "\t" << create_scalar << "\t" << extract_scalar << "\t" << io_scalar
			<< "\t" << logic_scalar << "\t" << manipulate_scalar << std::endl;
}

void reset_tally(){
	calc_simd = 0;
	calc_scalar = 0;
	compare_simd = 0;
	compare_scalar = 0;
	create_simd = 0;
	create_scalar = 0;
	extract_simd = 0;
	extract_scalar = 0;
	io_simd = 0;
	io_scalar = 0;
	logic_simd = 0;
	logic_scalar = 0;
	manipulate_simd = 0;
	manipulate_scalar = 0;
}
#endif
