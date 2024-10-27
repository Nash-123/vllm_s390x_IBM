#include "cpu_types_vxe.hpp"
#include <iostream>
#include <cmath>
#include <cassert>
#include <vecintrin.h>
#include <iomanip>
using vec_op::BF16Vec8;

#define vec_neg(a) (-(a))
#define vec_add(a, b) ((a) + (b))
#define vec_sub(a, b) ((a) - (b))
#define vec_mul(a, b) ((a) * (b))
#define vec_div(a, b) ((a) / (b))
#define vec_sr(a, b) ((a) >> (b))


void test_vec_add() {
    __vector float a = vec_splats(1.0f);
    __vector float b = vec_splats(2.0f);
    __vector float expected = vec_splats(3.0f);
    __vector float result = vec_add(a, b);
    // Logging the result
    std::cout << "Testing vec_add: ";
    if (result[0] == expected[0]) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
    }
}


void test_vec_neg() {
    __vector float a = vec_splats(-1.0f);
    __vector float expected = vec_splats(1.0f);
    __vector float result = vec_neg(a);
    std::cout << "Testing vec_neg: ";
    if (result[0] == expected[0]) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
    }
}

void test_vec_sub() {
    __vector float a = vec_splats(3.0f);
    __vector float b = vec_splats(1.0f);
    __vector float expected = vec_splats(2.0f);
    __vector float result = vec_sub(a, b);
    std::cout << "Testing vec_sub: ";
    if (result[0] == expected[0]) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
    }
}

void test_vec_mul() {
    __vector float a = vec_splats(2.0f);
    __vector float b = vec_splats(3.0f);
    __vector float expected = vec_splats(6.0f);
    __vector float result = vec_mul(a, b);
    std::cout << "Testing vec_mul: ";
    if (result[0] == expected[0]) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
    }
}

void test_vec_div() {
    __vector float a = vec_splats(6.0f);
    __vector float b = vec_splats(2.0f);
    __vector float expected = vec_splats(3.0f);
    __vector float result = vec_div(a, b);
    std::cout << "Testing vec_div: ";
    if (result[0] == expected[0]) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
    }
}

void test_vec_sr() {
    __vector int a = vec_splats(8);
    int shift = 1;
    __vector int expected = vec_splats(4);
    __vector int result = vec_sr(a, shift);
    std::cout << "Testing vec_sr: ";
    if (result[0] == expected[0]) {
        std::cout << "PASS" << std::endl;
    } else {
        std::cout << "FAIL" << std::endl;
    }
}


void detailed_logging(const float* data, const char* label, int size) {
    std::cout << label << ": ";
    for (int i = 0; i < size; i++) {
        std::cout << std::fixed << std::setprecision(7) << data[i] << " ";
    }
    std::cout << std::endl;
}

void test_BF16Vec8_save1() {
    alignas(16) float input_data[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0}; // Ensure alignment
    BF16Vec8 vec(input_data);
    alignas(16) float output_data[8]; // Ensure alignment
    vec.save(output_data);

    detailed_logging(input_data, "Input", 8);
    detailed_logging(output_data, "Output", 8);
}

void test_BF16Vec8_operations() {
    alignas(16) float input_data[8] = {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0};
    BF16Vec8 vec(input_data);
    alignas(16) float output_data[8];
    vec.save(output_data);

    std::cout << "Input vs Output:\n";
    for (int i = 0; i < 8; i++) {
        std::cout << "Input[" << i << "]: " << std::fixed << std::setprecision(7) << input_data[i]
                  << " Output[" << i << "]: " << output_data[i] << "\n";
    }
}

void test_BF16Vec8_save() {
   
    bool pass;

    float input_data[8] = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0}; // Test with zeroes
    BF16Vec8 vec(input_data);
    float output_data[8];
    vec.save(output_data);
    pass = true;
    std::cout << "Input vs Output:" << std::endl;
    for (int i = 0; i < 8; i++) {
	std::cout << "Input[" << i << "]: " << input_data[i] << " Output[" << i << "]: " << output_data[i] << std::endl;
        if (input_data[i] != output_data[i]) {
            pass = false;
            
        }
    }
    std::cout << "Testing BF16Vec8 save with zeroes: " << (pass ? "PASS" : "FAIL") << std::endl;

    float input_data_ones[8] = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0}; // Test with ones
    BF16Vec8 vec_ones(input_data_ones);
    float output_data_ones[8];
    vec_ones.save(output_data_ones);
    pass = true;
    std::cout << "Input vs Output:" << std::endl;
    for (int i = 0; i < 8; i++) {
	std::cout << "Input[" << i << "]: " << input_data_ones[i] << " Output[" << i << "]: " << output_data_ones[i] << std::endl;
        if (input_data_ones[i] != output_data_ones[i]) {
            pass = false;
            
        }
    }
    std::cout << "Testing BF16Vec8 save with ones: " << (pass ? "PASS" : "FAIL") << std::endl;

    float input_data_small[8] = {0.5, -0.5, 0.25, -0.25, 0.75, -0.75, 0.1, -0.1}; // Test with alternating small values
    BF16Vec8 vec_small(input_data_small);
    float output_data_small[8];
    vec_small.save(output_data_small);
    pass = true;
    std::cout << "Input vs Output:" << std::endl;
    for (int i = 0; i < 8; i++) {
	std::cout << "Input[" << i << "]: " << input_data_small[i] << " Output[" << i << "]: " << output_data_small[i] << std::endl;
        if (input_data_small[i] != output_data_small[i]) {
            pass = false;
           
        }
    }
    std::cout << "Testing BF16Vec8 save with small values: " << (pass ? "PASS" : "FAIL") << std::endl;
    
    float  input_data_one_8[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    BF16Vec8 vec_one_8(input_data_one_8);
    float output_data_one_8[8];
    vec_one_8.save(output_data_one_8);
    pass = true;
    std::cout << "Input vs Output:" << std::endl;
    for (int i = 0; i < 8; i++) {
	std::cout << "Input[" << i << "]: " << input_data_one_8[i] << " Output[" << i << "]: " << output_data_one_8[i] << std::endl;
        if (input_data_one_8[i] != output_data_one_8[i]) {
            pass = false;
        }
    }
    std::cout << "Testing BF16Vec8 save with values from 1.0-8.0: " << (pass ? "PASS" : "FAIL") << std::endl;
}

void test_BF16Vec8_conversion() {
    alignas(16) float input_data[8] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
    vec_op::FP32Vec8 fp32_vec(input_data);  // Ensure that FP32Vec8 is appropriately constructed
    vec_op::BF16Vec8 bf16_vec(fp32_vec);    // This uses the explicit BF16Vec8(const FP32Vec8&) constructor

    alignas(16) float output_data[8];
    bf16_vec.save(output_data);  // Output the converted data

    std::cout << "FP32 to BF16 Conversion Test:\n";
    for (int i = 0; i < 8; i++) {
        std::cout << "FP32 Input[" << i << "]: " << input_data[i]
                  << " -> BF16 Output[" << i << "]: " << output_data[i] << "\n";
    }
}

int main() {
  //  test_vec_add();
  //  test_vec_neg();
  //  test_vec_sub();
  //  test_vec_mul();
  //  test_vec_div();
  //  test_vec_sr();
    test_BF16Vec8_save();
    test_BF16Vec8_conversion();
    return 0;
}

