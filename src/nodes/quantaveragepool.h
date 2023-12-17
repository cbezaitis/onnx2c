/* This file is part of onnx2c.
 *
 * QuantAvgPool2d
 * Calculates average from the elements that
 * are under the kernel.
 * But it is from QOONX, so it is quantized
 *
 */
#include <cmath>
#include "pooling.h"
namespace toC {

class QuantAvgPool2d : public Pooling {
	public:
	QuantAvgPool2d() : Pooling() {
		op_name = "QuantAvgPool2d";
	}
	std::string data_layout = "empty";
	int64_t ibits = 0;
	int64_t obits = 0;
	void parseAttributes( onnx::NodeProto &node ) override {
		LOG(DEBUG) << "parse first" <<std::endl;
		for( const auto& a : node.attribute() ) {
			if( a.name() == "data_layout" )
				data_layout = parse_attribute_string(a);
			if( a.name() == "ibits" )
				ibits = parse_attribute_int(a);
			if( a.name() == "kernel" )
			{
				// input as one atrribute on the QONNX standard
				kernel_shape.push_back(a.i());
				kernel_shape.push_back(a.i());
			}
			if( a.name() == "obits" )
				obits = parse_attribute_int(a);
			if( a.name() == "stride" )
			{
				// input as one atrribute on the QONNX standard
				strides.push_back(a.i());
				strides.push_back(a.i());
			}
			if( a.name() == "data_layout" )
				data_layout = parse_attribute_string(a);
		}
	}
	virtual void print_output_cell_init(std::ostream &dst, const std::string &y_idx) const override
	{
		INDT_3  << get_Y()->data_type_str() << " curavg = 0.0;" << std::endl;
		INDT_3  << "int numavg = 0;" << std::endl;
	}
	virtual void print_output_cell_calc(
		std::ostream &dst,
		const std::string &x_idx,
		const std::string &w_idx,
		const std::string &y_idx) const override
	{
		// Sum up the cells
		INDT_4 << "numavg += 1;" <<std::endl;
		INDT_4 << "curavg += x" << x_idx << ";" <<std::endl;
	}
	virtual void print_output_cell_finalize(std::ostream &dst, const std::string &y_idx) const override
	{
		// Calculate the averageing part
		if( count_include_pad ) {
			int numavg=1;
			for ( auto k : kernel_shape )
				numavg *= k;
			INDT_3 << "/* Counting padding into the average is requested */" << std::endl;
			INDT_3 << "numavg = " << numavg << ";" << std::endl;
		}
		INDT_3 << "y" << y_idx << "= curavg/numavg;" << std::endl;
	}


	virtual void print(std::ostream &dst) const override
	{
		// Regular not quantized average pooling
		print_header_info_comment(dst);
		print_loop_with_padding_checks(dst);

		// remove scaling introduced by average
		// Get Accumulator Size
		int64_t max_value = (std::pow(2,ibits) - 1) * kernel_shape[0] * kernel_shape[0];
		int64_t bit_length = static_cast<int>(std::log2(max_value) + 1);
		// Get number of Shifts
		int64_t shifts = bit_length - obits;
		if (shifts <= 0)
		{
			shifts = 0;
		}
		
		// Shift the result
		Tensor *C = outputs[0];
		INDT_1 << "for( uint32_t chan=0; chan<" << C->data_dim[1] << "; chan++) " << std::endl;
		INDT_2 << "for( uint32_t r=0; r<" << C->data_dim[2] << "; r++)" << std::endl;
		INDT_3 << "for( uint32_t c=0; c<" << C->data_dim[3] << "; c++) " << std::endl;
		INDT_4 << "y[0][chan][r][c] = ( (uint32_t) ( y[0][chan][r][c] * "<< kernel_shape[0]*kernel_shape[0] <<")) >> "<< shifts<<";" << std::endl;
	}
 
	virtual void resolve(void) override
	{
		register_input(inputs[0], "x");
		resolve_strides();
		resolve_dilations();
		resolve_pads();
		resolve_kernel_shape();
		Tensor *rv = new Tensor;
		rv->data_dim = resolve_output_size();
		rv->data_type = get_X()->data_type;
		register_output(rv, "y");

		update_pads();

		// optional indices vector
		Tensor *indices_out = new Tensor;
		indices_out->data_type = onnx::TensorProto_DataType::TensorProto_DataType_INT64;
		indices_out->data_dim = rv->data_dim;
		register_output(indices_out, "ind");
	}
};
}
