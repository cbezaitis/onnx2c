/* This file is part of onnx2c.
 *
 * TEMPLATE node.
 * When implementing a new node, use this template
 * as a starting point.
 *
 * This file can be kept as a single .h file with an
 * in-header implementation, or it can be split into
 * a .h and a .cc file.
 *
 * Replace all occurances of TEMPLATE in this file.
 * Some representative dummy implementation provided.
 *
 * The functions here are callbacks from the onnx2c
 * framework. See node.h for more documentation.
 */
#include "node.h"

namespace toC {

class MultiThreshold : public Node {
	public:
	MultiThreshold() {
		op_name = "MultiThreshold";
	}
	/* Examples of ONNX Operand attributes */
	// std::vector<float> a_floatarray_attribute;
	float out_bias;
	std::string out_dtype;

	// Mandatory "API" functions towards the rest of onnx2c
	virtual void parseAttributes( onnx::NodeProto &node ) override;
	virtual void resolve(void) override;
	virtual void print(std::ostream &dst) const override;
};


/* Parse attributes, if this node has them. */
void MultiThreshold::parseAttributes( onnx::NodeProto &node )
{
	for( const auto& a : node.attribute() ) {
		LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;
		if( a.name() == "out_bias" )
			out_bias = parse_attribute_float(a);
		else if( a.name() == "out_dtype" )
			out_dtype = parse_attribute_string(a);
		else
			LOG(ERROR) << "Ignoring attribute " << a.name() << " for node MultiThreshold/" << onnx_name << std::endl;
	}
}


/* Assign input tensors, resolve output tensor shapes, allocate output tensors */
void MultiThreshold::resolve(void)
{
	Tensor *input_1  = inputs[0];
	Tensor *B = inputs[1];
	// Remember the parameters to the generated function,
	// along with a descriptive name that is used locally in the generated source.
	// The most "descriptive name" usually is the one this tensor has in the ONNX documentation.
	register_input(input_1, "A");
	register_input(B, "B");


	/* Create output tensors.
	 * Set data dimensions and data type for the created tensors. */
	Tensor *t = new Tensor;
	// Same data dimensions for input and output
	t->data_dim = inputs[0]->data_dim;

	// Multithreshold should turn this to an integer,
	// but for simulation purposes this is now a 
	if ( inputs[0]->data_dim.size() != 4)
	{
		ERROR("Not implemented for anything different than 4 dimensions");
	}
	
	t->data_type = onnx::TensorProto_DataType_FLOAT;
	register_output(t, "Y");

	/* TODO: optional outputs? */
}


/* Body of the node implementing function */
void MultiThreshold::print(std::ostream &dst) const
{
	Tensor *A = inputs[0];
	Tensor *B = inputs[1];
	INDT_1 << "/* MultiThreshold */" << std::endl;
	INDT_1 << "/* Print info on this node here, for debugging purposes */" << std::endl;
	int32_t rows = A->data_dim[2];
	int32_t cols = A->data_dim[3];
	// int32_t batch = A->data_dim[0];
	int32_t channels = A->data_dim[1];
	int32_t thresholds = B->data_dim[1];
	int32_t channelsInput = B->data_dim[0];
	if ( channelsInput != channels)
	{
		ERROR("Channels of input and and threshold are different");
	}
	
	INDT_1 << "/* First Reshape */" << std::endl;
	INDT_1 << "float reshape_input["<< channels << "][" << rows*cols << "];"<<std::endl;
	INDT_1 << "for( uint32_t chan=0; chan<" << channels << "; chan++) {" << std::endl;
	INDT_2 << "uint32_t index = 0;" << std::endl;
	INDT_2 << "for( uint32_t r=0; r<" << rows << "; r++)" << std::endl;
	INDT_3 << "for( uint32_t c=0; c<" << cols << "; c++) {" << std::endl;
	INDT_4 << "reshape_input[chan][index]=A[0][chan][r][c];" << std::endl;
	INDT_4 << "index++;"<< std::endl;
	INDT_3 << "}"<< std::endl;
	INDT_1 << "}"<< std::endl;
	INDT_1 << "/* Initialize intermediate reshape */" << std::endl;
	INDT_1 << "float reshape_output["<< channels << "][" << rows*cols << "];"<<std::endl;
	INDT_1 << "for( uint32_t chan=0; chan<" << channels << "; chan++) " << std::endl;
	INDT_2 << "for( uint32_t pixel=0; pixel<" << rows*cols << "; pixel++)" << std::endl;
	INDT_3 << "reshape_output[chan][pixel]= 0;"<<std::endl;
	INDT_1 << "/* Do the actual thresholding */" << std::endl;
	INDT_1 << "for( uint32_t chan=0; chan<" << channels << "; chan++)" << std::endl;
	INDT_2 << "for( uint32_t threshold=0; threshold<" << thresholds << "; threshold++)" << std::endl;
	INDT_3 << "for( uint32_t pixel=0; pixel<" << rows*cols << "; pixel++)" << std::endl;
	INDT_4 << "if(reshape_input[chan][pixel]>=B[chan][threshold])" << std::endl;
	INDT_5 << "reshape_output[chan][pixel]++;" << std::endl;
	INDT_1 << "/* Final Reshape For Output*/" << std::endl;
	INDT_1 << "for( uint32_t chan=0; chan<" << channels << "; chan++) {" << std::endl;
	INDT_2 << "uint32_t index = 0;" << std::endl;
	INDT_2 << "for( uint32_t r=0; r<" << rows << "; r++)" << std::endl;
	INDT_3 << "for( uint32_t c=0; c<" << cols << "; c++) {" << std::endl;
	INDT_4 << "Y[0][chan][r][c] = reshape_output[chan][index] + ("<< out_bias<<");" << std::endl;
	INDT_4 << "index++;"<< std::endl;
	INDT_3 << "}"<< std::endl;
	INDT_1 << "}"<< std::endl;


}


} // namespace

