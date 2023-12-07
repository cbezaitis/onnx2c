/* This file is part of onnx2c.
 *
 * Custom Operation of QONNX
 * It is a quantization step based on thresholding 
 * the input values. 
 * It only outputs integers representated by float numbers 
 * for the time being
 * 
 */
#include "node.h"

namespace toC {

class MultiThreshold : public Node {
	public:
	MultiThreshold() {
		op_name = "MultiThreshold";
	}
	/* Examples of ONNX Operand attributes */
	float out_bias;
	std::string out_dtype;
	float out_scale = 0.0;
	std::string data_layout = "empty";

	// Mandatory "API" functions towards the rest of onnx2c
	virtual void parseAttributes( onnx::NodeProto &node ) override;
	virtual void resolve(void) override;
	virtual void print(std::ostream &dst) const override;
	// Extra helper functions
	virtual void print4D(std::ostream &dst) const;
	virtual void print4DLayout(std::ostream &dst) const;
	virtual void print2D(std::ostream &dst) const;
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
		else if( a.name() == "data_layout" )
			data_layout = parse_attribute_string(a);
		else if( a.name() == "out_scale" )
			out_scale = parse_attribute_float(a);
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

	t->data_type = onnx::TensorProto_DataType_FLOAT;
	register_output(t, "Y");

	/* TODO: optional outputs? */
}

void MultiThreshold::print4D(std::ostream &dst) const
{
	Tensor *A = inputs[0];
	Tensor *B = inputs[1];
	INDT_1 << "/* MultiThreshold 4D Layout need Transpose */" << std::endl;
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
	if (out_scale != 0)
	{
		INDT_4 << "Y[0][chan][r][c] = "<< out_scale << " * reshape_output[chan][index] + ("<< out_bias<<");" << std::endl;
	} else
	{
		INDT_4 << "Y[0][chan][r][c] = reshape_output[chan][index] + ("<< out_bias<<");" << std::endl;
	}
	INDT_4 << "index++;"<< std::endl;
	INDT_3 << "}"<< std::endl;
	INDT_1 << "}"<< std::endl;
}

void MultiThreshold::print4DLayout(std::ostream &dst) const
{
	Tensor *A = inputs[0];
	Tensor *B = inputs[1];
	INDT_1 << "/* MultiThreshold 4D Immedieate Layout*/" << std::endl;
	int32_t rows = A->data_dim[2];
	int32_t cols = A->data_dim[3];
	int32_t batch = A->data_dim[0];
	if (batch != 1)
	{
		ERROR("batch not implemented");
	}
	int32_t channels = A->data_dim[1];
	int32_t thresholds = B->data_dim[1];
	int32_t channelsInput = B->data_dim[0];
	if ( channelsInput != cols)
	{
		ERROR("colums of input should be the same as threshold");
	}
	
	INDT_1 << "/* Thresholding */" << std::endl;
	INDT_1 << "for( uint32_t chan=0; chan<" << channels << "; chan++) " << std::endl;
	INDT_2 << "for( uint32_t r=0; r<" << rows << "; r++)" << std::endl;
	INDT_3 << "for( uint32_t c=0; c<" << cols << "; c++) {" << std::endl;
	INDT_4 << "Y[0][chan][r][c]= 0;" << std::endl;
	INDT_4 << "for( uint32_t thresh=0; thresh<" << thresholds << "; thresh++)" << std::endl;
	INDT_5 << "if (A[0][chan][r][c] >= B[c][thresh])"<< std::endl;
	INDT(6)<< "Y[0][chan][r][c] = Y[0][chan][r][c] + 1;"<< std::endl;
	INDT_3 << "}"<< std::endl;

	INDT_1 << "\n/* Scale && Bias */" << std::endl;

	INDT_1 << "/* immediately thresholding */" << std::endl;
	INDT_1 << "for( uint32_t chan=0; chan<" << channels << "; chan++) " << std::endl;
	INDT_2 << "for( uint32_t r=0; r<" << rows << "; r++)" << std::endl;
	INDT_3 << "for( uint32_t c=0; c<" << cols << "; c++)" << std::endl;
	if (out_scale != 0)
	{
		INDT_4 << "Y[0][chan][r][c] = "<< out_scale << " * Y[0][chan][r][c] + ("<< out_bias<<");"<< std::endl;
	} else
	{
		INDT_4 << "Y[0][chan][r][c] = Y[0][chan][r][c] + ("<< out_bias<<");"<< std::endl;
	}
}

void MultiThreshold::print2D(std::ostream &dst)  const
{
	Tensor *A = inputs[0];
	Tensor *B = inputs[1];
	INDT_1 << "/* MultiThreshold 2D */" << std::endl;
	int32_t pixels = A->data_dim[1];
	int32_t batch  = A->data_dim[0];
	if (batch != 1 )
	{
		ERROR("not implememented");
	}
	int32_t threshold_pixels = B->data_dim[0];
	int32_t thresholds = B->data_dim[1];
	if ( threshold_pixels != pixels)
	{
		ERROR("input pixels and parameter pixels should be the same");
	}
	
	INDT_1 << "/* Do the actual thresholding */" << std::endl;
	INDT_1 << "for( uint32_t pixel=0; pixel<" << pixels << "; pixel++){" << std::endl;
	INDT_2 << "Y[0][pixel] = 0;" << std::endl;
	INDT_2 << "for( uint32_t threshold=0; threshold<" << thresholds << "; threshold++)" << std::endl;
	INDT_3 << "if(A[0][pixel]>=B[pixel][threshold])" << std::endl;
	INDT_4 << "Y[0][pixel]++;" << std::endl;
	INDT_1 << "}" << std::endl;
	INDT_1 << "\n/* Scale && Bias */" << std::endl;
	INDT_1 << "for( uint32_t pixel=0; pixel<" << pixels << "; pixel++)" << std::endl;

	if (out_scale != 0)
	{
		INDT_2 << "Y[0][pixel] = " << out_scale << " * Y[0][pixel] + ("<< out_bias<<"); " << std::endl;
	} else
	{
		INDT_2 << "Y[0][pixel] = Y[0][pixel] + ("<< out_bias<<"); " << std::endl;
	}
}

/* Body of the node implementing function */
void MultiThreshold::print(std::ostream &dst) const
{
	if (inputs[0]->data_dim.size() == 4 )
	{
		if (data_layout.compare("NHWC")== 0)
		{
			print4DLayout(dst);
		} else
		{
			print4D(dst);
		}
		
		
	} else if(inputs[0]->data_dim.size() == 2 )
	{
		print2D(dst);
	} else
	{
		ERROR("Not implemented for anything different than 4 dimensions");
	}
	



}


} // namespace

