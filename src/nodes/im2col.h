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

class Im2Col : public Node {
	public:
	Im2Col() {
		op_name = "Im2Col";
	}
	/* Examples of ONNX Operand attributes */
	std::vector<int64_t> dilations;
	int depthwise;
	std::vector<int64_t> kernel_size;

	std::vector<int64_t> stride;
	std::vector<int64_t> pad_amount;

	// Mandatory "API" functions towards the rest of onnx2c
	virtual void parseAttributes( onnx::NodeProto &node ) override;
	virtual void resolve(void) override;
	virtual void print(std::ostream &dst) const override;
};


/* Parse attributes, if this node has them. */
void Im2Col::parseAttributes( onnx::NodeProto &node )
{
	for( const auto& a : node.attribute() ) {
		LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;
		if( a.name() == "depthwise" )
			depthwise = parse_attribute_int(a);
		else if( a.name() == "dilations" )
			dilations = parse_attribute_ints(a);
		else if( a.name() == "kernel_size" )
		{
			kernel_size = parse_attribute_ints(a);
		}
		else if( a.name() == "stride" )
		{
			stride  = parse_attribute_ints(a);
		}
		else if( a.name() == "pad_amount" )
		{
			pad_amount = parse_attribute_ints(a);
		}
		else
			LOG(DEBUG) << "Ignoring attribute " << a.name() << " for node Im2Col/" << onnx_name << std::endl;
	}
}


/* Assign input tensors, resolve output tensor shapes, allocate output tensors */
void Im2Col::resolve(void)
{
	Tensor *input_1  = inputs[0];
	// Remember the parameters to the generated function,
	// along with a descriptive name that is used locally in the generated source.
	// The most "descriptive name" usually is the one this tensor has in the ONNX documentation.
	register_input(input_1, "A");

	if (inputs[0]->data_dim.size() != 4) {
		ERROR("Not Implemented for different input size");
	}
	// else leave input_2_optional as null so other functions here know to ignore it

	if (stride.size() == 0 || pad_amount.size() == 0 || dilations.size() == 0 || kernel_size.size() == 0 ) {
		ERROR("Not Implemented for input sizes not existent");
	}

	if (pad_amount[0] != pad_amount[1] || pad_amount[1] != pad_amount[2] || pad_amount[2] != pad_amount[3] || pad_amount[3] != pad_amount[0]   )
	{
		ERROR("Not Implemented for different padding");
	}
	

	if ( dilations[0] != 1 ||  dilations[1] != 1  )
	{
		ERROR("Not Implemented for different dilations");
	}

	/* Create output tensors.
	 * Set data dimensions and data type for the created tensors. */
	Tensor *t = new Tensor;
	// batch
	t->data_dim.push_back(1);
	// channel :(ifm_dim + total_pad - dilation * (k - 1) - 1) / stride) + 1
	t->data_dim.push_back((input_1->data_dim[1] + 2*pad_amount[0]  - kernel_size[0])/stride[0] + 1);
	// rows :(ifm_dim + total_pad - dilation * (k - 1) - 1) / stride) + 1
	t->data_dim.push_back((input_1->data_dim[2] + 2*pad_amount[0] - kernel_size[1])/stride[1] + 1);
	// collumns : kernel_rows * kernel_height * channels 
	t->data_dim.push_back(kernel_size[1]*kernel_size[0]*input_1->data_dim[3]);
	t->data_type = onnx::TensorProto_DataType_FLOAT;
	register_output(t, "Y");

	/* TODO: optional outputs? */
}


/* Body of the node implementing function */
void Im2Col::print(std::ostream &dst) const
{
	Tensor *A = inputs[0];
	INDT_1 << "/* Im2Col */" << std::endl;
	INDT_1 << "/* Operation needed to change Convolution to GEMM */" << std::endl;
    int32_t input_channels = A->data_dim[1];
	int32_t input_rows 	   = A->data_dim[2];
	int32_t input_column  = A->data_dim[3];
	int32_t pads_h = pad_amount[0] + pad_amount[2];
	int32_t pads_w = pad_amount[1] + pad_amount[3];
	INDT_1 << "/* First Transpose */" << std::endl;
	/* Genereate the C code here */
	INDT_1 << "float first_transpose[1]["<<input_column << "][" << input_channels<< "]["<< input_rows<< "];" << std::endl;
	INDT_1 << "for( uint32_t chan=0; chan<" << input_channels << "; chan++) {" << std::endl;
	INDT_2 << "for( uint32_t r=0; r<" << input_rows << "; r++)" << std::endl;
	INDT_3 << "for( uint32_t c=0; c<" << input_column << "; c++) {" << std::endl;
	INDT_4 << "first_transpose[0][c][chan][r]=A[0][chan][r][c];" << std::endl;
	INDT_3 << "}"<< std::endl;
	INDT_1 << "}"<< std::endl;
	/*  https://github.com/pjreddie/darknet/blob/master/src/im2col.c   */
	INDT_1 << "\n/* Actual Im2Col */" << std::endl;
	int32_t col_channels = kernel_size[1]*kernel_size[0]*A->data_dim[3];
	int32_t col_rows = (A->data_dim[1] + pads_h - kernel_size[0])/stride[0] + 1;
	int32_t col_columns = (A->data_dim[2] + pads_w - kernel_size[1])/stride[1] + 1; 
	INDT_1 << "float after_im_col["<<  col_channels << "][" << col_rows*col_columns << "];" << std::endl;
	
	INDT_1 << "for( uint32_t chan=0; chan<" << col_channels << "; chan++) {" << std::endl;
	INDT_2 << "uint32_t w_offset = chan%" << kernel_size[0] << ";" <<std::endl;
	INDT_2 << "uint32_t h_offset = (chan/"<< kernel_size[0] <<")%" << kernel_size[1] << ";" <<std::endl;
	INDT_2 << "uint32_t input_channel = (chan/"<< kernel_size[0] <<")/" << kernel_size[1] << ";" <<std::endl;
	INDT_2 << "uint32_t index_after = 0" << ";" <<std::endl;
	INDT_2 << "for( uint32_t r=0; r<" << col_rows << "; r++){" << std::endl;
	INDT_3 << "for( uint32_t c=0; c<" << col_columns << "; c++) {" << std::endl;
	/* Add Stride if needed */
	INDT_4 << "uint32_t input_column = h_offset + r * "<< stride[0]  << " - " << pad_amount[0] << ";" << std::endl;
	INDT_4 << "uint32_t input_row    = w_offset + c * "<< stride[1] << " - " << pad_amount[1] << ";" << std::endl;
	INDT_4 << "if (input_channel>"<< input_column<<"||input_channel<0||input_column>="<< input_channels <<"||input_column<0||input_row>="<< input_rows<< "||input_row<0){" << std::endl;
	INDT_5 << " after_im_col[chan][index_after] =0.0;" << std::endl;
	INDT_4 << "}else" << std::endl;
	INDT_5 << "after_im_col[chan][index_after]=first_transpose[0][input_channel][input_column][input_row];" << std::endl;
	INDT_4 << "index_after++;" << std::endl;
	INDT_3 << "}"<< std::endl;
	INDT_2 << "}"<< std::endl;
	INDT_1 << "}"<< std::endl;
	INDT_1 << "/* First Reshape */" << std::endl;

	INDT_1 << "float intermediate_output[1]["<<  input_column << "][" << kernel_size[0] << "][" << kernel_size[1]<<"]["<<col_rows<<"]["<<col_columns <<"];" << std::endl;
	INDT_1 <<"float " << " *data_ptr = (float*)after_im_col;" << std::endl;
	INDT_1 <<"float " << " *reshaped_ptr = (float*)intermediate_output;" << std::endl;
	INDT_1 << "for( uint32_t i=0; i<" << col_channels*col_rows*col_columns<< "; i++ )" << std::endl;
	INDT_2 << "reshaped_ptr[i] = data_ptr[i];" << std::endl;
	INDT_1 << std::endl;

	INDT_1 << "\n/* Second Transpose */" << std::endl;
	INDT_1 << "float intermediate_output_second[1]["<<  col_rows << "][" << col_columns << "][" << kernel_size[0]<<"]["<<kernel_size[1]<<"]["<< input_column <<"];" << std::endl;
	INDT_1 << "for( uint32_t i1=0; i1<" << input_column << "; i1++)" << std::endl;
	INDT_2 << "for( uint32_t i2=0; i2<" << kernel_size[0] << "; i2++)" << std::endl;
	INDT_3 << "for( uint32_t i3=0; i3<" << kernel_size[1] << "; i3++)" << std::endl;
	INDT_4 << "for( uint32_t i4=0; i4<" << col_rows << "; i4++)" << std::endl;
	INDT_5 << "for( uint32_t i5=0; i5<" << col_columns << "; i5++)" << std::endl;
	INDT(6) << "intermediate_output_second[0][i4][i5][i2][i3][i1]=intermediate_output[0][i1][i2][i3][i4][i5];" << std::endl;

	INDT_1 << "/* Second Reshape */" << std::endl;

	INDT_1 <<"float " << " *data_ptr_1 = (float*)intermediate_output_second;" << std::endl;
	INDT_1 <<"float " << " *reshaped_ptr_1 = (float*)Y;" << std::endl;
	INDT_1 << "for( uint32_t i=0; i<" << col_channels*col_rows*col_columns<< "; i++ )" << std::endl;
	INDT_2 << "reshaped_ptr_1[i] = data_ptr_1[i];" << std::endl;
	INDT_1 << std::endl;
}


} // namespace

