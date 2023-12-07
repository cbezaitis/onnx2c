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

class TopK : public Node {
	public:
	TopK() {
		op_name = "TopK";
	}
	/* Examples of ONNX Operand attributes */
	std::vector<float> a_floatarray_attribute;
	int axis;
	int largest;
	int sorted;

	// Mandatory "API" functions towards the rest of onnx2c
	virtual void parseAttributes( onnx::NodeProto &node ) override;
	virtual void resolve(void) override;
	virtual void print(std::ostream &dst) const override;
};


/* Parse attributes, if this node has them. */
void TopK::parseAttributes( onnx::NodeProto &node )
{
	for( const auto& a : node.attribute() ) {
		LOG(TRACE) << "Parsing attribute " << a.name() << std::endl;
		if( a.name() == "axis" )
			axis = parse_attribute_int(a);
		else if( a.name() == "largest" )
			largest = parse_attribute_int(a);
		else if( a.name() == "sorted" )
			sorted = parse_attribute_int(a);
		else
			LOG(ERROR) << "Ignoring attribute " << a.name() << " for node TopK/" << onnx_name << std::endl;
	}
}


/* Assign input tensors, resolve output tensor shapes, allocate output tensors */
void TopK::resolve(void)
{
	Tensor *input_1  = inputs[0];
	// Tensor *K  = NULL;
	// Remember the parameters to the generated function,
	// along with a descriptive name that is used locally in the generated source.
	// The most "descriptive name" usually is the one this tensor has in the ONNX documentation.
	register_input(input_1, "A");

	if (inputs.size() == 2) {
		Tensor *input_2_optional = inputs[1];
		register_input(input_2_optional, "K");
	}
	// else leave input_2_optional as null so other functions here know to ignore it


	/* Create output tensors.
	 * Set data dimensions and data type for the created tensors. */
	Tensor *t = new Tensor;
	t->data_dim.push_back(1);
	t->data_dim.push_back(inputs[1]->get_data_element(0));
	t->data_type = onnx::TensorProto_DataType_FLOAT;
	register_output(t, "values");

	Tensor *t1 = new Tensor;
	t1->data_dim.push_back(1);
	t1->data_dim.push_back(inputs[1]->get_data_element(0));
	t1->data_type = onnx::TensorProto_DataType_INT64;
	register_output(t1, "indices");
	/* TODO: optional outputs? */
}


/* Body of the node implementing function */
void TopK::print(std::ostream &dst) const
{
	Tensor *A = inputs[0];
	// Tensor *B = inputs[1];
	INDT_1 << "/* TopK */" << std::endl;
	INDT_1 << "/* Return the largest value */" << std::endl;
	int32_t rows = A->data_dim[1];

	if (largest != 1 )
	{
		ERROR("Not implemented for largest != 1");
	}

	
	/* Genereate the C code here */
	INDT_1 << "float max = -FLT_MAX; " << std::endl;
	INDT_1 << "uint64_t indice = -9; " << std::endl;
	INDT_1 << "for( uint32_t r=0; r <"<< rows <<"; r++ ) " << std::endl;
	INDT_2 << "if(A[0][r] > max ){" << std::endl;
	INDT_3 << "max = A[0][r];" << std::endl; 
	INDT_3 << "indice = r;" << std::endl; 
	INDT_2 << "}" << std::endl;
	INDT_1 << "values[0][0] = max;" << std::endl;
	INDT_1 << "indices[0][0] = indice;" << std::endl;
}


} // namespace

