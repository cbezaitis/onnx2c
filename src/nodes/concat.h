/* This file is part of onnx2c.
 *
 * Concate ... concatenates a number of input tensors
 * across a given dimension.
 */

namespace toC {

	class Concat : public Node {
	public:
		Concat() {
			op_name = "Concat";
			axis = 1;
		}

		// attribute
		int axis;

		void parseAttributes(onnx::NodeProto &node) override {
			for (const auto &a : node.attribute()) {
				if (a.name() == "axis") {
					if (a.type() != onnx::AttributeProto_AttributeType_INT)
						ERROR("Bad attribute " << a.name());
					axis = a.i();
				} else
					ERROR("Unknown attribute " << a.name());
			}
		}

		void print(std::ostream &dst) const override {

			dst << "\t/* Concat */" << std::endl;
			const Tensor *concat_result = outputs[0];

			// the axisPitch is the number of elements to add to move to the next split axis in the concat_result
			int64_t axisPitch = 1;
			for (int i = concat_result->data_dim.size() - 1; i >= axis; i--) {
				axisPitch *= concat_result->data_dim[i];
			}

			dst << "\tint64_t outputOffset;" << std::endl;

			int64_t outputBase = 0;
			int64_t input_count = inputs.size();

			for (int64_t inputIndex = 0; inputIndex < input_count; inputIndex++) {

				std::string input_name = "input_";
				input_name += std::to_string(inputIndex);

				// the inputAxisPitch is the number of elements to add to move to the next split axis in the inputs
				int64_t inputAxisPitch = 1;
				for (int i = inputs[inputIndex]->data_dim.size() - 1; i >= axis; i--) {
					inputAxisPitch *= inputs[inputIndex]->data_dim[i];
				}

				int64_t inputSize = inputs[inputIndex]->data_num_elem();

				// copy the data across: for every 'inputAxisPitch' values copied, we move over by the 'axisPitch'
				dst << "\toutputOffset = " << outputBase << ";" << std::endl;
				dst << "\tfor (int64_t i = 0, j = 0; i < " << inputSize << "; i++) {" << std::endl;

				dst << "\t\t*((" << concat_result ->data_type_str() << "*)output";
				dst << " + (outputOffset + i)) = *((" << concat_result ->data_type_str() << "*)";
				dst << input_name << " + i);" << std::endl;

				dst << "\t\tif (++j == " << inputAxisPitch << ") {" << std::endl;
				dst << "\t\t\toutputOffset += (" << axisPitch - inputAxisPitch << ");" << std::endl;
				dst << "\t\t\tj = 0;" << std::endl;
				dst << "\t\t}" << std::endl;

				dst << "\t}" << std::endl;

				outputBase += inputAxisPitch;
			}

		}

		void resolve(void) override {
			if (inputs.size() == 1 ) {
				LOG(WARNING) << "Concat node " << onnx_name << " has only one input." << std::endl;
			}

			if (axis < 0)
				axis = inputs[0]->data_dim.size() + axis;

			auto *rv = new Tensor;
			rv->data_dim = inputs[0]->data_dim;
			size_t input_count = inputs.size();
			size_t output_axis_size = 0;
			size_t i, j;
			std::vector<int> dims = inputs[0]->data_dim;
			LOG(TRACE) << "Concatenating on axis " << axis << std::endl;
			for (i = 0; i < input_count; i++) {
				if( inputs[0]->rank() != inputs[i]->rank() ) {
					LOG(DEBUG) << "Input " << inputs[0]->name << " has " << inputs[0]->rank() << " dimensions" << std::endl;
					LOG(DEBUG) << "Input " << inputs[i]->name << " has " << inputs[i]->rank() << " dimensions" << std::endl;
					ERROR("Concat expects all inputs to have equal number of dimensions");
				}
				for (j = 0; j < dims.size(); j++) {
					if (dims[j] != inputs[i]->data_dim[j] && (int) j != axis)
						ERROR("Concat's input tensors must have the same shape, except for the "
							  "dimension size of the axis to concatenate on.");
				}

				std::string input_name = "input_";
				input_name += std::to_string(i);
				register_input(inputs[i], input_name);
				output_axis_size += inputs[i]->data_dim[axis];
			}
			rv->data_dim[axis] = output_axis_size;
			rv->data_type = inputs[0]->data_type;
			register_output(rv, "output");
		}
	};
}
