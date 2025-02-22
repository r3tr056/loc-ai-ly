
#include <model_manager/gguf_loader.hpp>


std::string read_gguf_string(std::ifstream& file) {
	uint64_t len;
	file.read(reinterpret_cast<char*>(&len), sizeof(uint64_t));
	if (len > 1024 * 1024) {
		throw std::runtime_error("Error: GGUF string length too large, likely corrupted file.");
	}
	std::unique_ptr<char[]> buffer(new char[len]);
	file.read(buffer.get(), len);
	return std::string(buffer.get(), len);
}

void free_gguf_metadata_value(std::pair<gguf_metadata_value_t, gguf_metadata_value_type>& value) {
    if (value.second == GGUF_METADATA_VALUE_TYPE_STRING && value.first.string.string != nullptr) {
        delete[] value.first.string.string;
        value.first.string.string = nullptr;
    } else if (value.second == GGUF_METADATA_VALUE_TYPE_ARRAY && value.first.array.array != nullptr) {
        for (uint64_t i = 0; i < value.first.array.len; ++i) {
            std::pair<gguf_metadata_value_t, gguf_metadata_value_type> element_value = 
                { value.first.array.array[i], GGUF_METADATA_VALUE_TYPE_ARRAY };
            
            free_gguf_metadata_value(element_value);
        }
        delete[] value.first.array.array;
        value.first.array.array = nullptr;
    }
}

std::pair<gguf_metadata_value_t, gguf_metadata_value_type> read_gguf_metadata_value(std::ifstream& file) {
	gguf_metadata_value_type value_type;
	file.read(reinterpret_cast<char*>(&value_type), sizeof(gguf_metadata_value_type));
	
	gguf_metadata_value_t value;

	switch (value_type) {
		case GGUF_METADATA_VALUE_TYPE_UINT8:   file.read(reinterpret_cast<char*>(&value.uint8), sizeof(uint8_t)); break;

		case GGUF_METADATA_VALUE_TYPE_INT8:    file.read(reinterpret_cast<char*>(&value.int8), sizeof(int8_t)); break;

		case GGUF_METADATA_VALUE_TYPE_UINT16:  file.read(reinterpret_cast<char*>(&value.uint16), sizeof(uint16_t)); break;

		case GGUF_METADATA_VALUE_TYPE_INT16:   file.read(reinterpret_cast<char*>(&value.int16), sizeof(int16_t)); break;

		case GGUF_METADATA_VALUE_TYPE_UINT32:  file.read(reinterpret_cast<char*>(&value.uint32), sizeof(uint32_t)); break;

		case GGUF_METADATA_VALUE_TYPE_INT32:   file.read(reinterpret_cast<char*>(&value.int32), sizeof(int32_t)); break;

		case GGUF_METADATA_VALUE_TYPE_FLOAT32: file.read(reinterpret_cast<char*>(&value.float32), sizeof(float)); break;

		case GGUF_METADATA_VALUE_TYPE_BOOL:    file.read(reinterpret_cast<char*>(&value.bool_), sizeof(bool)); break;

		case GGUF_METADATA_VALUE_TYPE_STRING: {
			std::string str_val = read_gguf_string(file);
			value.string.len = str_val.length();
			value.string.string = new char[value.string.len + 1];   // +1 for null terminator
			std::strcpy(value.string.string, str_val.c_str());
			break;
		}

		case GGUF_METADATA_VALUE_TYPE_ARRAY: {
			file.read(reinterpret_cast<char*>(&value.array.type), sizeof(gguf_metadata_value_type));
			file.read(reinterpret_cast<char*>(&value.array.len), sizeof(uint64_t));
			if (value.array.len > 1024 * 1024) {
				throw std::runtime_error("Error: GGUF array length too large, likely corrupted file.");
			}
			value.array.array = new gguf_metadata_value_t[value.array.len];
			for (uint64_t i = 0; i < value.array.len; ++i) {
				auto [metadata_value, _] = read_gguf_metadata_value(file);
				value.array.array[i] = metadata_value;
			}
			break;
		}

		case GGUF_METADATA_VALUE_TYPE_UINT64:  file.read(reinterpret_cast<char*>(&value.uint64), sizeof(uint64_t)); break;

		case GGUF_METADATA_VALUE_TYPE_INT64:   file.read(reinterpret_cast<char*>(&value.int64), sizeof(int64_t)); break;

		case GGUF_METADATA_VALUE_TYPE_FLOAT64: file.read(reinterpret_cast<char*>(&value.float64), sizeof(double)); break;

		default:
			throw std::runtime_error("Error: Unknown GGUF metadata value type: " + std::to_string(value_type));
	}
	return {value, value_type};
}
