import onnxruntime
import argparse

def main(onnx_path):

    # Load the ONNX model
    session = onnxruntime.InferenceSession(onnx_path)

    # Get inputs and outputs
    inputs = session.get_inputs()
    outputs = session.get_outputs()

    # Print inputs
    print("")
    print("Inputs:")
    for input in inputs:
        print("Name:", input.name)
        print("Shape:", input.shape)
        print("Type:", input.type)

    # Print outputs
    print("")
    print("Outputs:")
    for output in outputs:
        print("Name:", output.name)
        print("Shape:", output.shape)
        print("Type:", output.type)
    print("")

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description='Prints the inputs and outputs of an ONNX model.')
    parser.add_argument('onnx_path', type=str, help='Choose an ONNX model')

    args = parser.parse_args()

    main(args.onnx_path)