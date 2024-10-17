import json
import argparse
def main(input_file_path, output_file_path):
    with open(input_file_path, 'r') as file:
        data = json.load(file)

    for item in data:
        if 'conversations' in item:
            for convo in item['conversations']:
                if 'value' in convo:
                    value = convo['value']
                    if value.endswith('\n<image>'):
                        # Remove '\n<image>' from the end
                        value = value[:-8]
                        # Add '<image>\n' at the beginning
                        value = '<image>\n' + value
                        # Update the value in the item
                        convo['value'] = value

    with open(output_file_path, 'w') as file:
        json.dump(data, file, indent=4)

    print(f'Modified data has been saved to {output_file_path}')

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modify LLaVA dataset JSON file.")
    parser.add_argument("input_file", help="Path to the input JSON file")
    parser.add_argument("output_file", help="Path to save the modified JSON file")
    args = parser.parse_args()

    main(args.input_file, args.output_file)
