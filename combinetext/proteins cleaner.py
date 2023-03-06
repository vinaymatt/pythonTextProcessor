import re

def split_and_combine(input_string):
    # Find words before and after 'and'
    split_regex = re.compile(r'(\b\w+\b)(?:\s*and\s*)(\b\w+\b)')
    matches = split_regex.findall(input_string)

    # Check if input follows the pattern
    if matches:
        output_parts = []
        for prefix1, prefix2 in matches:
            common_words = set(input_string.split()) - {prefix1, prefix2, 'and'}
            common = ' '.join(common_words)

            # Check for special case
            special_regex = re.compile(r'(\b\w+\b\s+\b\w+\b\s+and\s+\b\w+\b\s+\b\w+\b)')
            special_match = special_regex.search(input_string)

            if special_match:
                input_string = input_string.replace(' and ', ', ')
                result = input_string
                break

            output_parts.append(f'{common} {prefix1}')
            input_string = input_string.replace(f'{prefix1} and {prefix2}', '')
        else:
            output_parts.append(f'{common} {prefix2}')
            result = ', '.join(output_parts).strip()
    else:
        result = input_string.replace(' and ', ', ')

    return result


# Test cases
with open('/Users/vinay/PycharmProjects/pythonTextProcessor/combinetext/cleanedproteinforword2vecfinal.txt', 'r') as file:
    test_cases = file.readlines()


for test in test_cases:
    print(split_and_combine(test))
