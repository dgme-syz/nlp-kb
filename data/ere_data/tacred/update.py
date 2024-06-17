import json

def find_sublist_indices(main_list, sublist):
    """Find the start and end indices of sublist in main_list"""
    sublist_len = len(sublist)
    for i in range(len(main_list) - sublist_len + 1):
        if main_list[i:i + sublist_len] == sublist:
            return [i, i + sublist_len]
    return []

def update_positions(file_path):
    with open(file_path, 'r', encoding='utf-8') as file:
        lines = file.readlines()

    updated_data = []

    for line in lines:
        data = json.loads(line.strip())

        tokens = data['token']
        h_name = data['h']['name']
        t_name = data['t']['name']

        # Split h_name and t_name into tokens
        h_tokens = h_name.split()
        t_tokens = t_name.split()

        # Find positions of h_name and t_name in tokens
        new_h_pos = find_sublist_indices(tokens, h_tokens)
        new_t_pos = find_sublist_indices(tokens, t_tokens)

        if not new_h_pos:
            print(f"Error: '{h_name}' not found in tokens list")
        else:
            data['h']['pos'] = new_h_pos

        if not new_t_pos:
            print(f"Error: '{t_name}' not found in tokens list")
        else:
            data['t']['pos'] = new_t_pos

        updated_data.append(data)

    # Write the updated data back to a new file
    with open(file_path[:-4] + "_updated.txt", 'w', encoding='utf-8') as file:
        for item in updated_data:
            json.dump(item, file)
            file.write('\n')
# 调用函数并传入文件路径
update_positions('D:\\projects\\nlp\\nlp-kb\\data\\ere_data\\tacred\\mycorpus.txt')
