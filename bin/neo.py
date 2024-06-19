import os, argparse
import re

def replace_space_with_underscore(filename):
    with open(filename, 'r') as file:
        lines = file.readlines()

    processed_lines = []
    for line in lines:
        # 分割每一行的元素
        elements = line.strip().split('\t')
        if len(elements) >= 2:
            # 替换第二个元素中的空格为下划线
            elements[1] = elements[1].replace(' ', '_')
            # 重新组合元素并添加到处理后的列表
            processed_line = '\t'.join(elements)
            processed_lines.append(processed_line)

    # 返回处理后的所有行
    return processed_lines

# 关系处理
def process_relationships_in_input_files(input_dir):
    # 确保输入目录存在
    if not os.path.exists(input_dir):
        print(f"Input directory {input_dir} does not exist.")
        return

    # 遍历输入目录中的所有文件
    for filename in os.listdir(input_dir):
        if filename.endswith('.txt'):
            input_filepath = os.path.join(input_dir, filename)

            # 读取文件并修改内容
            with open(input_filepath, 'r', encoding='utf-8') as file:
                lines = file.readlines()  # 读取所有行

            # 创建一个新的列表来存储修改后的内容
            new_lines = []
            for line in lines:
                parts = line.strip().split("\t")
                if len(parts) == 3:
                    entity1, relationship, entity2 = parts

                    # 如果关系中包含冒号，删除冒号前的内容
                    if ':' in relationship:
                        relationship = relationship.split(':')[1]
                    if ' ' in relationship:
                        relationship = relationship.replace(' ', '_')
                    if '-' in relationship:
                        relationship = relationship.replace('-', '_')
                    if '/' in relationship:
                        relationship = relationship.replace('/', '_')
                    if '(' in relationship:
                        relationship = relationship.replace('(', '_')
                    if ')' in relationship:
                        relationship = relationship.replace(')', '_')
                    # 重新组合行并添加到新列表中
                    new_line = f"{entity1}\t{relationship}\t{entity2}\n"
                    new_lines.append(new_line)

            # 将修改后的内容写回到同一个文件中
            with open(input_filepath, 'w', encoding='utf-8') as file:
                file.writelines(new_lines)


def read_triples_and_create_neo4j_statements(input_dir, output_dir, filename):
    # 确保输出目录存在
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 文件的完整路径
    input_filepath = os.path.join(input_dir, filename)
    output_filepath = os.path.join(output_dir, filename.replace('.txt', '_Neo4j.txt'))

    # 存储实体和它们对应的编号
    unique_entities = {}
    created_entities = set()
    # 用于给实体编号
    entity_index = 1

    # 用于存储创建语句，同时去除重复
    unique_statements = set()
    # 读取文件并生成语句
    with open(input_filepath, 'r', encoding='utf-8') as file:
        for line in file:
            parts = line.strip().split("\t")
            if len(parts) == 3:
                entity1, relationship, entity2 = parts

                # 去除重复的实体并编号，同时确保每个实体只被创建一次
                # 为entity1创建节点
                unique_entities.setdefault(entity1, entity_index)
                if entity1 not in created_entities:
                    # 检查实体名称是否包含单引号，以决定使用单引号还是双引号
                    quote = '"' if "'" in entity1 else "'"
                    create_entity1_statement = f"CREATE (subj{unique_entities[entity1]}:Entity {{name: {quote}{entity1}{quote}}})"
                    unique_statements.add(create_entity1_statement)
                    created_entities.add(entity1)
                    entity_index += 1

                # 对entity2执行相同的操作
                unique_entities.setdefault(entity2, entity_index)
                if entity2 not in created_entities:
                    quote = '"' if "'" in entity2 else "'"
                    create_entity2_statement = f"CREATE (subj{unique_entities[entity2]}:Entity {{name: {quote}{entity2}{quote}}})"
                    unique_statements.add(create_entity2_statement)
                    created_entities.add(entity2)
                    entity_index += 1

                # 生成实体间关系的唯一语句
                relationship_statement = f"CREATE (subj{unique_entities[entity1]})-[:{relationship}]->(subj{unique_entities[entity2]})"
                if relationship_statement not in unique_statements:
                    unique_statements.add(relationship_statement)

    # 写入文件时，先写入所有创建节点的语句，然后写入所有创建关系的语句
    with open(output_filepath, 'w', encoding='utf-8') as output_file:
        # 先写入所有创建节点的语句
        for statement in unique_statements:
            if "CREATE (subj" in statement and ":Entity" in statement:
                output_file.write(f"{statement}\n")

        # 然后写入所有创建关系的语句
        for statement in unique_statements:
            if "CREATE (subj" in statement and "-[:" in statement:
                output_file.write(f"{statement}\n")
        # 在所有CREATE语句写入之后，追加RETURN语句
        return_statement = "RETURN "
        for entity_num in range(1, entity_index):
            return_statement += f"subj{entity_num}, "
        return_statement = return_statement.rstrip(", ") + "\n"
        output_file.write(return_statement)

def remove_duplicates_from_txt_files(input_folder):
    """
    遍历指定文件夹下的所有txt文件，去除每个文件中的重复行并保存到新文件中。

    :param input_folder: 字符串，包含txt文件的文件夹路径。
    """
    for filename in os.listdir(input_folder):
        if filename.endswith('.txt'):
            filepath = os.path.join(input_folder, filename)
            output_filepath = os.path.join(input_folder, filename)

            with open(filepath, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            unique_lines = list(dict.fromkeys(lines))

            with open(output_filepath, 'w', encoding='utf-8') as file:
                for line in unique_lines:
                    file.write(line)

def main(args):

    # 获取文件夹中所有的文件名
    for filename in ['train_gcn.txt', 'val_gcn.txt']:
            # 调用函数处理文件
            process_relationships_in_input_files(args.input_dir)
            # 调用函数处理文件
            read_triples_and_create_neo4j_statements(args.input_dir, args.output_dir, filename)
            remove_duplicates_from_txt_files(args.output_dir)

"""
问题：
生成的句柄中单引号
去重
按顺序生成实体

cypher自带关键词 by
不要出现空格
"""

if __name__ == '__main__':
    paser = argparse.ArgumentParser()
    paser.add_argument('--input_dir', type=str, default='./data/gcn_data', help='input directory')
    paser.add_argument('--output_dir', type=str, default='./neos/', help='output directory')
    args = paser.parse_args()
    main(args)