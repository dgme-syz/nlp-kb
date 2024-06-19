import os, json, shutil
import gradio as gr
from config import gcn_data_root, schema, ner_model

from data.llm import llm
from bin.interface import InterFace
from bin.neo_graph import Neo

Infer = InterFace()
import nltk
nltk.download('punkt')
    
app = gr.Blocks(title='App',
    theme=gr.themes.Soft(primary_hue='orange', secondary_hue="blue"))

def llama_process_data(data: str, step, prompt: str):
    # 使用 llama 处理数据，返回到控件 llama_data
    llama_data = llm(prompt, data, int(step))
    return llama_data
def ere_demo(text: str, method: str, pretrained_model: str, ner_model: str, rebel_model: str):
    # 使用 ere 推理，返回到控件 out_text
    # 创建临时输出文件
    # text 起初多余的特殊符号
    text = text.replace('\n', ' ').replace('\r', ' ').replace('\t', ' ')
    if method == 'bert-large-uncased(tuning)':
        # 使用 os 调用 bin/extract.py
        os.system(f'python -m bin.extract --sentence "{text}" --output_dir ./tmp --model_path {pretrained_model} --ner_path {ner_model}')
    else:
        # 使用 os 调用 bin/rebel-extract.py
        os.system(f'python -m bin.rebel-extract --sentence "{text}" --output_dir ./tmp --model_path {rebel_model}')
    # 读取输出文件
    out_text = ''
    if os.path.exists('./tmp/train_gcn.txt'):
        with open('./tmp/train_gcn.txt', 'r', encoding='utf-8') as f:
            out_text = f.read()
    if os.path.exists('./tmp/val_gcn.txt'):
        with open('./tmp/val_gcn.txt', 'r', encoding='utf-8') as f:
            out_text += f.read()
    if os.path.exists('./tmp'):
        shutil.rmtree('./tmp')
    return out_text

def convert_data(dir: str) -> None:
    try:
        os.system(f'python -m data.ere_data.data_utils --data_dir {dir}')
    except Exception as e:
        gr.Error(f'数据处理失败: {e}')
        return
    gr.Info('数据处理完成')

def train_ere_fn(epochs, batch_size, save_steps, dir, pretrained_model, relations) -> None:
    # relations 写入文件
    schema = relations
    try:
        os.system(f'python -m bin.ere --epoch {int(epochs)} --batch_size {int(batch_size)} --save_steps {int(save_steps)} --data_dir {dir} --pretrained_model {pretrained_model}')
    except Exception as e:
        gr.Error(f'训练失败: {e}')
        return
    gr.Info('训练完成')

def ere_file(data_path: str, method: str, out_dir: str, pretrained_model: str, ner_model:str, rebel_model) -> str:
    try:
        if method == 'bert-large-uncased(tuning)':
            # 使用 os 调用 bin/extract.py
            print(ner_model)
            os.system(f'python -m bin.extract --article "{data_path}" --output_dir {out_dir} --model_path {pretrained_model} --ner_path {ner_model}')
        else:
            # 使用 os 调用 bin/rebel-extract.py
            os.system(f'python -m bin.rebel-extract --article "{data_path}" --output_dir {out_dir} --model_path {rebel_model}')
        # 发送消息提示操作成功
        part_of_out = ''
        with open(os.path.join(out_dir, "train_gcn.txt"), 'r', encoding='utf-8') as f:
            part_of_out += "".join(f.readlines()[:5])
        return part_of_out
    except Exception as e:
        gr.Error(f'数据处理失败: {e}')
        return ''

def update_visibility(selected_method):
    if selected_method == 'bert-large-uncased(tuning)':
        return [gr.Dropdown(visible=True), gr.Dropdown(visible=True), gr.Dropdown(visible=False)]
    else:
        return [gr.Dropdown(visible=False), gr.Dropdown(visible=False), gr.Dropdown(visible=True)]

def get_directories_in_folder(folder_path):
    try:
        return [os.path.join(folder_path, d) for d in os.listdir(folder_path) if os.path.isdir(os.path.join(folder_path, d))]
    except FileNotFoundError:
        return []
li = get_directories_in_folder(".\\results")

def split_fn(dir: str) -> None:
    try:
        os.system(f'python -m data.gcn_data.data_utils --data_dir {dir}')
    except Exception as e:
        gr.Error(f'数据处理失败: {e}')
        return
    gr.Info('数据划分完成')

def train_rgcn(epochs, batch_size, eval_step, dir):
    try:
        os.system(f'python -m bin.rgcn --epoch {int(epochs)} --batch_size {int(batch_size)} --eval_step {int(eval_step)} --data_path {dir}')
    except Exception as e:
        gr.Error(f'训练失败: {e}')
        return
    gr.Info('训练完成')

def infer1_fn(entity1: str, entity2: str, threshold: float, data: str):
    try:
        Infer.update(data)
        ans = Infer.use_2entitys_to_get_relation(entity1, entity2, threshold)
        di, tot = {}, 1e-7
        for i in ans:
            tot += i[1]
        for i in ans:
            di[i[0][1]] = i[1] / tot
    except Exception as e:
        gr.Error(f'推理失败: {e}')
        return {}
    gr.Info('推理完成')
    return di
def infer2_fn(entity1: str, relation: str, lim_edge_1: int, threshold: float, data: str):
    try:
        Infer.update(data)
        ans = Infer.use_entity1_plus_relation_to_get_relation(entity1, relation, threshold, lim_edge_1)
        di, tot = {}, 1e-7
        for i in ans:
            tot += i[1]
        for i in ans:
            di[i[0][2]] = i[1] / tot
    except Exception as e:
        gr.Error(f'推理失败: {e}')
        return {}

    gr.Info('推理完成')
    return di
def infer3_fn(relation: str, entity2: str, lim_edge_2: int, threshold: float, data: str):
    try:
        Infer.update(data)
        ans = Infer.use_entity2_plus_relation_to_get_relation(entity2, relation, threshold, lim_edge_2)
        di, tot = {}, 1e-7
        for i in ans:
            tot += i[1]
        for i in ans:
            di[i[0][0]] = i[1] / tot
    except Exception as e:
        gr.Error(f'推理失败: {e}')
        return {}
    gr.Info('推理完成')
    return di

def enjoy_neo(txt_path: str):
    try:
        # 解析三元组生成指令
        os.system(f'python -m bin.neo --input_dir {txt_path}')
        gr.Info('解析三元组完成，你能在 ./neos/ 文件夹下找到运行指令')
    except Exception as e:
        gr.Error(f'生成失败: {e}')
        return gr.Markdown('生成失败')

with app:
    gr.Markdown(value="""# 知识表示 课设
        **Author**:[DGMEFG](https://github.com/DGMEFG) """)
    
    with gr.Tabs():
        with gr.Tab(label='How to start'):
            gr.Markdown("### 1. Python环境(建议使用 conda 创建 Python>=3.10 的环境)")
            gr.Markdown("```shell pip install -r requirements.txt ```")
            gr.Markdown("值得注意的是，gradio需要使用较新版本，老版本会报一类似，递归深度过大 (3.xx.x 版本) 的错误")
            gr.Markdown("### 2. 关于模型")
            gr.Markdown("秉持着一键式的初心，这里我并没花太多精力编写下载模型的脚本，因为如果上网姿势正确，运行代码应该是能一键到底，这里提示几个需要注意的地方")
            gr.Markdown("* nltk下载punkt可能会失败，因为上网姿势不太正确")
            gr.Markdown("* 本地模型的路径需要自己输入，如./models/bert-large-ner','./models/rebel")
        with gr.Tab(label='数据预处理'):
            with gr.Row():
                raw_data = gr.Textbox(label='原始数据', type='text', lines=10)
                llama_data = gr.Textbox(label='LLAMA数据', value='wait for processing', type='text', lines=10)
            with gr.Row():
                step = gr.Number(label='处理批次大小', value=1, interactive=True)
                prompt = gr.Textbox(label='Prompt', 
                    value='Please Output only the answer, remove unnecessary spaces, and avoid any other contents as much as possible.Use the simplest vocabulary.Make the following text easier to extract relationships from, and avoid using pronouns. Ideally, use only one form of words with the same meaning', lines=3,
                    interactive=True)
            run1 = gr.Button()
            run1.click(llama_process_data, inputs=[raw_data, step, prompt], outputs=[llama_data])
        with gr.Tab(label='实体关系抽取'):
            gr.Markdown("TIP: 如果有本地模型,可以自己输入地址,如./models/bert-large-ner','./models/rebel")
            # 推理
            gr.Markdown("### 测试")
            gr.Markdown("#### DEMO")
            with gr.Row():
                demo_text = gr.Textbox(label='文本', type='text', lines=10)
                with gr.Column():
                    out_text = gr.Textbox(label='输出', value='wait for processing', type='text', lines=10)
                with gr.Column():
                    ere_merthod_demo = gr.Radio(label='关系抽取方法', choices=['bert-large-uncased(tuning)', 'rebel'],
                            value='rebel', interactive=True)
                    ner_model_demo = gr.Dropdown(label='NER模型', choices=[ner_model], 
                            visible=False, allow_custom_value=True, interactive=True)
                    pretrained_model_demo = gr.Dropdown(label='选择预训练模型', choices=li, visible=False, allow_custom_value=True)
                    rebel_model_demo = gr.Dropdown(label='选择rebel模型', choices=["Babelscape/rebel-large"], 
                            visible=True, allow_custom_value=True, interactive=True)
                    ere_merthod_demo.change(update_visibility, inputs=[ere_merthod_demo], 
                        outputs=[ner_model_demo, pretrained_model_demo, rebel_model_demo])
                    start_demo = gr.Button()
            start_demo.click(ere_demo, inputs=[demo_text, ere_merthod_demo, pretrained_model_demo, ner_model_demo, rebel_model_demo], outputs=[out_text])
            gr.Markdown("#### 从文件中提取")
            with gr.Row():
                data_path = gr.Textbox(label='文件路径', type='text', value='.\\data\\raw.txt', interactive=True)
                part_of_out = gr.Textbox(label='部分输出', value='wait for processing', type='text', lines=10)
                with gr.Column():
                    ere_merthod = gr.Radio(label='关系抽取方法', choices=['bert-large-uncased(tuning)', 'rebel'], 
                            value='rebel', interactive=True)
                    ner_model_txt = gr.Dropdown(label='NER模型', choices=[ner_model], 
                            visible=False, allow_custom_value=True, interactive=True)
                    pretrained_models = gr.Dropdown(label='选择预训练模型', choices=li, visible=False, allow_custom_value=True)
                    rebel_model_txt = gr.Dropdown(label='选择rebel模型', choices=["Babelscape/rebel-large"], 
                            visible=True, allow_custom_value=True, interactive=True)
                    ere_merthod.change(update_visibility, inputs=ere_merthod, 
                        outputs=[ner_model_txt, pretrained_models, rebel_model_txt])
                    out_dir = gr.Textbox(label='输出路径', type='text', value=gcn_data_root, interactive=True)
                    run2 = gr.Button()
            run2.click(ere_file, inputs=[data_path, ere_merthod, out_dir, pretrained_models, ner_model_txt, rebel_model_txt], outputs=[part_of_out])
            # 训练
            gr.Markdown("### 训练")
            with gr.Row():
                # 部分参数
                with gr.Column():
                    ere_dir = gr.Textbox(label='数据存放目录所在文件夹', type='text', value='./data/ere_data/tacred', interactive=True, lines=2)
                    gr.Markdown('🚀处理数据集 train.txt, val.txt 为 train_ner.txt, val_ner.txt, 存储在同一目录下')
                    convert = gr.Button()
                convert.click(fn=convert_data, inputs=[ere_dir])
                with gr.Column():
                    batch_size = gr.Number(label='batch_size', interactive=True, value=8)
                    save_steps = gr.Number(label='save_steps', interactive=True, value=10)
                    epochs = gr.Number(label='epochs', interactive=True, value=3)
                    pretrained_model = gr.Dropdown(label='预训练模型', choices=['bert-base-uncased', 'bert-large-uncased'])
                    relations = gr.Code(label='关系', language='json', interactive=True, value=json.dumps(schema), lines=1)
                    train_ere = gr.Button()
                train_ere.click(fn=train_ere_fn, inputs=[epochs, batch_size, save_steps, ere_dir, pretrained_model, relations])
        with gr.Tab(label='简单知识推理'):
            gr.Markdown("### 训练R-GCN(两层)")
            with gr.Row():
                with gr.Column():
                    dir = gr.Textbox(label='数据存放目录所在文件夹', type='text', value=gcn_data_root, interactive=True,
                                lines=6)
                    gr.Markdown('🚀划分数据集得到train.txt, val.txt, entities.json, relations.json')
                    split = gr.Button()
                with gr.Column():
                    gr.Markdown('✨训练R-GCN')
                    epochs = gr.Number(label='epochs', interactive=True, value=3)
                    batch_size = gr.Number(label='batch_size(训练时每次从图中选多少三元组)', interactive=True, value=8)
                    eval_step = gr.Number(label='eval_step(多少次训练后进行验证)', interactive=True, value=20)
                    train = gr.Button()
                split.click(fn=split_fn, inputs=[dir])
                train.click(fn=train_rgcn, inputs=[epochs, batch_size, eval_step, dir])
            
            gr.Markdown("### 知识推理")
            data_for_infer = gr.Textbox(label='数据存放目录所在文件夹', type='text', value=gcn_data_root, interactive=True, lines=1)
            with gr.Row():
                # 使用 entity1 entity2 -> relation
                with gr.Column():
                    gr.Markdown('#### 1. 实体1 + 实体2 -> 关系')
                    entity1 = gr.Textbox(label='实体1', type='text', interactive=True)
                    entity2 = gr.Textbox(label='实体2', type='text', interactive=True)
                    th1 = gr.Number(label='阈值', interactive=True, value=2.)
                    infer1 = gr.Button()
                    out1 = gr.Label(label='推理 -> 关系')
                infer1.click(infer1_fn, inputs=[entity1, entity2, th1, data_for_infer], outputs=[out1])
                # 使用 relation entity1 -> entity2
                with gr.Column():
                    gr.Markdown('#### 2. 关系 + 实体1 -> 实体2')
                    entity1 = gr.Textbox(label='实体1', type='text', interactive=True)
                    relation1 = gr.Textbox(label='关系', type='text', interactive=True)
                    lim_edge_1 = gr.Number(label='BFS深度', interactive=True, value=2)
                    th2 = gr.Number(label='阈值', interactive=True, value=2.)
                    infer2 = gr.Button()
                    out2 = gr.Label(label='推理 -> 实体2')
                infer2.click(infer2_fn, inputs=[entity1, relation1, lim_edge_1, th2, data_for_infer], outputs=[out2])
                with gr.Column():
                    gr.Markdown('#### 3. 实体2 + 关系 -> 实体1')
                    relation2 = gr.Textbox(label='关系', type='text', interactive=True)
                    entity2 = gr.Textbox(label='实体2', type='text', interactive=True)
                    lim_edge_2 = gr.Number(label='BFS深度', interactive=True, value=2)
                    th3 = gr.Number(label='阈值', interactive=True, value=2.)
                    infer3 = gr.Button()
                    out3 = gr.Label(label='推理 -> 实体1')
                infer3.click(infer3_fn, inputs=[relation2, entity2, lim_edge_2, th3, data_for_infer], outputs=[out3])
        with gr.Tab(label='知识存储 & 可视化'):
            with gr.Row():
                txt_dir = gr.Textbox(label='三元组存储文件路径', type='text', value='./data/gcn_data', interactive=True)
                encode = gr.Button(value='解析三元组生成知识图谱')
            encode.click(fn=enjoy_neo, inputs=[txt_dir])
if __name__ == '__main__':
    app.launch()
    
        