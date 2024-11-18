import csv
import json
import pathlib
from collections import defaultdict
from tabulate import tabulate
import pandas as pd
# import json
# with open('people_wiki_map_index_to_word.json', 'r') as f:
#     data = json.load(f)

data_dict = defaultdict(list)

# Read the JSON file into a pandas DataFrame
for json_file in pathlib.Path('/mnt/hof626/PycharmProjects/CompCLIP/output/sugarcrepe').glob('*.json') :
    with open(json_file) as f:
        data = json.load(f)
        data_dict['model'].append(json_file.stem)
        for key in data.keys():
            data_dict[key].append(data[key])


#fine_tuned_decoder_laion400m_coglaion - medcaps
#/mnt/nsingh/project_multimodal/clip-finetune/CompCLIP/CLIP_VITB16_laioncog_mlm_False_jt_2_loss_dt_15_5_9_35_lr_0.00001_bs_256_ep_2/checkpoint.pt - 2epoch warm0.3
data_dict = [
{'model': 'original-laion', 'add_obj': 0.8612997090203686, 'add_att': 0.7846820809248555, 'replace_obj': 0.9394673123486683, 'replace_att': 0.8248730964467005, 'replace_rel': 0.6763869132290184, 'swap_obj': 0.6081632653061224, 'swap_att': 0.6651651651651652},
{'model': '2ep-Mid0.3','add_obj': 0.7866149369544132, 'add_att': 0.8222543352601156, 'replace_obj': 0.9424939467312349, 'replace_att': 0.8730964467005076, 'replace_rel': 0.6834992887624467, 'swap_obj': 0.6244897959183674, 'swap_att': 0.6951951951951952},
{'model': '2ep-Mid0.8','add_obj': 0.8001939864209505, 'add_att': 0.8222543352601156, 'replace_obj': 0.950363196125908, 'replace_att': 0.8629441624365483, 'replace_rel': 0.6827880512091038, 'swap_obj': 0.6285714285714286, 'swap_att': 0.6816816816816816},
{'model': '4ep-Mid1.5','add_obj': 0.7987390882638216, 'add_att': 0.8222543352601156, 'replace_obj': 0.9455205811138014, 'replace_att': 0.8578680203045685, 'replace_rel': 0.6891891891891891, 'swap_obj': 0.6612244897959184, 'swap_att': 0.6816816816816816},
{'model': '2ep-Short0.8','add_obj': 0.8637245392822502, 'add_att': 0.8034682080924855, 'replace_obj': 0.9443099273607748, 'replace_att': 0.8426395939086294, 'replace_rel': 0.7119487908961594, 'swap_obj': 0.6693877551020408, 'swap_att': 0.6681681681681682},
{'model': '4ep-Short1.5','add_obj': 0.7987390882638216, 'add_att': 0.8222543352601156, 'replace_obj': 0.9455205811138014, 'replace_att': 0.8578680203045685, 'replace_rel': 0.6891891891891891, 'swap_obj': 0.6612244897959184, 'swap_att': 0.6816816816816816}]

table = tabulate(data_dict, headers='keys', tablefmt='github')
# table = table.replace('|', ',')
# save the table into a file
# with open('/mnt/hof626/PycharmProjects/CompCLIP/output/sugarcrepe/summary.csv', 'w') as f:
#     f.write(table)

print(table)
# df = pd.read_json('/mnt/hof626/PycharmProjects/CompCLIP/output/sugarcrepe/ViT-B-16-laion400m_e32.json')



# Write the DataFrame to a CSV file
# df.to_csv('/mnt/hof626/PycharmProjects/CompCLIP/output/sugarcrepe/csv_ViT-B-16-laion400m_e32.csv', index=False)