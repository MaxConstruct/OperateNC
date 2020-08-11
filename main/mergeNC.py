# import os
# import xarray as xr
#
# import main.util as util
# from hurry.filesize import size
#
# #%%
#
# for var in util.VARIABLES:
#     # print(var, '-------------------------------')
#     for lab in util.LABELS:
#         models_path = os.path.join(util.ROOT_DIR, var, lab)
#         models = os.listdir(models_path)
#         # print('\t|', lab)
#         for model_name in models:
#             root_file_path = os.path.join(models_path, model_name)
#             # file_name = os.listdir(root_file_path)[0]
#             # file_path = os.path.join(root_file_path, file_name)
#             nb = sum(os.path.getsize(os.path.join(root_file_path, f)) for f in os.listdir(root_file_path) if os.path.isfile(os.path.join(root_file_path, f)))
#             print("\t\t", '{:20s}'.format(model_name), '\t', '{:.0f}'.format(nb/(1<<20)))
#
