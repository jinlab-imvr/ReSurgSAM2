# 读取checkpoint，然后仅保存'model'部分
import torch


input_checkpoint = "hiera_small_ref18.pt"
output_checkpoint = "hiera_small_ref18_new.pt"
checkpoint = torch.load(input_checkpoint, map_location="cpu")
# save only the 'model' part
state_dict = {'model': checkpoint['model']}
torch.save(state_dict, output_checkpoint)
print(f"Saved model state dict to {output_checkpoint}")