import torch
ckpt = torch.load('best_model.pth', map_location='cpu')
state = ckpt.get('model_state_dict', ckpt.get('model_state', ckpt))
keys = list(state.keys())
print(f"Total keys: {len(keys)}")
print("\nAll keys:")
for k in keys:
    print(k)