import time, torch, sys
from evo2 import Evo2
name = sys.argv[1] if len(sys.argv)>1 else "evo2_7b"
t0=time.time(); print(f"Loading {name} ...", flush=True)
model = Evo2(name)
print(f"Loaded in {time.time()-t0:.1f}s", flush=True)
# list submodule names (candidate layers)
sd_keys = list(model.model.state_dict().keys())
print("NUM state_dict keys:", len(sd_keys), flush=True)
import re
# print unique block submodule patterns
blocks = sorted({re.sub(r'blocks\.\d+', 'blocks.N', k) for k in sd_keys if k.startswith('blocks')})
for b in blocks: print("  KEY", b, flush=True)
# named_modules for hookable layers
mods = [n for n,_ in model.model.named_modules()]
print("NUM named_modules:", len(mods), flush=True)
for n in mods:
    if n.startswith('blocks.0.') or n in ('blocks.0','unembed','embedding_layer'): print("  MOD", n, flush=True)
# smoke: WT vs mut embedding delta at a layer
seq = "ACGT"*80
mut = seq[:160] + ("A" if seq[160]!="A" else "C") + seq[161:]
ids = torch.tensor(model.tokenizer.tokenize(seq),dtype=torch.int).unsqueeze(0).cuda()
mids= torch.tensor(model.tokenizer.tokenize(mut),dtype=torch.int).unsqueeze(0).cuda()
LN = ["blocks.24.mlp.l3","blocks.16.mlp.l3","blocks.28.mlp.l3"]
with torch.inference_mode():
    _,e1 = model(ids, return_embeddings=True, layer_names=LN)
    _,e2 = model(mids, return_embeddings=True, layer_names=LN)
for k in LN:
    d=(e2[k]-e1[k]).float()
    print(f"EMB {k} shape={tuple(e1[k].shape)} dtype={e1[k].dtype} delta@160 L2={d[0,160].norm().item():.4f} mean|delta|={d.abs().mean().item():.5f}", flush=True)
print("PEAK GB", torch.cuda.max_memory_allocated()/1e9, flush=True)
print("SMOKE_OK", flush=True)
