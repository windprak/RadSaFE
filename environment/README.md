# Apptainer recipes

Two images cover the entire pipeline.

| Recipe          | Used by         | Provides                                          |
| --------------- | --------------- | ------------------------------------------------- |
| `rag.def`       | Phase 2         | sentence-transformers, FAISS, BGE-large, tiktoken |
| `inference.def` | Phases 3, 4, 5  | vLLM, PyTorch, transformers, openai client, numpy |

Build:

```bash
apptainer build environment/rag.sif        environment/rag.def
apptainer build environment/inference.sif  environment/inference.def
```

Both can be built without root via `apptainer build --fakeroot ...`.

## Package list

- `PACKAGES.md` — curated overview of which packages are used where, plus
  pinned versions for both images.
- `inference_packages.lock.txt` — full `pip list` output from inside the
  built `inference.sif` (vLLM 0.19.0, torch 2.10.0+cu129, CUDA 12.9).
