"""End-to-end demo using llama-cpp-python (in-process inference) for feature generation."""
import numpy as np
from llama_cpp import Llama

from semaxis import LlamaCppClient, SupervisedTransformer

texts = [
    "This blender is amazing! It crushes ice perfectly and the motor is super powerful.",
    "Terrible product. Broke after two weeks. The plastic feels cheap.",
    "Decent blender for the price. Nothing fancy but gets the job done.",
    "I've had this for 3 years and it still works great. Very durable.",
    "Stopped working after one month. Customer service was unhelpful.",
    "Great value! Makes perfect smoothies every morning.",
    "The blades are sharp and it blends everything smoothly.",
    "Disappointed. The motor burned out after heavy use.",
    "Exactly what I needed. Simple to use and easy to clean.",
    "Horrible experience. Leaked from day one. Returning immediately.",
]

y = np.array([1, 0, 1, 1, 0, 1, 1, 0, 1, 0], dtype=float)

# Load a GGUF model — download from e.g. https://huggingface.co/TheBloke
llm = LlamaCppClient(Llama(model_path="path/to/model.gguf", n_ctx=4096))

vect = SupervisedTransformer(llm=llm, nli_model="cross-encoder/nli-deberta-v3-large")

print("=== Fit ===")
X = vect.fit_transform(texts, y)
print(f"  X shape: {X.shape}")

for f in vect.features_:
    print(f"  - {f.hypothesis}")
