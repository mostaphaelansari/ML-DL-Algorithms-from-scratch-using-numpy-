import os

structure = {
    "utils": ["activations.py", "losses.py", "metrics.py", "initializers.py", "layers_base.py"],
    "ml": {
        "linear_regression": ["model.py", "train.py", "tests.py"],
        "logistic_regression": ["model.py", "train.py", "tests.py"],
        "knn": ["model.py", "train.py", "tests.py"],
        "decision_tree": ["model.py", "train.py", "tests.py"],
        "svm": ["model.py", "train.py", "tests.py"]
    },
    "dl": {
        "perceptron": ["model.py", "train.py"],
        "feedforward_nn": ["model.py", "train.py"],
        "cnn": ["model.py", "train.py"],
        "rnn": ["model.py", "train.py"],
        "transformer": ["attention.py", "encoder.py", "decoder.py", "transformer.py"]
    },
    "notebooks": [
        "01-linear-regression.ipynb", 
        "02-logistic-regression.ipynb"
    ],
    "tests": ["test_utils.py", "test_layers.py", "test_transformer.py"]
}

root_files = ["README.md", "requirements.txt", ".gitignore"]

def create_structure(base_path="."):
    for f in root_files:
        open(os.path.join(base_path, f), 'a').close()

    for key, value in structure.items():
        path = os.path.join(base_path, key)
        os.makedirs(path, exist_ok=True)
        if isinstance(value, list):
            for file in value:
                open(os.path.join(path, file), 'a').close()
        elif isinstance(value, dict):
            for subfolder, files in value.items():
                subpath = os.path.join(path, subfolder)
                os.makedirs(subpath, exist_ok=True)
                for file in files:
                    open(os.path.join(subpath, file), 'a').close()

if __name__ == "__main__":
    create_structure()
    print("Project skeleton created.")
